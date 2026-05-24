import hashlib
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from utils.dataset import ImageDataset


VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif")


def is_image_file(filename):
    """Comprova si el fitxer té una extensió d'imatge coneguda."""
    return filename.lower().endswith(VALID_EXTENSIONS)


def check_image_is_valid(image_path):
    """Detecta imatges corruptes quan volem fer una càrrega més estricta."""
    try:
        with Image.open(image_path) as image:
            image.verify()
        return True
    except Exception:
        return False


def compute_file_hash(file_path):
    """Hash simple per eliminar duplicats exactes si s'activa l'opció."""
    hasher = hashlib.md5()

    with open(file_path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(block)

    return hasher.hexdigest()


def resize_keep_aspect_short_side(image, image_size):
    """Manté aspect ratio i fa que el costat curt sigui image_size."""
    image = image.convert("RGB")
    width, height = image.size

    if width <= height:
        new_size = (image_size, int(height * image_size / width))
    else:
        new_size = (int(width * image_size / height), image_size)

    return image.resize(new_size, resample=Image.Resampling.LANCZOS)


def resize_with_padding(image, image_size):
    """Manté aspect ratio i omple fins a un quadrat."""
    image = image.convert("RGB")
    image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (image_size, image_size), color=(128, 128, 128))
    left = (image_size - image.width) // 2
    top = (image_size - image.height) // 2
    canvas.paste(image, (left, top))

    return canvas


def get_cached_image_path(source_root, cache_root, image_path):
    relative_path = os.path.relpath(image_path, source_root)
    class_name = os.path.dirname(relative_path)
    filename = os.path.basename(relative_path)
    stem, _ = os.path.splitext(filename)
    path_hash = hashlib.md5(relative_path.encode("utf-8")).hexdigest()[:10]

    return os.path.join(cache_root, class_name, f"{stem}_{path_hash}.jpg")


def resize_and_save_cached_image(args):
    (
        source_root,
        cache_root,
        image_path,
        image_size,
        force_rebuild,
        jpeg_quality,
        cache_resize_mode,
    ) = args

    cached_path = get_cached_image_path(source_root, cache_root, image_path)
    os.makedirs(os.path.dirname(cached_path), exist_ok=True)

    if os.path.exists(cached_path) and not force_rebuild:
        return "skipped_existing"

    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if cache_resize_mode == "square":
                image = image.resize(
                    (image_size, image_size),
                    resample=Image.Resampling.LANCZOS,
                )
            elif cache_resize_mode == "short_side":
                image = resize_keep_aspect_short_side(image, image_size)
            elif cache_resize_mode == "padding":
                image = resize_with_padding(image, image_size)
            else:
                raise ValueError(f"cache_resize_mode no reconegut: {cache_resize_mode}")

            image.save(cached_path, format="JPEG", quality=jpeg_quality, optimize=True)
    except Exception:
        return "skipped_corrupted"

    return "processed"


def prepare_resized_cache_dataset(
    source_root,
    cache_root,
    image_size=384,
    force_rebuild=False,
    jpeg_quality=90,
    num_workers=8,
    cache_resize_mode="square",
):
    """Crea un cache local redimensionat per accelerar l'entrenament."""
    os.makedirs(cache_root, exist_ok=True)
    marker_path = os.path.join(
        cache_root,
        f".cache_complete_{image_size}_{cache_resize_mode}.txt",
    )

    if os.path.exists(marker_path) and not force_rebuild:
        print(f"Cache redimensionat trobat: {cache_root}")
        return cache_root

    image_paths = []
    class_names = sorted(
        class_name
        for class_name in os.listdir(source_root)
        if os.path.isdir(os.path.join(source_root, class_name))
    )

    for class_name in class_names:
        class_dir = os.path.join(source_root, class_name)
        for filename in sorted(os.listdir(class_dir)):
            if is_image_file(filename):
                image_paths.append(os.path.join(class_dir, filename))

    print(
        f"Creant cache a {cache_root} "
        f"({len(image_paths)} imatges, image_size={image_size}, mode={cache_resize_mode})"
    )

    stats = {
        "processed": 0,
        "skipped_existing": 0,
        "skipped_corrupted": 0,
    }

    worker_args = [
        (
            source_root,
            cache_root,
            image_path,
            image_size,
            force_rebuild,
            jpeg_quality,
            cache_resize_mode,
        )
        for image_path in image_paths
    ]

    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(resize_and_save_cached_image, worker_args)
            for result in tqdm(
                results,
                total=len(worker_args),
                desc="Caching resized images",
                mininterval=1.0,
            ):
                stats[result] += 1
    else:
        for args in tqdm(worker_args, desc="Caching resized images", mininterval=1.0):
            result = resize_and_save_cached_image(args)
            stats[result] += 1

    with open(marker_path, "w", encoding="utf-8") as marker:
        marker.write(
            f"source_root={source_root}\n"
            f"image_size={image_size}\n"
            f"cache_resize_mode={cache_resize_mode}\n"
            f"processed={stats['processed']}\n"
            f"skipped_existing={stats['skipped_existing']}\n"
            f"skipped_corrupted={stats['skipped_corrupted']}\n"
        )

    print(
        "Cache acabat: "
        f"processed={stats['processed']}, "
        f"skipped_existing={stats['skipped_existing']}, "
        f"skipped_corrupted={stats['skipped_corrupted']}"
    )

    return cache_root


def load_wikiart_dataset(root_dir, remove_duplicates=True, check_corrupted=True):
    """Carrega WikiArt des de carpetes, una carpeta per classe."""
    image_paths = []
    labels = []
    class_to_idx = {}
    seen_hashes = set()

    stats = {
        "total_files_seen": 0,
        "valid_images": 0,
        "skipped_non_images": 0,
        "skipped_corrupted": 0,
        "skipped_duplicates": 0,
    }

    class_names = sorted(
        class_name
        for class_name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, class_name))
    )

    for class_idx, class_name in enumerate(class_names):
        class_to_idx[class_name] = class_idx
        class_dir = os.path.join(root_dir, class_name)

        for filename in sorted(os.listdir(class_dir)):
            stats["total_files_seen"] += 1

            if not is_image_file(filename):
                stats["skipped_non_images"] += 1
                continue

            image_path = os.path.join(class_dir, filename)

            if check_corrupted and not check_image_is_valid(image_path):
                stats["skipped_corrupted"] += 1
                continue

            if remove_duplicates:
                file_hash = compute_file_hash(image_path)
                if file_hash in seen_hashes:
                    stats["skipped_duplicates"] += 1
                    continue
                seen_hashes.add(file_hash)

            image_paths.append(image_path)
            labels.append(class_idx)
            stats["valid_images"] += 1

    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

    return image_paths, labels, class_to_idx, idx_to_class, stats


def get_class_distribution(labels, idx_to_class):
    counts = Counter(labels)
    distribution = [
        (idx_to_class[class_idx], count)
        for class_idx, count in counts.items()
    ]
    return sorted(distribution, key=lambda item: item[1], reverse=True)


def print_dataset_summary(image_paths, labels, class_to_idx, idx_to_class, stats):
    """Mostra una foto ràpida del dataset carregat."""
    distribution = get_class_distribution(labels, idx_to_class)

    print("\n========== DATASET SUMMARY ==========")
    print(f"Total files seen:      {stats['total_files_seen']}")
    print(f"Valid images:          {stats['valid_images']}")
    print(f"Skipped non-images:    {stats['skipped_non_images']}")
    print(f"Skipped corrupted:     {stats['skipped_corrupted']}")
    print(f"Skipped duplicates:    {stats['skipped_duplicates']}")
    print(f"Number of classes:     {len(class_to_idx)}")
    print(f"Number of image paths: {len(image_paths)}")
    print(f"Number of labels:      {len(labels)}")

    print("\nTop 10 largest classes:")
    for class_name, count in distribution[:10]:
        print(f"  {class_name}: {count}")

    print("\nTop 10 smallest classes:")
    for class_name, count in distribution[-10:]:
        print(f"  {class_name}: {count}")

    print("=====================================\n")


def split_dataset(image_paths, labels, val_size=0.15, test_size=0.15, random_state=42):
    """Split train/validation/test estratificat."""
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    val_relative_size = val_size / (1.0 - test_size)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=val_relative_size,
        random_state=random_state,
        stratify=train_val_labels,
    )

    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels


def print_split_summary(train_labels, val_labels, test_labels):
    print("\n========== SPLIT SUMMARY ==========")
    print(f"Train images: {len(train_labels)}")
    print(f"Val images:   {len(val_labels)}")
    print(f"Test images:  {len(test_labels)}")
    print("===================================\n")


def get_transforms(
    image_size=384,
    resize_images=True,
    use_augmentation=False,
    use_aspect_crop_cache=False,
):
    train_transforms = []
    val_test_transforms = []

    if use_aspect_crop_cache:
        train_transforms.append(transforms.RandomCrop(image_size))
        val_test_transforms.append(transforms.CenterCrop(image_size))
    elif resize_images:
        train_transforms.append(transforms.Resize((image_size, image_size)))
        val_test_transforms.append(transforms.Resize((image_size, image_size)))

    if use_augmentation:
        train_transforms.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.12,
                hue=0.02,
            ),
        ])

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transforms.extend([transforms.ToTensor(), normalize])
    val_test_transforms.extend([transforms.ToTensor(), normalize])

    return transforms.Compose(train_transforms), transforms.Compose(val_test_transforms)


def create_dataloaders(
    train_paths,
    val_paths,
    test_paths,
    train_labels,
    val_labels,
    test_labels,
    batch_size=32,
    image_size=384,
    num_workers=2,
    resize_images=True,
    use_augmentation=False,
    use_weighted_sampler=False,
    use_aspect_crop_cache=False,
):
    """Crea DataLoaders per train, validation i test."""
    train_transform, val_test_transform = get_transforms(
        image_size=image_size,
        resize_images=resize_images,
        use_augmentation=use_augmentation,
        use_aspect_crop_cache=use_aspect_crop_cache,
    )

    train_dataset = ImageDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = ImageDataset(val_paths, val_labels, transform=val_test_transform)
    test_dataset = ImageDataset(test_paths, test_labels, transform=val_test_transform)

    train_sampler = None
    train_shuffle = True
    if use_weighted_sampler:
        train_sampler = create_weighted_sampler(train_labels)
        train_shuffle = False

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs.update({
            "persistent_workers": True,
            "prefetch_factor": 4,
        })

    train_loader = DataLoader(
        train_dataset,
        shuffle=train_shuffle,
        sampler=train_sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


def filter_top_k_classes(image_paths, labels, idx_to_class, top_k=14):
    """Manté les classes amb més imatges i reindexa les labels."""
    counts = Counter(labels)
    top_labels = [label for label, _ in counts.most_common(top_k)]
    top_labels_set = set(top_labels)

    kept_class_names = [idx_to_class[label] for label in top_labels]
    new_class_to_idx = {
        class_name: new_idx
        for new_idx, class_name in enumerate(kept_class_names)
    }
    new_idx_to_class = {
        new_idx: class_name
        for class_name, new_idx in new_class_to_idx.items()
    }

    filtered_paths = []
    filtered_labels = []
    for image_path, old_label in zip(image_paths, labels):
        if old_label in top_labels_set:
            class_name = idx_to_class[old_label]
            filtered_paths.append(image_path)
            filtered_labels.append(new_class_to_idx[class_name])

    print(
        f"\nTop-{top_k}: {len(filtered_paths)} imatges, "
        f"{len(new_class_to_idx)} classes."
    )

    return filtered_paths, filtered_labels, new_class_to_idx, new_idx_to_class


def group_similar_classes(image_paths, labels, idx_to_class, class_groups):
    """Agrupa classes en labels noves quan s'activa explícitament."""
    class_to_group = {}
    for group_name, class_names in class_groups.items():
        for class_name in class_names:
            class_to_group[class_name] = group_name

    old_idx_to_grouped_class = {}
    final_class_names = []
    for old_label in sorted(idx_to_class.keys()):
        old_class_name = idx_to_class[old_label]
        grouped_class_name = class_to_group.get(old_class_name, old_class_name)
        old_idx_to_grouped_class[old_label] = grouped_class_name

        if grouped_class_name not in final_class_names:
            final_class_names.append(grouped_class_name)

    new_class_to_idx = {
        class_name: new_idx
        for new_idx, class_name in enumerate(final_class_names)
    }
    new_idx_to_class = {
        new_idx: class_name
        for class_name, new_idx in new_class_to_idx.items()
    }
    grouped_labels = [
        new_class_to_idx[old_idx_to_grouped_class[label]]
        for label in labels
    ]

    print("\n========== CLASS GROUPING ==========")
    for group_name, class_names in class_groups.items():
        print(f"  {group_name}: {', '.join(class_names)}")
    print(f"Nombre final de classes: {len(new_class_to_idx)}")
    print("====================================\n")

    return image_paths, grouped_labels, new_class_to_idx, new_idx_to_class


def compute_class_weights(labels, num_classes, idx_to_class=None):
    """Pesa més les classes minoritàries dins de CrossEntropyLoss."""
    counts = Counter(labels)
    weights = torch.zeros(num_classes, dtype=torch.float32)

    for class_idx in range(num_classes):
        weights[class_idx] = 1.0 / (counts[class_idx] ** 0.5)

    weights = weights / weights.sum() * num_classes

    print("\n========== CLASS WEIGHTS ==========")
    for class_idx in range(num_classes):
        class_name = idx_to_class[class_idx] if idx_to_class else str(class_idx)
        print(
            f"{class_name}: count={counts[class_idx]}, "
            f"weight={weights[class_idx]:.4f}"
        )
    print("===================================\n")

    return weights


def create_weighted_sampler(labels):
    """Sampler opcional per veure més sovint les classes petites."""
    counts = Counter(labels)
    sample_weights = torch.DoubleTensor([
        1.0 / counts[label]
        for label in labels
    ])

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
