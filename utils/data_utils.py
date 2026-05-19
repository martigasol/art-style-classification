import os
import hashlib
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from utils.dataset import ImageDataset


VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def is_image_file(filename):
    """
    Comprova si el fitxer té extensió d'imatge.
    """
    return filename.lower().endswith(VALID_EXTENSIONS)


def check_image_is_valid(image_path):
    """
    Comprova que la imatge no està corrupta.
    Pot trigar si el dataset és gran.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def compute_file_hash(file_path):
    """
    Calcula un hash per detectar duplicats exactes.
    Important per evitar fuga de dades entre train i test.
    """
    hasher = hashlib.md5()

    with open(file_path, "rb") as f:
        hasher.update(f.read())

    return hasher.hexdigest()


def get_cached_image_path(source_root, cache_root, image_path):
    """
    Retorna una ruta determinista dins el cache per una imatge original.
    """
    relative_path = os.path.relpath(image_path, source_root)
    class_name = os.path.dirname(relative_path)
    filename = os.path.basename(relative_path)
    stem, _ = os.path.splitext(filename)
    path_hash = hashlib.md5(relative_path.encode("utf-8")).hexdigest()[:10]

    return os.path.join(cache_root, class_name, f"{stem}_{path_hash}.jpg")


def resize_and_save_cached_image(args):
    source_root, cache_root, image_path, image_size, force_rebuild, jpeg_quality, cache_resize_mode = args

    cached_path = get_cached_image_path(source_root, cache_root, image_path)
    os.makedirs(os.path.dirname(cached_path), exist_ok=True)

    if os.path.exists(cached_path) and not force_rebuild:
        return "skipped_existing"

    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")

            if cache_resize_mode == "square":
                # EXP14:
                # Cache clàssic. Força totes les imatges a image_size x image_size.
                img = img.resize(
                    (image_size, image_size),
                    resample=Image.Resampling.LANCZOS,
                )

            elif cache_resize_mode == "short_side":
                # EXP19:
                # Manté aspect ratio. El costat curt passa a ser image_size.
                # El crop final es farà després al DataLoader.
                img = resize_keep_aspect_short_side(img, image_size)

            elif cache_resize_mode == "padding":
                # EXP17:
                # Manté aspect ratio i afegeix padding fins a image_size x image_size.
                img = resize_with_padding(img, image_size)

            else:
                raise ValueError(f"cache_resize_mode no reconegut: {cache_resize_mode}")

            img.save(cached_path, format="JPEG", quality=jpeg_quality, optimize=True)

    except Exception:
        return "skipped_corrupted"

    return "processed"


def resize_keep_aspect_short_side(img, image_size):
    """
    Redimensiona la imatge mantenint aspect ratio.
    El costat curt passa a ser image_size.
    El costat llarg queda variable.
    """
    img = img.convert("RGB")
    width, height = img.size

    if width <= height:
        new_width = image_size
        new_height = int(height * image_size / width)
    else:
        new_height = image_size
        new_width = int(width * image_size / height)

    img = img.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)

    return img

def resize_with_padding(img, image_size):
    """
    Redimensiona mantenint aspect ratio i afegeix padding fins a image_size x image_size.
    """
    img = img.convert("RGB")

    # Manté aspect ratio i posa la imatge dins d'un quadrat image_size x image_size.
    img.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (image_size, image_size), color=(128, 128, 128))

    left = (image_size - img.width) // 2
    top = (image_size - img.height) // 2

    canvas.paste(img, (left, top))

    return canvas

def prepare_resized_cache_dataset(
    source_root,
    cache_root,
    image_size=224,
    force_rebuild=False,
    jpeg_quality=90,
    num_workers=8,
    cache_resize_mode="square",
):
    """
    Crea un cache local amb les imatges ja redimensionades.

    Això fa que cada epoch no hagi de llegir les imatges originals grans
    ni executar Resize sobre totes les imatges.
    """
    os.makedirs(cache_root, exist_ok=True)
    marker_path = os.path.join(
        cache_root,
        f".cache_complete_{image_size}_{cache_resize_mode}.txt"
    )

    if os.path.exists(marker_path) and not force_rebuild:
        print(f"Cache redimensionat trobat: {cache_root}")
        return cache_root

    image_paths = []
    class_names = sorted([
        class_name
        for class_name in os.listdir(source_root)
        if os.path.isdir(os.path.join(source_root, class_name))
    ])

    for class_name in class_names:
        class_dir = os.path.join(source_root, class_name)

        for filename in sorted(os.listdir(class_dir)):
            if is_image_file(filename):
                image_paths.append(os.path.join(class_dir, filename))

    print(
        f"Creant cache redimensionat a {cache_root} "
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


def load_wikiart_dataset(root_dir, remove_duplicates=True, check_corrupted=True,exclude_paths_file=None):
    """
    Carrega un dataset organitzat en carpetes per classe.

    Exemple:
        root_dir/
            Impressionism/
                img1.jpg
            Cubism/
                img2.jpg

    Retorna:
        image_paths
        labels
        class_to_idx
        idx_to_class
        stats
    """

    image_paths = []
    labels = []
    class_to_idx = {}

    stats = {
        "total_files_seen": 0,
        "valid_images": 0,
        "skipped_non_images": 0,
        "skipped_corrupted": 0,
        "skipped_duplicates": 0,
        "skipped_excluded": 0,
    }

    seen_hashes = set()
    excluded_paths = set()

    if exclude_paths_file is not None and os.path.exists(exclude_paths_file):
        with open(exclude_paths_file, "r", encoding="utf-8") as f:
            excluded_paths = set(line.strip() for line in f if line.strip())

        print(f"Excloent {len(excluded_paths)} imatges segons: {exclude_paths_file}")
    
    class_names = sorted([
        class_name
        for class_name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, class_name))
    ])

    for class_idx, class_name in enumerate(class_names):
        class_to_idx[class_name] = class_idx
        class_dir = os.path.join(root_dir, class_name)

        for filename in sorted(os.listdir(class_dir)):
            stats["total_files_seen"] += 1

            if not is_image_file(filename):
                stats["skipped_non_images"] += 1
                continue

            image_path = os.path.join(class_dir, filename)

            if image_path in excluded_paths:
                stats["skipped_excluded"] += 1
                continue

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
    """
    Retorna la distribució de classes ordenada de més gran a més petita.
    """
    counts = Counter(labels)

    distribution = []
    for class_idx, count in counts.items():
        class_name = idx_to_class[class_idx]
        distribution.append((class_name, count))

    distribution = sorted(distribution, key=lambda x: x[1], reverse=True)

    return distribution


def print_dataset_summary(image_paths, labels, class_to_idx, idx_to_class, stats):
    """
    Mostra un resum del dataset carregat.
    """
    distribution = get_class_distribution(labels, idx_to_class)

    print("\n========== DATASET SUMMARY ==========")
    print(f"Total files seen:        {stats['total_files_seen']}")
    print(f"Valid images:            {stats['valid_images']}")
    print(f"Skipped non-images:      {stats['skipped_non_images']}")
    print(f"Skipped corrupted:       {stats['skipped_corrupted']}")
    print(f"Skipped duplicates:      {stats['skipped_duplicates']}")
    print(f"Number of classes:       {len(class_to_idx)}")
    print(f"Number of image paths:   {len(image_paths)}")
    print(f"Number of labels:        {len(labels)}")
    print(f"Skipped excluded:       {stats.get('skipped_excluded', 0)}")

    print("\nTop 10 largest classes:")
    for class_name, count in distribution[:10]:
        print(f"  {class_name}: {count}")

    print("\nTop 10 smallest classes:")
    for class_name, count in distribution[-10:]:
        print(f"  {class_name}: {count}")

    print("=====================================\n")


def split_dataset(image_paths, labels, val_size=0.15, test_size=0.15, random_state=42):
    """
    Split 70/15/15 estratificat.

    Estratificat vol dir que intenta mantenir la proporció de classes
    a train, validation i test.
    """

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
    """
    Mostra mides finals dels conjunts.
    """
    print("\n========== SPLIT SUMMARY ==========")
    print(f"Train images: {len(train_labels)}")
    print(f"Val images:   {len(val_labels)}")
    print(f"Test images:  {len(test_labels)}")
    print("===================================\n")


def get_transforms(
    image_size=336,
    resize_images=True,
    use_augmentation=False,
    use_aspect_crop_cache=False,
):
    train_transforms = []
    val_test_transforms = []

    if use_aspect_crop_cache:
        # El cache ja té costat curt = image_size.
        # Ara només fem crop quadrat.
        train_transforms.append(transforms.RandomCrop(image_size))
        val_test_transforms.append(transforms.CenterCrop(image_size))

    else:
        if resize_images:
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

    train_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_test_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return transforms.Compose(train_transforms), transforms.Compose(val_test_transforms)


def create_dataloaders(
    train_paths,
    val_paths,
    test_paths,
    train_labels,
    val_labels,
    test_labels,
    batch_size=32,
    image_size=224,
    num_workers=2,
    resize_images=True,
    use_augmentation=False,
    use_weighted_sampler=False,
    use_aspect_crop_cache=False,
):
    """
    Crea els Dataset i DataLoader de train, validation i test.
    """

    train_transform, val_test_transform = get_transforms(
        image_size=image_size,
        resize_images=resize_images,
        use_augmentation=use_augmentation,
        use_aspect_crop_cache=use_aspect_crop_cache,
    )

    train_dataset = ImageDataset(
        image_paths=train_paths,
        labels=train_labels,
        transform=train_transform,
    )

    val_dataset = ImageDataset(
        image_paths=val_paths,
        labels=val_labels,
        transform=val_test_transform,
    )

    test_dataset = ImageDataset(
        image_paths=test_paths,
        labels=test_labels,
        transform=val_test_transform,
    )

    train_sampler = None
    train_shuffle = True

    if use_weighted_sampler:
        train_sampler = create_weighted_sampler(train_labels)
        train_shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    return train_loader, val_loader, test_loader


#Experiment2

def filter_top_k_classes(image_paths, labels, idx_to_class, top_k=14):
    """
    Conserva només les top_k classes amb més imatges.

    Important:
    Després de filtrar, reindexem les labels perquè siguin:
        0, 1, 2, ..., num_classes-1

    Això és necessari perquè CrossEntropyLoss espera labels consecutives.
    """

    counts = Counter(labels)

    # Agafem les classes més grans segons nombre d'imatges
    top_labels = [
        label
        for label, _ in counts.most_common(top_k)
    ]

    top_labels_set = set(top_labels)

    removed_classes = [
        idx_to_class[label]
        for label in counts.keys()
        if label not in top_labels_set
    ]

    print(f"\nConservant les {top_k} classes amb més imatges.")
    print(f"Eliminant {len(removed_classes)} classes:")
    for class_name in removed_classes:
        print(f"  - {class_name}")

    # Nou mapping classe -> índex
    # Mantinc l'ordre de major a menor nombre d'imatges perquè sigui clar.
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
            old_class_name = idx_to_class[old_label]
            new_label = new_class_to_idx[old_class_name]

            filtered_paths.append(image_path)
            filtered_labels.append(new_label)

    print(
        f"Després del filtre top-{top_k}: "
        f"{len(filtered_paths)} imatges, {len(new_class_to_idx)} classes.\n"
    )

    return filtered_paths, filtered_labels, new_class_to_idx, new_idx_to_class


#Experiment11

def cap_images_per_class(image_paths, labels, idx_to_class, max_images_per_class=6000, random_seed=42):
    """
    Limita el nombre màxim d'imatges per classe.

    Exemple:
        si Impressionism té 12000 imatges i max_images_per_class=6000,
        en conservem només 6000.

    Les classes amb menys de max_images_per_class imatges es mantenen igual.

    Important:
        això redueix el pes de les classes majoritàries
        sense repetir imatges minoritàries ni tocar la loss.
    """

    rng = random.Random(random_seed)

    # Agrupem els índexs de les imatges per classe.
    indices_by_class = {}

    for idx, label in enumerate(labels):
        if label not in indices_by_class:
            indices_by_class[label] = []

        indices_by_class[label].append(idx)

    selected_indices = []

    print(f"\nAplicarem cap màxim de {max_images_per_class} imatges per classe:")

    for label, indices in indices_by_class.items():
        class_name = idx_to_class[label]
        original_count = len(indices)

        if original_count > max_images_per_class:
            selected = rng.sample(indices, max_images_per_class)
            print(f"  {class_name}: {original_count} -> {max_images_per_class}")
        else:
            selected = indices
            print(f"  {class_name}: {original_count} -> {original_count}")

        selected_indices.extend(selected)

    # Ordenem per tenir una sortida estable.
    selected_indices = sorted(selected_indices)

    capped_paths = [image_paths[i] for i in selected_indices]
    capped_labels = [labels[i] for i in selected_indices]

    print(
        f"Després del cap: {len(capped_paths)} imatges "
        f"i {len(set(capped_labels))} classes.\n"
    )

    return capped_paths, capped_labels


#Experiment3:

def compute_class_weights(labels, num_classes, idx_to_class=None):
    """
    Calcula pesos de classe a partir de les labels de train.

    Les classes amb menys imatges reben més pes.
    Les classes amb més imatges reben menys pes.

    Aquests pesos s'utilitzaran a CrossEntropyLoss(weight=...).
    """

    counts = Counter(labels)

    weights = torch.zeros(num_classes, dtype=torch.float32)

    for class_idx in range(num_classes):
        weights[class_idx] = 1.0 / (counts[class_idx] ** 0.5)
    
    # Normalitzem perquè la mitjana dels pesos sigui aproximadament 1.
    # Això ajuda a mantenir l'escala de la loss més estable.
    weights = weights / weights.sum() * num_classes

    print("\n========== CLASS WEIGHTS ==========")
    for class_idx in range(num_classes):
        class_name = idx_to_class[class_idx] if idx_to_class is not None else str(class_idx)
        print(
            f"{class_name}: "
            f"count={counts[class_idx]}, "
            f"weight={weights[class_idx]:.4f}"
        )
    print("===================================\n")

    return weights

def create_weighted_sampler(labels):
    """
    Crea un WeightedRandomSampler per compensar el desbalanceig de classes.

    Cada imatge rep un pes inversament proporcional al nombre d'imatges
    de la seva classe. Això fa que les classes minoritàries apareguin
    més sovint durant l'entrenament.
    """

    counts = Counter(labels)

    sample_weights = []
    for label in labels:
        class_count = counts[label]
        sample_weight = 1.0 / class_count
        sample_weights.append(sample_weight)

    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    return sampler

