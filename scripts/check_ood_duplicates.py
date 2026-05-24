import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


ORIGINAL_DATASET_ROOT = "/home/datasets/wikiart"
OOD_DATASET_ROOT = "data/ood_art_external"
OUTPUT_DIR = "results/ood_duplicate_check"
DEFAULT_VISUAL_THRESHOLD = 5

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif")
HASH_BLOCK_SIZE = 1024 * 1024


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Detecta duplicats exactes i possibles duplicats visuals entre "
            "WikiArt i un mini dataset OOD extern."
        )
    )
    parser.add_argument("--original_root", type=str, default=ORIGINAL_DATASET_ROOT)
    parser.add_argument("--ood_root", type=str, default=OOD_DATASET_ROOT)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--visual_threshold", type=int, default=DEFAULT_VISUAL_THRESHOLD)
    return parser.parse_args()


def validate_root(root_path, name):
    root = Path(root_path)
    if not root.exists():
        raise SystemExit(f"ERROR: La ruta {name} no existeix: {root}")
    if not root.is_dir():
        raise SystemExit(f"ERROR: La ruta {name} no es un directori: {root}")
    return root


def find_image_paths(root):
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def calculate_file_hash(image_path):
    sha256 = hashlib.sha256()

    with image_path.open("rb") as image_file:
        for block in iter(lambda: image_file.read(HASH_BLOCK_SIZE), b""):
            sha256.update(block)

    return sha256.hexdigest()


def calculate_exact_hashes(image_paths, label):
    hashes_by_path = {}
    paths_by_hash = {}

    for image_path in tqdm(image_paths, desc=f"Hashes exactes {label}", unit="img"):
        try:
            exact_hash = calculate_file_hash(image_path)
        except OSError as error:
            print(
                f"WARNING: No s'ha pogut llegir el fitxer {image_path}: {error}",
                file=sys.stderr,
            )
            continue

        path_str = str(image_path)
        hashes_by_path[path_str] = exact_hash
        paths_by_hash.setdefault(exact_hash, []).append(path_str)

    return hashes_by_path, paths_by_hash


def find_exact_duplicates(ood_hashes_by_path, original_paths_by_hash):
    rows = []

    for ood_path, exact_hash in ood_hashes_by_path.items():
        for original_path in original_paths_by_hash.get(exact_hash, []):
            rows.append(
                {
                    "ood_path": ood_path,
                    "original_path": original_path,
                    "exact_hash": exact_hash,
                }
            )

    return rows


def load_imagehash():
    try:
        import imagehash
    except ImportError as error:
        raise SystemExit(
            "ERROR: Falta la dependencia 'imagehash'. Instal-la amb:\n"
            "  pip install imagehash"
        ) from error

    return imagehash


def calculate_phashes(image_paths, label, imagehash_module):
    phashes = {}

    for image_path in tqdm(image_paths, desc=f"pHash {label}", unit="img"):
        try:
            with Image.open(image_path) as image:
                phash = imagehash_module.phash(image.convert("RGB"))
        except Exception as error:
            print(
                f"WARNING: No s'ha pogut obrir/processar la imatge {image_path}: {error}",
                file=sys.stderr,
            )
            continue

        phashes[str(image_path)] = phash

    return phashes


def find_visual_duplicates(ood_phashes, original_phashes, threshold):
    rows = []
    original_items = list(original_phashes.items())

    for ood_path, ood_phash in tqdm(
        ood_phashes.items(), desc="Comparant pHash OOD vs original", unit="img"
    ):
        for original_path, original_phash in original_items:
            distance = ood_phash - original_phash
            if distance <= threshold:
                rows.append(
                    {
                        "ood_path": ood_path,
                        "original_path": original_path,
                        "phash_distance": distance,
                        "ood_phash": str(ood_phash),
                        "original_phash": str(original_phash),
                    }
                )

    return rows


def get_class_name(image_path, dataset_root):
    image_path = Path(image_path)

    try:
        relative_path = image_path.relative_to(dataset_root)
    except ValueError:
        relative_path = image_path

    if len(relative_path.parts) <= 1:
        return "unknown"

    return relative_path.parts[0]


def build_ood_class_distribution(ood_paths, ood_root, visual_rows):
    duplicate_ood_paths = {row["ood_path"] for row in visual_rows}
    before_counts = Counter(get_class_name(path, ood_root) for path in ood_paths)
    duplicate_counts = Counter(
        get_class_name(path, ood_root)
        for path in duplicate_ood_paths
    )

    rows = []
    for class_name in sorted(before_counts):
        before_count = before_counts[class_name]
        duplicated_count = duplicate_counts.get(class_name, 0)
        rows.append(
            {
                "class_name": class_name,
                "before_removing_visual_duplicates": before_count,
                "visual_duplicates_to_remove_for_info_only": duplicated_count,
                "after_removing_visual_duplicates_for_info_only": (
                    before_count - duplicated_count
                ),
            }
        )

    return rows


def format_class_distribution(distribution_rows):
    if not distribution_rows:
        return ["No hi ha classes OOD per mostrar."]

    header = (
        f"{'classe':<35} "
        f"{'abans':>8} "
        f"{'duplicades':>11} "
        f"{'despres_info':>13}"
    )
    separator = "-" * len(header)
    lines = [header, separator]

    for row in distribution_rows:
        lines.append(
            f"{row['class_name']:<35} "
            f"{row['before_removing_visual_duplicates']:>8} "
            f"{row['visual_duplicates_to_remove_for_info_only']:>11} "
            f"{row['after_removing_visual_duplicates_for_info_only']:>13}"
        )

    return lines


def save_results(output_dir, exact_rows, visual_rows, class_distribution_rows, summary_lines):
    output_dir.mkdir(parents=True, exist_ok=True)

    exact_path = output_dir / "exact_duplicates.csv"
    visual_path = output_dir / "visual_duplicates.csv"
    class_distribution_path = output_dir / "ood_class_distribution_without_visual_duplicates.csv"
    summary_path = output_dir / "summary.txt"

    exact_columns = ["ood_path", "original_path", "exact_hash"]
    visual_columns = [
        "ood_path",
        "original_path",
        "phash_distance",
        "ood_phash",
        "original_phash",
    ]

    pd.DataFrame(exact_rows, columns=exact_columns).to_csv(exact_path, index=False)
    pd.DataFrame(visual_rows, columns=visual_columns).to_csv(visual_path, index=False)
    pd.DataFrame(class_distribution_rows).to_csv(class_distribution_path, index=False)
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return exact_path, visual_path, class_distribution_path, summary_path


def main():
    args = parse_args()

    original_root = validate_root(args.original_root, "--original_root")
    ood_root = validate_root(args.ood_root, "--ood_root")
    output_dir = Path(args.output_dir)

    imagehash_module = load_imagehash()

    print(f"Dataset original: {original_root}")
    print(f"Dataset OOD:      {ood_root}")
    print(f"Output:           {output_dir}")
    print(f"Threshold visual: {args.visual_threshold}")

    print("\nBuscant imatges...")
    original_paths = find_image_paths(original_root)
    ood_paths = find_image_paths(ood_root)

    print(f"Imatges originals trobades: {len(original_paths)}")
    print(f"Imatges OOD trobades:       {len(ood_paths)}")

    _, original_paths_by_hash = calculate_exact_hashes(
        original_paths, "originals"
    )
    ood_hashes_by_path, _ = calculate_exact_hashes(ood_paths, "OOD")

    exact_rows = find_exact_duplicates(ood_hashes_by_path, original_paths_by_hash)

    original_phashes = calculate_phashes(original_paths, "originals", imagehash_module)
    ood_phashes = calculate_phashes(ood_paths, "OOD", imagehash_module)

    visual_rows = find_visual_duplicates(
        ood_phashes=ood_phashes,
        original_phashes=original_phashes,
        threshold=args.visual_threshold,
    )
    class_distribution_rows = build_ood_class_distribution(
        ood_paths=ood_paths,
        ood_root=ood_root,
        visual_rows=visual_rows,
    )

    duplicated_ood_paths = {row["ood_path"] for row in visual_rows}
    ood_after_visual_filter = len(ood_paths) - len(duplicated_ood_paths)
    class_distribution_lines = format_class_distribution(class_distribution_rows)

    summary_lines = [
        f"nombre d'imatges originals escanejades: {len(original_paths)}",
        f"nombre d'imatges OOD escanejades: {len(ood_paths)}",
        f"nombre de duplicats exactes: {len(exact_rows)}",
        f"nombre de possibles duplicats visuals: {len(visual_rows)}",
        (
            "nombre d'imatges OOD uniques duplicades visuals "
            f"(nomes informatiu): {len(duplicated_ood_paths)}"
        ),
        (
            "nombre d'imatges OOD si s'exclouen duplicats visuals "
            f"(nomes informatiu): {ood_after_visual_filter}"
        ),
        f"threshold utilitzat: {args.visual_threshold}",
        "",
        "distribucio de classes OOD abans/despres d'excloure duplicats visuals "
        "(nomes informatiu; no s'esborra cap fitxer):",
        *class_distribution_lines,
    ]

    exact_path, visual_path, class_distribution_path, summary_path = save_results(
        output_dir=output_dir,
        exact_rows=exact_rows,
        visual_rows=visual_rows,
        class_distribution_rows=class_distribution_rows,
        summary_lines=summary_lines,
    )

    print("\n========== RESUM DUPLICATS OOD ==========")
    for line in summary_lines:
        print(line)
    print("=========================================")
    print(f"CSV duplicats exactes:   {exact_path}")
    print(f"CSV duplicats visuals:   {visual_path}")
    print(f"CSV distribucio classes: {class_distribution_path}")
    print(f"Resum guardat a:         {summary_path}")


if __name__ == "__main__":
    main()
