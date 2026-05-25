import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.transfer_model import create_resnet_model


EXP27_CLASS_NAMES = [
    "Abstract_Expressionism",
    "Action_painting",
    "Analytical_Cubism",
    "Art_Nouveau_Modern",
    "Baroque",
    "Color_Field_Painting",
    "Contemporary_Realism",
    "Cubism",
    "Early_Renaissance",
    "Expressionism",
    "Fauvism",
    "High_Renaissance",
    "Impressionism",
    "Mannerism_Late_Renaissance",
    "Minimalism",
    "Naive_Art_Primitivism",
    "New_Realism",
    "Northern_Renaissance",
    "Pointillism",
    "Pop_Art",
    "Post_Impressionism",
    "Realism",
    "Rococo",
    "Romanticism",
    "Symbolism",
    "Synthetic_Cubism",
    "Ukiyo_e",
]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif"}


class OODFolderDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return image, label, str(image_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an EXP27 27-class checkpoint on an external OOD folder "
            "dataset."
        )
    )
    parser.add_argument("--ood_root", type=str, default="data/ood_art_external")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="results/ood_exp27")
    parser.add_argument("--wandb_project", type=str, default="wikiart-classification")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--disable_wandb", action="store_true")
    return parser.parse_args()


def build_inference_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def normalize_idx_to_class(idx_to_class):
    if isinstance(idx_to_class, (list, tuple)):
        return list(idx_to_class)

    if isinstance(idx_to_class, dict):
        try:
            return [idx_to_class[i] for i in range(len(idx_to_class))]
        except KeyError:
            return [idx_to_class[str(i)] for i in range(len(idx_to_class))]

    raise ValueError(
        "checkpoint idx_to_class must be a list, tuple, or dict indexed from 0."
    )


def get_checkpoint_classes(checkpoint):
    if isinstance(checkpoint, dict) and "idx_to_class" in checkpoint:
        class_names = normalize_idx_to_class(checkpoint["idx_to_class"])
        print("Using class order from checkpoint idx_to_class.")
    else:
        class_names = EXP27_CLASS_NAMES
        print("Checkpoint has no idx_to_class. Using default EXP27 class order.")

    if len(class_names) != 27:
        raise ValueError(
            f"Expected 27 classes for EXP27, but got {len(class_names)} classes."
        )

    return class_names


def get_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]

    return checkpoint


def strip_module_prefix_if_present(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict

    if not all(isinstance(key, str) for key in state_dict.keys()):
        return state_dict

    if any(key.startswith("module.") for key in state_dict.keys()):
        return {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    return state_dict


def load_exp27_checkpoint(model, checkpoint, checkpoint_path):
    state_dict = strip_module_prefix_if_present(get_state_dict(checkpoint))

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        message = str(exc)
        if "size mismatch" in message and ("fc." in message or "classifier" in message):
            raise RuntimeError(
                "Could not load checkpoint into a 27-class ResNet head. "
                "This probably means the checkpoint is not from the 27-class EXP27 run."
            ) from exc

        raise RuntimeError(
            f"Could not load checkpoint '{checkpoint_path}' into the EXP27 model."
        ) from exc

    return model


def collect_ood_samples(ood_root, class_to_idx):
    ood_root = Path(ood_root)

    if not ood_root.exists():
        raise FileNotFoundError(f"OOD root does not exist: {ood_root}")

    if not ood_root.is_dir():
        raise NotADirectoryError(f"OOD root is not a directory: {ood_root}")

    samples = []
    ignored_folders = []
    detected_classes = []

    for class_dir in sorted(path for path in ood_root.iterdir() if path.is_dir()):
        class_name = class_dir.name

        if class_name not in class_to_idx:
            ignored_folders.append(class_name)
            continue

        detected_classes.append(class_name)
        label = class_to_idx[class_name]

        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in VALID_EXTENSIONS:
                samples.append((image_path, label))

    for folder_name in ignored_folders:
        print(
            "WARNING: Ignoring OOD folder not present in checkpoint classes: "
            f"{folder_name}"
        )

    if not samples:
        raise ValueError(
            f"No OOD images found in {ood_root} with extensions: "
            f"{', '.join(sorted(VALID_EXTENSIONS))}"
        )

    return samples, detected_classes


def print_dataset_info(samples, detected_classes, class_names):
    class_counts = Counter(label for _, label in samples)

    print("\n========== OOD DATASET ==========")
    print(f"Total OOD images: {len(samples)}")

    print("\nImages per class:")
    for class_idx, class_name in enumerate(class_names):
        print(f"  {class_name}: {class_counts.get(class_idx, 0)}")

    print("\nClasses detected in OOD dataset:")
    for class_name in detected_classes:
        print(f"  {class_name}")

    print("\nClasses expected by checkpoint:")
    for class_name in class_names:
        print(f"  {class_name}")

    print("=================================\n")


def build_class_distribution(samples, class_names):
    class_counts = Counter(label for _, label in samples)

    return pd.DataFrame(
        [
            {
                "class_name": class_name,
                "num_images": class_counts.get(class_idx, 0),
            }
            for class_idx, class_name in enumerate(class_names)
        ]
    )


@torch.inference_mode()
def run_inference(model, data_loader, device, class_names):
    model.eval()

    image_paths = []
    all_labels = []
    all_preds = []
    all_confidences = []

    for images, labels, paths in tqdm(
        data_loader,
        desc="OOD inference",
        mininterval=1.0,
    ):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        probs = F.softmax(logits, dim=1)

        confidences, preds = torch.max(probs, dim=1)

        image_paths.extend(paths)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_confidences.extend(confidences.cpu().tolist())

    predictions = pd.DataFrame(
        {
            "image_path": image_paths,
            "true_label": [class_names[label] for label in all_labels],
            "pred_label": [class_names[pred] for pred in all_preds],
            "confidence": all_confidences,
            "correct": [
                label == pred for label, pred in zip(all_labels, all_preds)
            ],
        }
    )

    return predictions, all_labels, all_preds


def save_confusion_matrix_image(cm, class_names, output_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 14))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )
    disp.plot(ax=ax, xticks_rotation=90, colorbar=True)
    plt.title("Confusion Matrix - OOD Set")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def build_classification_report_table(labels, preds, class_names):
    label_ids = list(range(len(class_names)))
    report = classification_report(
        labels,
        preds,
        labels=label_ids,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            rows.append(
                {
                    "label": label,
                    "precision": values.get("precision"),
                    "recall": values.get("recall"),
                    "f1_score": values.get("f1-score"),
                    "support": values.get("support"),
                }
            )
        else:
            rows.append(
                {
                    "label": label,
                    "precision": None,
                    "recall": None,
                    "f1_score": values,
                    "support": len(labels),
                }
            )

    return pd.DataFrame(rows)


def save_results(
    output_dir,
    checkpoint_path,
    class_names,
    predictions,
    labels,
    preds,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_ids = list(range(len(class_names)))

    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(
        labels,
        preds,
        labels=label_ids,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        labels,
        preds,
        labels=label_ids,
        average="weighted",
        zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=label_ids)
    report = classification_report(
        labels,
        preds,
        labels=label_ids,
        target_names=class_names,
        zero_division=0,
    )

    predictions_path = output_dir / "ood_predictions.csv"
    summary_path = output_dir / "ood_summary.csv"
    cm_path = output_dir / "ood_confusion_matrix.csv"
    cm_image_path = output_dir / "ood_confusion_matrix.png"
    report_path = output_dir / "ood_classification_report.txt"

    predictions.to_csv(predictions_path, index=False)

    summary = pd.DataFrame(
        [
            {
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "num_images": len(labels),
                "num_classes": len(class_names),
                "checkpoint_path": checkpoint_path,
            }
        ]
    )
    summary.to_csv(summary_path, index=False)

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(cm_path)
    save_confusion_matrix_image(cm, class_names, cm_image_path)

    report_path.write_text(report, encoding="utf-8")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "num_images": len(labels),
        "num_classes": len(class_names),
        "predictions_path": predictions_path,
        "summary_path": summary_path,
        "cm_path": cm_path,
        "cm_image_path": cm_image_path,
        "report_path": report_path,
    }


def build_wandb_config(args, device, class_names, detected_classes, samples):
    class_distribution = build_class_distribution(samples, class_names)

    return {
        "ood_root": args.ood_root,
        "checkpoint_path": args.checkpoint_path,
        "model_name": args.model_name,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "output_dir": args.output_dir,
        "device": str(device),
        "num_images": len(samples),
        "num_classes": len(class_names),
        "class_names": class_names,
        "detected_classes": detected_classes,
        "class_distribution": class_distribution.to_dict(orient="records"),
        "use_top_k_classes": False,
        "use_class_grouping": False,
    }


def init_wandb(args, device, class_names, detected_classes, samples):
    if args.disable_wandb:
        return None

    checkpoint_name = Path(args.checkpoint_path).stem
    run_name = args.wandb_run_name or f"ood_exp27_{checkpoint_name}"

    return wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=build_wandb_config(
            args=args,
            device=device,
            class_names=class_names,
            detected_classes=detected_classes,
            samples=samples,
        ),
    )


def log_ood_to_wandb(metrics, predictions, labels, preds, class_names, samples):
    if wandb.run is None:
        return

    class_distribution = build_class_distribution(samples, class_names)
    classification_report_table = build_classification_report_table(
        labels=labels,
        preds=preds,
        class_names=class_names,
    )

    wandb.log(
        {
            "ood/accuracy": metrics["accuracy"],
            "ood/macro_f1": metrics["macro_f1"],
            "ood/weighted_f1": metrics["weighted_f1"],
            "ood/num_images": metrics["num_images"],
            "ood/num_classes": metrics["num_classes"],
            "ood/confusion_matrix": wandb.Image(str(metrics["cm_image_path"])),
            "ood/predictions": wandb.Table(dataframe=predictions),
            "ood/class_distribution": wandb.Table(dataframe=class_distribution),
            "ood/classification_report": wandb.Table(
                dataframe=classification_report_table
            ),
        }
    )

    wandb.summary.update(
        {
            "ood_accuracy": metrics["accuracy"],
            "ood_macro_f1": metrics["macro_f1"],
            "ood_weighted_f1": metrics["weighted_f1"],
        }
    )

    artifact = wandb.Artifact(
        name=f"ood-results-{wandb.run.id}",
        type="ood-evaluation",
    )
    for path_key in ("predictions_path", "summary_path", "cm_path", "report_path"):
        artifact.add_file(str(metrics[path_key]))
    artifact.add_file(str(metrics["cm_image_path"]))
    wandb.log_artifact(artifact)


def print_final_summary(metrics):
    print("\n========== OOD EXP27 RESULTS ==========")
    print(f"Accuracy:     {metrics['accuracy']:.4f}")
    print(f"Macro F1:     {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:  {metrics['weighted_f1']:.4f}")
    print(f"Num images:   {metrics['num_images']}")
    print(f"Num classes:  {metrics['num_classes']}")
    print("=======================================\n")

    print(f"Predictions saved to: {metrics['predictions_path']}")
    print(f"Summary saved to: {metrics['summary_path']}")
    print(f"Confusion matrix saved to: {metrics['cm_path']}")
    print(f"Confusion matrix image saved to: {metrics['cm_image_path']}")
    print(f"Classification report saved to: {metrics['report_path']}")


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"OOD root: {args.ood_root}")
    print("use_top_k_classes: False")
    print("use_class_grouping: False")

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    class_names = get_checkpoint_classes(checkpoint)
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    samples, detected_classes = collect_ood_samples(args.ood_root, class_to_idx)
    print_dataset_info(samples, detected_classes, class_names)
    wandb_run = init_wandb(
        args=args,
        device=device,
        class_names=class_names,
        detected_classes=detected_classes,
        samples=samples,
    )

    try:
        transform = build_inference_transform(args.image_size)
        dataset = OODFolderDataset(samples=samples, transform=transform)
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        model = create_resnet_model(
            num_classes=27,
            model_name=args.model_name,
            feature_extraction=False,
            partial_finetuning=True,
            unfreeze_layer3=False,
        )
        model = load_exp27_checkpoint(model, checkpoint, args.checkpoint_path)
        model = model.to(device)

        predictions, labels, preds = run_inference(
            model=model,
            data_loader=data_loader,
            device=device,
            class_names=class_names,
        )

        metrics = save_results(
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint_path,
            class_names=class_names,
            predictions=predictions,
            labels=labels,
            preds=preds,
        )
        log_ood_to_wandb(
            metrics=metrics,
            predictions=predictions,
            labels=labels,
            preds=preds,
            class_names=class_names,
            samples=samples,
        )
        print_final_summary(metrics)
    finally:
        if wandb_run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
