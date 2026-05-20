import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils.data_utils import (
    prepare_resized_cache_dataset,
    load_wikiart_dataset,
    filter_top_k_classes,
    split_dataset,
    create_dataloaders,
    group_similar_classes,
)
from models.transfer_model import create_resnet_model


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model


@torch.inference_mode()
def evaluate_tta(model, test_loader, device, criterion, use_flip=True):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_preds = []
    all_labels = []

    for images, labels in tqdm(test_loader, desc="Testing with TTA", mininterval=1.0):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits_original = model(images)
        probs = F.softmax(logits_original, dim=1)

        if use_flip:
            flipped_images = torch.flip(images, dims=[3])
            logits_flip = model(flipped_images)
            probs_flip = F.softmax(logits_flip, dim=1)

            probs = (probs + probs_flip) / 2.0

        # Per calcular una loss aproximada amb les probabilitats finals.
        log_probs = torch.log(torch.clamp(probs, min=1e-8))
        loss = F.nll_loss(log_probs, labels)

        preds = torch.argmax(probs, dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, accuracy, macro_f1, weighted_f1, cm, all_labels, all_preds


def save_results(output_dir, idx_to_class, test_loss, test_acc, macro_f1, weighted_f1, cm):
    os.makedirs(output_dir, exist_ok=True)

    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    summary = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

    summary_path = os.path.join(output_dir, "tta_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    cm_path = os.path.join(output_dir, "tta_confusion_matrix.csv")
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(cm_path)

    print("\n========== EXP24 TTA RESULTS ==========")
    print(f"Test Loss:    {test_loss:.4f}")
    print(f"Test Acc:     {test_acc:.4f}")
    print(f"Macro F1:     {macro_f1:.4f}")
    print(f"Weighted F1:  {weighted_f1:.4f}")
    print("=======================================\n")

    print(f"Summary guardat a: {summary_path}")
    print(f"Confusion matrix guardada a: {cm_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_root", type=str, default="/home/datasets/wikiart/")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="resnet50")

    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--use_resized_cache", action="store_true")
    parser.add_argument("--resized_cache_root", type=str, default="/tmp/wikiart_384")
    parser.add_argument("--cache_num_workers", type=int, default=12)

    parser.add_argument("--top_k", type=int, default=14)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)

    parser.add_argument("--no_flip", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/exp24_tta")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_root = args.dataset_root

    if args.use_resized_cache:
        dataset_root = prepare_resized_cache_dataset(
            source_root=args.dataset_root,
            cache_root=args.resized_cache_root,
            image_size=args.image_size,
            force_rebuild=False,
            num_workers=args.cache_num_workers,
            cache_resize_mode="square",
        )

    image_paths, labels, class_to_idx, idx_to_class, stats = load_wikiart_dataset(
        root_dir=dataset_root,
        remove_duplicates=False,
        check_corrupted=False,
    )

    image_paths, labels, class_to_idx, idx_to_class = filter_top_k_classes(
        image_paths=image_paths,
        labels=labels,
        idx_to_class=idx_to_class,
        top_k=args.top_k,
    )

    class_groups = {
        "Impressionism_Post_Impressionism": [
            "Impressionism",
            "Post_Impressionism",
        ],
        "Baroque_Rococo": [
            "Baroque",
            "Rococo",
        ],
        "AbstractExpressionism_ColorField": [
            "Abstract_Expressionism",
            "Color_Field_Painting",
        ],
    }

    image_paths, labels, class_to_idx, idx_to_class = group_similar_classes(
        image_paths=image_paths,
        labels=labels,
        idx_to_class=idx_to_class,
        class_groups=class_groups,
    )   

    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = split_dataset(
        image_paths=image_paths,
        labels=labels,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_seed,
    )

    _, _, test_loader = create_dataloaders(
        train_paths=train_paths,
        val_paths=val_paths,
        test_paths=test_paths,
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        resize_images=not args.use_resized_cache,
        use_augmentation=False,
        use_weighted_sampler=False,
        use_aspect_crop_cache=False,
    )

    num_classes = len(class_to_idx)

    model = create_resnet_model(
        num_classes=num_classes,
        model_name=args.model_name,
        feature_extraction=False,
        partial_finetuning=True,
        unfreeze_layer3=False,
    )

    model = load_checkpoint(model, args.checkpoint_path, device)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    use_flip = not args.no_flip

    print("\n========== TTA CONFIG ==========")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Image size: {args.image_size}")
    print(f"Use flip:   {use_flip}")
    print(f"Top-k:      {args.top_k}")
    print("================================\n")

    test_loss, test_acc, macro_f1, weighted_f1, cm, all_labels, all_preds = evaluate_tta(
        model=model,
        test_loader=test_loader,
        device=device,
        criterion=criterion,
        use_flip=use_flip,
    )

    save_results(
        output_dir=args.output_dir,
        idx_to_class=idx_to_class,
        test_loss=test_loss,
        test_acc=test_acc,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        cm=cm,
    )


if __name__ == "__main__":
    main()
