import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from models.transfer_model import create_resnet_model
from test import test_model
from train import train_model
from utils.data_utils import (
    compute_class_weights,
    create_dataloaders,
    filter_top_k_classes,
    group_similar_classes,
    load_wikiart_dataset,
    prepare_resized_cache_dataset,
    print_dataset_summary,
    print_split_summary,
    split_dataset,
)


def set_seed(seed):
    """Fixem llavors per fer l'experiment més reproduïble."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint(model, checkpoint_path, device):
    """Carrega el millor checkpoint guardat durant l'entrenament."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Checkpoint carregat: {checkpoint_path}")
    print(f"Epoch checkpoint: {checkpoint['epoch'] + 1}")
    print(f"Validation macro F1 checkpoint: {checkpoint['val_macro_f1']:.4f}")

    return model


def build_config():
    return {
        "experiment_name": "exp27_resnet50_384_all_classes",
        "dataset_root": "/home/datasets/wikiart/",
        "model_name": "resnet50",
        "image_size": 384,
        "batch_size": 32,
        "epochs": 20,
        "learning_rate": 1e-5,
        "weight_decay": 5e-4,
        "early_stopping_patience": 3,
        "seed": 42,
        "val_size": 0.15,
        "test_size": 0.15,
        "num_workers": 8,
        "feature_extraction": False,
        "partial_finetuning": True,
        "unfreeze_layer3": False,
        "use_top_k_classes": False,
        "top_k_classes": 14,
        "use_class_grouping": False,
        "class_groups": {},
        "use_class_weights": True,
        "use_weighted_sampler": False,
        "use_label_smoothing": True,
        "label_smoothing": 0.05,
        "use_scheduler": True,
        "scheduler_factor": 0.5,
        "scheduler_patience": 1,
        "scheduler_min_lr": 1e-6,
        "use_augmentation": True,
        "use_resized_cache": True,
        "resized_cache_root": "/tmp/wikiart_384",
        "force_rebuild_cache": False,
        # Per reproduir experiments d'aspect ratio: "square", "short_side" o "padding".
        "cache_resize_mode": "square",
        "use_aspect_crop_cache": False,
    }


def build_loss(config, train_labels, num_classes, idx_to_class, device):
    label_smoothing = (
        config["label_smoothing"] if config["use_label_smoothing"] else 0.0
    )

    if config["use_class_weights"]:
        class_weights = compute_class_weights(
            labels=train_labels,
            num_classes=num_classes,
            idx_to_class=idx_to_class,
        ).to(device)

        criterion_train = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
        wandb.config.update({"class_weights": class_weights.cpu().tolist()})
    else:
        criterion_train = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    criterion_eval = nn.CrossEntropyLoss()
    return criterion_train, criterion_eval


def main():
    config = build_config()
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilitzat: {device}")

    checkpoint_path = os.path.join(
        "results",
        "checkpoints",
        f"{config['experiment_name']}_best.pth",
    )

    wandb.init(
        project="wikiart-classification",
        name=config["experiment_name"],
        config=config,
    )

    dataset_root = config["dataset_root"]
    resize_images_in_dataloader = True

    if config["use_resized_cache"]:
        dataset_root = prepare_resized_cache_dataset(
            source_root=config["dataset_root"],
            cache_root=config["resized_cache_root"],
            image_size=config["image_size"],
            force_rebuild=config["force_rebuild_cache"],
            num_workers=config["num_workers"],
            cache_resize_mode=config["cache_resize_mode"],
        )
        resize_images_in_dataloader = False

    image_paths, labels, class_to_idx, idx_to_class, stats = load_wikiart_dataset(
        root_dir=dataset_root,
        remove_duplicates=False,
        check_corrupted=False,
    )

    print_dataset_summary(
        image_paths=image_paths,
        labels=labels,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        stats=stats,
    )

    if config["use_top_k_classes"]:
        image_paths, labels, class_to_idx, idx_to_class = filter_top_k_classes(
            image_paths=image_paths,
            labels=labels,
            idx_to_class=idx_to_class,
            top_k=config["top_k_classes"],
        )

    if config["use_class_grouping"]:
        image_paths, labels, class_to_idx, idx_to_class = group_similar_classes(
            image_paths=image_paths,
            labels=labels,
            idx_to_class=idx_to_class,
            class_groups=config["class_groups"],
        )

    num_classes = len(class_to_idx)
    print(f"Nombre de classes utilitzades: {num_classes}")

    wandb.config.update({
        "num_classes": num_classes,
        "num_images": len(image_paths),
    })

    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = split_dataset(
        image_paths=image_paths,
        labels=labels,
        val_size=config["val_size"],
        test_size=config["test_size"],
        random_state=config["seed"],
    )
    print_split_summary(train_labels, val_labels, test_labels)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_paths=train_paths,
        val_paths=val_paths,
        test_paths=test_paths,
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
        batch_size=config["batch_size"],
        image_size=config["image_size"],
        num_workers=config["num_workers"],
        resize_images=resize_images_in_dataloader,
        use_augmentation=config["use_augmentation"],
        use_weighted_sampler=config["use_weighted_sampler"],
        use_aspect_crop_cache=config["use_aspect_crop_cache"],
    )

    model = create_resnet_model(
        num_classes=num_classes,
        model_name=config["model_name"],
        feature_extraction=config["feature_extraction"],
        partial_finetuning=config["partial_finetuning"],
        unfreeze_layer3=config["unfreeze_layer3"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params_count}")
    print(f"Trainable ratio: {trainable_params_count / total_params:.4f}")

    wandb.config.update({
        "total_params": total_params,
        "trainable_params": trainable_params_count,
        "trainable_ratio": trainable_params_count / total_params,
    })

    criterion_train, criterion_eval = build_loss(
        config=config,
        train_labels=train_labels,
        num_classes=num_classes,
        idx_to_class=idx_to_class,
        device=device,
    )

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    scheduler = None
    if config["use_scheduler"]:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config["scheduler_factor"],
            patience=config["scheduler_patience"],
            min_lr=config["scheduler_min_lr"],
        )

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion_train=criterion_train,
        criterion_eval=criterion_eval,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        checkpoint_path=checkpoint_path,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        early_stopping_patience=config["early_stopping_patience"],
    )

    model = load_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    test_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion_eval,
        device=device,
        idx_to_class=idx_to_class,
        save_dir=os.path.join("results", "figures", config["experiment_name"]),
    )

    wandb.finish()


if __name__ == "__main__":
    main()
