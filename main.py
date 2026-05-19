import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from utils.data_utils import (
    prepare_resized_cache_dataset,
    load_wikiart_dataset,
    print_dataset_summary,
    split_dataset,
    print_split_summary,
    create_dataloaders,
    filter_top_k_classes,
    cap_images_per_class,
    compute_class_weights,
)

from models.transfer_model import create_resnet_model, freeze_batchnorm_layers
from train import train_model
from test import test_model
from utils.losses import compute_class_priors, LogitAdjustedCrossEntropyLoss


def set_seed(seed):
    """
    Fixem llavors per fer l'experiment més reproduïble.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint(model, checkpoint_path, device):
    """
    Carrega els pesos del millor model guardat.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Checkpoint carregat: {checkpoint_path}")
    print(f"Epoch checkpoint: {checkpoint['epoch'] + 1}")
    print(f"Validation accuracy checkpoint: {checkpoint['val_accuracy']:.4f}")

    return model


def main():
    config = {
        "experiment_name": "exp23_resnet50_384_logit_adjustment",

        "dataset_root": "/home/datasets/wikiart/",
        "model_name": "resnet50",
        "feature_extraction": False,
        "partial_finetuning": True,
        "unfreeze_layer3": False,

        "epochs": 20,
        "batch_size": 32,

        "learning_rate": 1e-5,
        "learning_rate_fc": 5e-4,
        "learning_rate_backbone": 1e-5,

        "weight_decay": 5e-4,
        "early_stopping_patience": 3,

        "image_size": 384,
        "val_size": 0.15,
        "test_size": 0.15,
        "random_seed": 42,
        "num_workers": 8,

        "remove_duplicates": False,
        "check_corrupted": False,

        "use_resized_cache": True,
        "resized_cache_root": "/tmp/wikiart_384",
        "force_rebuild_cache": False,
        "cache_num_workers": 12,
        "cache_resize_mode": "square",

        "use_aspect_crop_cache": False,
        "exclude_paths_file": None,

        "use_top_k_classes": True,
        "top_k_classes": 14,

        "use_class_cap": False,
        "max_images_per_class": None,

        "use_class_weights": False,
        "use_weighted_sampler": False,

        "use_augmentation": True,
        "use_label_smoothing": True,
        "label_smoothing": 0.05,

        # EXP23
        "use_logit_adjustment": True,
        "logit_adjustment_tau": 1.0,

        "freeze_batchnorm": False,
        "use_differential_lr": False,

        "use_scheduler": True,
        "scheduler_factor": 0.5,
        "scheduler_patience": 1,
        "scheduler_min_lr": 1e-6,
    }

    set_seed(config["random_seed"])

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
            num_workers=config["cache_num_workers"],
            cache_resize_mode=config.get("cache_resize_mode", "square")
        )
        resize_images_in_dataloader = False

    # 1. Carregar dataset complet
    image_paths, labels, class_to_idx, idx_to_class, stats = load_wikiart_dataset(
        root_dir=dataset_root,
        remove_duplicates=config["remove_duplicates"],
        check_corrupted=config["check_corrupted"],
        exclude_paths_file=config.get("exclude_paths_file"),
    )

    print_dataset_summary(
        image_paths=image_paths,
        labels=labels,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        stats=stats,
    )
    # 1.1. Experiment 2: conservar només les top-k classes més grans
    if config["use_top_k_classes"]:
        image_paths, labels, class_to_idx, idx_to_class = filter_top_k_classes(
            image_paths=image_paths,
            labels=labels,
            idx_to_class=idx_to_class,
            top_k=config["top_k_classes"],
        )

        print("\nResum després de filtrar top-k classes:")
        print_dataset_summary(
            image_paths=image_paths,
            labels=labels,
            class_to_idx=class_to_idx,
            idx_to_class=idx_to_class,
            stats=stats,
        )
    # 1.2. Experiment 11: limitar les classes majoritàries
    if config["use_class_cap"]:
        image_paths, labels = cap_images_per_class(
            image_paths=image_paths,
            labels=labels,
            idx_to_class=idx_to_class,
            max_images_per_class=config["max_images_per_class"],
            random_seed=config["random_seed"],
        )

        print("\nResum després de capar classes majoritàries:")
        print_dataset_summary(
            image_paths=image_paths,
            labels=labels,
            class_to_idx=class_to_idx,
            idx_to_class=idx_to_class,
            stats=stats,
        )


    num_classes = len(class_to_idx)
    print(f"Nombre de classes utilitzades: {num_classes}")

    wandb.config.update({
        "num_classes": num_classes,
        "num_images": len(image_paths),
        "class_cap_applied": config["use_class_cap"],
        "max_images_per_class": config["max_images_per_class"] if config["use_class_cap"] else None,
    })

    
    # 2. Split train / validation / test
    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = split_dataset(
        image_paths=image_paths,
        labels=labels,
        val_size=config["val_size"],
        test_size=config["test_size"],
        random_state=config["random_seed"],
    )

    print_split_summary(train_labels, val_labels, test_labels)

    # 3. Crear DataLoaders
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
        use_aspect_crop_cache=config.get("use_aspect_crop_cache", False),
    )

    # 4. Crear model ResNet
    model = create_resnet_model(
        num_classes=num_classes,
        model_name=config["model_name"],
        feature_extraction=config["feature_extraction"],
        partial_finetuning=config["partial_finetuning"],
        unfreeze_layer3=config["unfreeze_layer3"],
    )
    
    model = model.to(device)

    if config.get("freeze_batchnorm", False):
        freeze_batchnorm_layers(model)
        print("BatchNorm congelada.")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params_count}")
    print(f"Trainable ratio: {trainable_params_count / total_params:.4f}")

    wandb.config.update({
        "total_params": total_params,
        "trainable_params": trainable_params_count,
        "trainable_ratio": trainable_params_count / total_params,
    })

    # 5. Loss
    # criterion_train és la loss que fem servir per aprendre.
    # criterion_eval és la loss "normal" per validation i test.

    label_smoothing = config["label_smoothing"] if config["use_label_smoothing"] else 0.0

    if config.get("use_logit_adjustment", False):
        class_priors = compute_class_priors(
            labels=train_labels,
            num_classes=num_classes,
            device=device,
        )

        criterion_train = LogitAdjustedCrossEntropyLoss(
            class_priors=class_priors,
            tau=config["logit_adjustment_tau"],
            label_smoothing=label_smoothing,
        )

        print("\n========== LOGIT ADJUSTMENT ==========")
        print(f"tau: {config['logit_adjustment_tau']}")
        for class_idx in range(num_classes):
            print(
                f"{idx_to_class[class_idx]}: "
                f"prior={class_priors[class_idx].item():.6f}"
            )
        print("======================================\n")

        wandb.config.update({
            "class_priors": class_priors.detach().cpu().tolist(),
            "logit_adjustment_tau": config["logit_adjustment_tau"],
        })

    elif config["use_class_weights"]:
        class_weights = compute_class_weights(
            labels=train_labels,
            num_classes=num_classes,
            idx_to_class=idx_to_class,
        ).to(device)

        criterion_train = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

        wandb.config.update({
            "class_weights": class_weights.detach().cpu().tolist(),
        })

    else:
        criterion_train = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
        )
    criterion_eval = nn.CrossEntropyLoss()

    # Només entrenem paràmetres amb requires_grad=True.
    # En feature extraction això serà principalment la capa fc final.
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    if config.get("use_differential_lr", False):
        optimizer = optim.AdamW(
            [
                {
                    "params": model.layer4.parameters(),
                    "lr": config["learning_rate_backbone"],
                },
                {
                    "params": model.fc.parameters(),
                    "lr": config["learning_rate_fc"],
                },
            ],
            weight_decay=config["weight_decay"],
        )

        print("\nOptimizer amb learning rates diferencials:")
        print(f"  layer4 lr: {config['learning_rate_backbone']}")
        print(f"  fc lr:     {config['learning_rate_fc']}")
        print(f"  weight decay: {config['weight_decay']}\n")

    else:
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
    #Experiment 14: scheduler de reducció de lr quan la val acc no millora. Molt útil per partial finetuning.
    scheduler = None

    if config["use_scheduler"]:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config["scheduler_factor"],
            patience=config["scheduler_patience"],
            min_lr=config["scheduler_min_lr"],
        )

    # 6. Entrenament amb checkpoint
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
        freeze_batchnorm=config.get("freeze_batchnorm", False),
    )

    # 7. Carregar millor checkpoint abans del test
    model = load_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    # 8. Test final
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
