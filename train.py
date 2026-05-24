import os

import torch
import wandb
from sklearn.metrics import f1_score
from tqdm import tqdm


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Forward, loss, backward i actualització de pesos."""
    model.train()

    running_loss = torch.zeros((), device=device, dtype=torch.float64)
    running_correct = torch.zeros((), device=device, dtype=torch.int64)
    running_total = 0

    for images, labels in tqdm(train_loader, desc="Training", leave=False, mininterval=1.0):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.detach().argmax(dim=1)
        running_loss += loss.detach().double() * images.size(0)
        running_correct += (preds == labels).sum()
        running_total += labels.size(0)

    epoch_loss = (running_loss / running_total).item()
    epoch_acc = (running_correct.double() / running_total).item()

    return epoch_loss, epoch_acc


def validate_one_epoch(model, val_loader, criterion, device):
    """Avalua validation sense actualitzar pesos."""
    model.eval()

    running_loss = torch.zeros((), device=device, dtype=torch.float64)
    running_correct = torch.zeros((), device=device, dtype=torch.int64)
    running_total = 0

    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for images, labels in tqdm(val_loader, desc="Validation", leave=False, mininterval=1.0):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

            running_loss += loss.double() * images.size(0)
            running_correct += (preds == labels).sum()
            running_total += labels.size(0)
            all_preds.append(preds)
            all_labels.append(labels)

    epoch_loss = (running_loss / running_total).item()
    epoch_acc = (running_correct.double() / running_total).item()
    all_preds = torch.cat(all_preds).cpu().tolist()
    all_labels = torch.cat(all_labels).cpu().tolist()
    epoch_macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return epoch_loss, epoch_acc, epoch_macro_f1


def save_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    epoch,
    val_loss,
    val_accuracy,
    val_macro_f1,
    config,
    class_to_idx,
    idx_to_class,
):
    """Guarda pesos, config i mapping de classes."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_macro_f1,
            "config": config,
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
        },
        checkpoint_path,
    )


def train_model(
    model,
    train_loader,
    val_loader,
    criterion_train,
    criterion_eval,
    optimizer,
    scheduler,
    config,
    device,
    checkpoint_path,
    class_to_idx,
    idx_to_class,
    early_stopping_patience=None,
):
    """Bucle complet amb checkpoint pel millor validation macro F1."""
    best_val_macro_f1 = 0.0
    epochs_without_improvement = 0
    epochs = config["epochs"]

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion_train,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_acc, val_macro_f1 = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion_eval,
            device=device,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Val Macro F1: {val_macro_f1:.4f} | LR: {current_lr:.2e}"
        )

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_macro_f1,
            "learning_rate": current_lr,
        })

        if scheduler is not None:
            scheduler.step(val_macro_f1)

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                val_accuracy=val_acc,
                val_macro_f1=val_macro_f1,
                config=config,
                class_to_idx=class_to_idx,
                idx_to_class=idx_to_class,
            )
            print(f"Nou millor model guardat a: {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            print(
                f"No millora validation macro F1 "
                f"({epochs_without_improvement}/{early_stopping_patience})"
            )

            if (
                early_stopping_patience is not None
                and epochs_without_improvement >= early_stopping_patience
            ):
                print(
                    "Early stopping activat. "
                    f"Sense millora durant {early_stopping_patience} epochs."
                )
                break

    print(f"\nMillor validation macro F1: {best_val_macro_f1:.4f}")
    return best_val_macro_f1
