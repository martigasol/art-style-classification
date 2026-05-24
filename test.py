import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import wandb
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm


def test_model(model, test_loader, criterion, device, idx_to_class=None, save_dir="results/figures"):
    """Avalua el millor checkpoint sobre test."""
    model.eval()

    running_loss = torch.zeros((), device=device, dtype=torch.float64)
    running_correct = torch.zeros((), device=device, dtype=torch.int64)
    running_total = 0

    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for images, labels in tqdm(test_loader, desc="Testing", mininterval=1.0):
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

    test_loss = (running_loss / running_total).item()
    test_acc = (running_correct.double() / running_total).item()
    all_preds = torch.cat(all_preds).cpu().tolist()
    all_labels = torch.cat(all_labels).cpu().tolist()

    class_names = None
    label_ids = None
    if idx_to_class is not None:
        label_ids = list(range(len(idx_to_class)))
        class_names = [idx_to_class[i] for i in label_ids]

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        labels=label_ids,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        all_labels,
        all_preds,
        labels=label_ids,
        average="weighted",
        zero_division=0,
    )

    print("\n========== TEST RESULTS ==========")
    print(f"Test Loss:   {test_loss:.4f}")
    print(f"Test Acc:    {test_acc:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("==================================\n")

    os.makedirs(save_dir, exist_ok=True)

    summary_path = os.path.join(save_dir, "test_summary.csv")
    pd.DataFrame([
        {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }
    ]).to_csv(summary_path, index=False)

    cm = confusion_matrix(all_labels, all_preds, labels=label_ids)
    fig, ax = plt.subplots(figsize=(14, 14))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )
    disp.plot(ax=ax, xticks_rotation=90, colorbar=True)
    plt.title("Confusion Matrix - Test Set")
    plt.tight_layout()

    confusion_matrix_path = os.path.join(save_dir, "confusion_matrix_test.png")
    plt.savefig(confusion_matrix_path, dpi=300)
    plt.close(fig)

    report_path = os.path.join(save_dir, "classification_report_test.txt")
    report = classification_report(
        all_labels,
        all_preds,
        labels=label_ids,
        target_names=class_names,
        zero_division=0,
    )
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)

    print(f"Test summary guardat a: {summary_path}")
    print(f"Confusion matrix guardada a: {confusion_matrix_path}")
    print(f"Classification report guardat a: {report_path}")

    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": wandb.Image(confusion_matrix_path),
    })

    return test_loss, test_acc, macro_f1, weighted_f1, all_preds, all_labels
