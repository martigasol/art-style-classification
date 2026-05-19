import torch
import torch.nn as nn
from collections import Counter


def compute_class_priors(labels, num_classes, device=None):
    """
    Calcula la prior de cada classe a partir de les labels de train.

    prior[c] = nombre d'imatges de la classe c / total imatges train
    """

    counts = Counter(labels)
    total = len(labels)

    priors = torch.zeros(num_classes, dtype=torch.float32)

    for class_idx in range(num_classes):
        priors[class_idx] = counts[class_idx] / total

    # Evitem log(0), per seguretat.
    priors = torch.clamp(priors, min=1e-8)

    if device is not None:
        priors = priors.to(device)

    return priors


class LogitAdjustedCrossEntropyLoss(nn.Module):
    """
    CrossEntropy amb Logit Adjustment.

    Durant train, ajustem els logits segons la prior de cada classe:

        adjusted_logits = logits + tau * log(class_priors)

    Això ajuda a reduir el biaix cap a classes majoritàries sense fer
    oversampling ni usar class weights forts.
    """

    def __init__(self, class_priors, tau=1.0, label_smoothing=0.0):
        super().__init__()

        self.register_buffer(
            "logit_adjustment",
            tau * torch.log(class_priors)
        )

        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        )

    def forward(self, logits, targets):
        adjusted_logits = logits + self.logit_adjustment
        loss = self.criterion(adjusted_logits, targets)

        return loss