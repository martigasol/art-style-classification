import torch.nn as nn
from torchvision import models


def create_resnet_model(
    num_classes,
    model_name="resnet50",
    feature_extraction=True,
    partial_finetuning=False,
    unfreeze_layer3=False,
):
    """Crea una ResNet50 preentrenada i adapta la capa final."""
    if model_name != "resnet50":
        raise ValueError(f"Model no suportat: {model_name}. Fem servir resnet50.")

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    if partial_finetuning:
        # Fine-tuning parcial: congelem tot i entrenem layer4 + fc.
        for param in model.parameters():
            param.requires_grad = False

        if unfreeze_layer3:
            for param in model.layer3.parameters():
                param.requires_grad = True

        for param in model.layer4.parameters():
            param.requires_grad = True

    elif feature_extraction:
        # Feature extraction: només aprèn la capa final.
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
