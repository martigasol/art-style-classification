import torch.nn as nn
from torchvision import models



def create_resnet_model(
    num_classes,
    model_name="resnet18",
    feature_extraction=True,
    partial_finetuning=False,
    unfreeze_layer3=False,
):
    """
    Crea una ResNet preentrenada i adapta l'última capa al nostre nombre de classes.

    Modes principals:

    feature_extraction=True, partial_finetuning=False:
        congela tota la ResNet i només entrena la capa final fc.

    partial_finetuning=True:
        congela la major part de la ResNet, però descongela layer4 + fc.
        Això permet adaptar les últimes capes al domini WikiArt.

    feature_extraction=False, partial_finetuning=False:
        entrena tota la ResNet.
        No ho fem ara perquè és més costós i té més risc d'overfitting.
    """

    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Model no suportat: {model_name}")

    if partial_finetuning:
        # Primer congelem tota la xarxa.
        for param in model.parameters():
            param.requires_grad = False

        # EXP16:
        # Opcionalment descongelem layer3.
        # Això permet adaptar característiques més intermèdies al domini WikiArt.
        if unfreeze_layer3:
            for param in model.layer3.parameters():
                param.requires_grad = True

        # Sempre descongelem layer4 en fine-tuning parcial.
        # layer4 és el bloc més proper a la classificació final.
        for param in model.layer4.parameters():
            param.requires_grad = True
            
    elif feature_extraction:
        # Mode anterior: només entrenem la capa final.
        for param in model.parameters():
            param.requires_grad = False

    # Canviem la capa final perquè classifiqui els estils de WikiArt.
    # Aquesta capa sempre queda entrenable per defecte.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


import torch.nn as nn


def freeze_batchnorm_layers(model):
    """
    Congela totes les capes BatchNorm del model.

    Això evita que durant el fine-tuning s'actualitzin les estadístiques
    running_mean i running_var de BatchNorm.

    És útil en transfer learning perquè les estadístiques apreses a ImageNet
    poden ser més estables que recalcular-les amb el nostre dataset.
    """

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

            for param in module.parameters():
                param.requires_grad = False