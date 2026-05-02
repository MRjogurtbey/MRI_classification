import torch.nn as nn
from torchvision import models


def get_model(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    try:
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
    except ImportError:
        backbone = models.resnet18(pretrained=pretrained)

    in_features = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(512, num_classes),
    )

    total = sum(p.numel() for p in backbone.parameters())
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"Model: ResNet-18  |  Total params: {total:,}  |  Trainable: {trainable:,}")

    return backbone
