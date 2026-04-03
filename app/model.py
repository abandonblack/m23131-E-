from __future__ import annotations

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    """构建 ResNet18 分类器。

    Args:
        num_classes: 目标类别数。
        pretrained: 是否加载 ImageNet 预训练权重。
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model
