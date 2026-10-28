from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    efficientnet_b0,
    EfficientNet_B0_Weights,
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
)


SUPPORTED_MODELS = ["resnet50", "efficientnet_b0", "mobilenet_v3_large"]


def build_model(
    name: str,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    name = name.lower()

    if name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if name == "mobilenet_v3_large":
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_large(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model '{name}'. Choose from {SUPPORTED_MODELS}.")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_model(
        ckpt["model_name"],
        num_classes=len(ckpt["classes"]),
        pretrained=False,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt
