from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .degradations import RandomRealWorldDegradation


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(
    image_size: int = 224,
    training: bool = False,
    degradation_aware: bool = False,
):
    if training:
        ops = []
        if degradation_aware:
            ops.append(RandomRealWorldDegradation(p=0.65))

        ops.extend([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.82, 1.0),
                ratio=(0.90, 1.10),
            ),
            transforms.ColorJitter(
                brightness=0.18,
                contrast=0.18,
                saturation=0.12,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        return transforms.Compose(ops)

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_loaders(
    data_dir: str | Path,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 2,
    degradation_aware: bool = False,
):
    data_dir = Path(data_dir)

    train_ds = datasets.ImageFolder(
        data_dir / "train",
        transform=build_transforms(
            image_size=image_size,
            training=True,
            degradation_aware=degradation_aware,
        ),
    )
    val_ds = datasets.ImageFolder(
        data_dir / "val",
        transform=build_transforms(image_size=image_size, training=False),
    )
    test_ds = datasets.ImageFolder(
        data_dir / "test",
        transform=build_transforms(image_size=image_size, training=False),
    )

    if not (train_ds.classes == val_ds.classes == test_ds.classes):
        raise RuntimeError("Train/val/test class folders do not match.")

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
