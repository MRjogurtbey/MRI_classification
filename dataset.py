import os
from collections import Counter

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(train: bool) -> transforms.Compose:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])


def compute_class_weights(dataset: datasets.ImageFolder) -> torch.FloatTensor:
    counts = Counter(dataset.targets)
    total = len(dataset.targets)
    n_classes = len(dataset.classes)
    weights = [total / (n_classes * counts[i]) for i in range(n_classes)]
    return torch.FloatTensor(weights)


def get_dataloaders(
    train_dir: str,
    test_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
):
    train_dataset = datasets.ImageFolder(train_dir, transform=get_transforms(train=True))
    test_dataset = datasets.ImageFolder(test_dir, transform=get_transforms(train=False))

    class_weights = compute_class_weights(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Classes : {train_dataset.classes}")
    print(f"Train   : {len(train_dataset)} samples")
    print(f"Test    : {len(test_dataset)} samples")
    print(f"Class weights: {class_weights.tolist()}")

    return train_loader, test_loader, train_dataset.classes, class_weights
