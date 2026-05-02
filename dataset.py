import os
from collections import Counter

import torch
from torch.utils.data import DataLoader, random_split
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


def compute_class_weights(dataset: datasets.ImageFolder, glioma_boost: float = 1.5) -> torch.FloatTensor:
    """
    Compute class weights with optional boost for Glioma (class 0).
    
    Args:
        dataset: ImageFolder dataset
        glioma_boost: Multiplier for Glioma weight (default: 1.5x)
    
    Returns:
        Class weights tensor
    """
    counts = Counter(dataset.targets)
    total = len(dataset.targets)
    n_classes = len(dataset.classes)
    weights = [total / (n_classes * counts[i]) for i in range(n_classes)]
    
    # Boost Glioma weight (class 0 in alphabetical order: glioma, meningioma, notumor, pituitary)
    # This helps the model focus more on Glioma (which has lower recall)
    class_names = dataset.classes
    if "glioma" in class_names:
        glioma_idx = class_names.index("glioma")
        weights[glioma_idx] *= glioma_boost
        print(f"[Class Weights] Glioma boosted by {glioma_boost}x for better recall")
    
    return torch.FloatTensor(weights)


def get_dataloaders(
    train_dir: str,
    test_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    val_split: float = 0.2,
):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        train_dir: Path to training data (will be split into train/val)
        test_dir: Path to test data (kept separate for final evaluation)
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        val_split: Fraction of training data to use for validation (default: 0.2)
    
    Returns:
        train_loader, val_loader, test_loader, class_names, class_weights
    """
    # Load full training dataset
    full_train_dataset = datasets.ImageFolder(train_dir, transform=get_transforms(train=True))
    test_dataset = datasets.ImageFolder(test_dir, transform=get_transforms(train=False))
    
    # Split training into train and validation
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Compute class weights from full training set
    class_weights = compute_class_weights(full_train_dataset)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
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

    print(f"Classes    : {full_train_dataset.classes}")
    print(f"Train      : {len(train_dataset)} samples ({(1-val_split)*100:.0f}%)")
    print(f"Validation : {len(val_dataset)} samples ({val_split*100:.0f}%)")
    print(f"Test       : {len(test_dataset)} samples (held out)")
    print(f"Class weights: {class_weights.tolist()}")

    return train_loader, val_loader, test_loader, full_train_dataset.classes, class_weights
