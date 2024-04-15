# =============================================================================#
# Authors: Windsor Nguyen, Dwaipayan Saha
# File: cifar10.py
# =============================================================================#

"""Data pipeline for CIFAR10."""

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class Transform:
    """
    Class for normalizing images with dataset-specific statistics.
    """
    def __init__(self):
        self.means: torch.Tensor = torch.tensor([0.49139968, 0.48215841, 0.44653091])
        self.stds: torch.Tensor = torch.tensor([0.24703223, 0.24348513, 0.26158784])

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return (img - self.means[:, None, None]) / self.stds[:, None, None]

def get_transforms(is_train: bool = True) -> transforms.Compose:
    """
    Prepare the transformation pipeline for the CIFAR10 dataset.
    Includes data augmentations if is_train is True.

    Args:
        is_train (bool): Flag to indicate whether to apply training specific transforms.

    Returns:
        transforms.Compose: A composition of transformations.
    """
    basic_transforms = [
        transforms.ToTensor(),
        Transform()  # Normalization
    ]

    augmentation_transforms = [
        transforms.RandomResizedCrop(32, scale=(0.08, 1.0)),  # Random cropping and resizing back to 32x32
        transforms.RandomHorizontalFlip(),  # Random horizontal flipping
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),  # Color augmentation
    ]

    return transforms.Compose(augmentation_transforms + basic_transforms if is_train else basic_transforms)

def get_dataset(split: str) -> Dataset:
    """
    Retrieves the CIFAR10 dataset with preprocessing and augmentation for the training set.

    Args:
        split (str): The dataset split to use, either 'train' or 'test'.

    Returns:
        Dataset: The CIFAR10 dataset with the specified split, with appropriate preprocessing.
    """
    is_train: bool = split == 'train'
    transform = get_transforms(is_train)
    dataset: Dataset = datasets.CIFAR10(
        root='./data', train=is_train, download=True, transform=transform
    )

    return dataset

def get_dataloader(dataset: Dataset, batch_size: int, num_workers: int = 4, pin_memory: bool = True) -> DataLoader:
    """
    Creates a DataLoader for the given dataset.

    Args:
        dataset (Dataset): The dataset to create a DataLoader for.
        batch_size (int): The number of samples per batch to load.
        num_workers (int, optional): Number of subprocesses to use for data loading.
        pin_memory (bool, optional): If True, the data loader will copy Tensors into CUDA pinned memory before returning them.

    Returns:
        DataLoader: A DataLoader instance for the given dataset.
    """
    sampler = DistributedSampler(dataset)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
