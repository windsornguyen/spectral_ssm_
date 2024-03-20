# =============================================================================#
# Authors: Windsor Nguyen, Dwaipayan Saha
# File: cifar10.py
# =============================================================================#

"""Data pipeline for CIFAR10."""

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, DistributedSampler


class Transform:
    def __init__(self):
        self.means: torch.Tensor = torch.tensor([0.49139968, 0.48215841, 0.44653091])
        self.stds: torch.Tensor = torch.tensor([0.24703223, 0.24348513, 0.26158784])

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # Normalize by dataset-specific statistics
        return (img - self.means[:, None, None]) / self.stds[:, None, None]


def get_transforms() -> transforms.Compose:
    """Prepare the transformation pipeline for the CIFAR10 dataset."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            Transform(),
        ]
    )


def get_dataset(
    split: str, batch_size: int, num_workers: int = 4, pin_memory: bool = True
) -> DataLoader:
    """Retrieves a CIFAR10 dataset with preprocessing."""
    transform = get_transforms()

    is_train: bool = split == 'train'
    dataset: Dataset = datasets.CIFAR10(
        root='./data', train=is_train, download=True, transform=transform
    )

    # Ensure the DistributedSampler is used only when necessary
    sampler: DistributedSampler = (
        DistributedSampler(dataset, shuffle=is_train)
        if torch.distributed.is_initialized()
        else None
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None) and is_train,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return dataset
