# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Data pipeline for CIFAR10."""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# def prepare_data_loaders(self, dataset):
#         sampler = DistributedSampler(dataset)
#         loader = DataLoader(dataset, sampler=sampler, batch_size=...)
#         return loader


def get_dataset(split: str, batch_size: int) -> DataLoader:
    """
    Retrieves and preprocesses the CIFAR-10 dataset.

    Args:
      split (str): The dataset split to use ('train' or 'test').
      batch_size (int): The batch size to use.

    Returns:
      DataLoader: A DataLoader for the specified dataset split, with preprocessing.
    """

    # Normalization parameters specific to CIFAR-10
    means = [0.49139968, 0.48215841, 0.44653091]
    stds = [0.24703223, 0.24348513, 0.26158784]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ]
    )

    # Check the dataset split and create the corresponding dataset
    if split == "train":
        dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        shuffle = True
    elif split == "test":
        dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
        shuffle = False
    else:
        raise ValueError(f"Unknown split: {split}")

    # Create a DataLoader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True
    )

    return dataloader
