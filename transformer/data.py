import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict

class PhysicsDataset(Dataset):
    """Custom Dataset for loading sequence data."""

    def __init__(self, input_file, target_file, device):
        """
        Args:
            input_file (str): Path to the   numpy file containing inputs.
            target_file (str): Path to the numpy file containing targets.
            ctxt_len (int): Context length for each sample.
        """
        self.device = device
        
        # Shape (batch_size, 1000, 37), for testing batch_size=10
        self.inputs = torch.tensor(np.load(input_file), dtype=torch.float32).to(device)
        
        # Shape (batch_size, 1000, 29)
        self.targets = torch.tensor(np.load(target_file), dtype=torch.float32).to(device)


    def __len__(self):
        """Denotes the total number of samples."""
        return len(self.inputs)

    def __getitem__(self, index):
        """Generates one sample of data."""
        x_t = self.inputs[index]  # Current state and action for entire trajectory
        x_t_plus_1 = self.targets[index]  # Next state for entire trajectory

        return x_t, x_t_plus_1

# TODO: num_workers > 0 breaks it
def get_dataloader(train_path, val_path, split, batch_size, device, shuffle=False, num_workers=0):
    """Create a DataLoader for the given dataset."""
    dataset = PhysicsDataset(train_path, val_path, device)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        # pin_memory=True  # Useful if using CUDA, as it can speed up data transfer.
    )
