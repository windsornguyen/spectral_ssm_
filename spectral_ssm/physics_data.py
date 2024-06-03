import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict

class PhysicsDataset(Dataset):
    """Custom Dataset for loading sequence data."""

    def __init__(self, input_file, target_file):
        """
        Args:
            input_file (str): Path to the numpy file containing inputs.
            target_file (str): Path to the numpy file containing targets.
        """
        self.inputs = np.load(input_file)  # Shape (n, 1000, 37), for testing n=5
        self.targets = np.load(target_file)  # Shape (n, 1000, 29)
        print(f"inputs shape: {self.inputs.shape}, targets shape: {self.targets.shape}")

    def __len__(self):
        """Denotes the total number of samples."""
        return len(self.inputs)

    def __getitem__(self, index):
        """Generates one sample of data."""
        x_t = self.inputs[index, :, :]  # Current state and action for entire trajectory
        x_t_plus_1 = self.targets[index, :, :]  # Next state for entire trajectory

        # Convert to tensors
        x_t = torch.tensor(x_t, dtype=torch.float32)
        x_t_plus_1 = torch.tensor(x_t_plus_1, dtype=torch.float32)

        return x_t, x_t_plus_1

def get_dataloader(input_file, target_file, batch_size, shuffle=True, num_workers=1):
    """Create a DataLoader for the given dataset."""
    dataset = PhysicsDataset(input_file, target_file)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # Useful if using CUDA, as it can speed up data transfer.
    )

# # Example usage
# input_file = 'input_data.npy'
# target_file = 'target_data.npy'
# batch_size = 10

# dataloader = get_dataloader(input_file, target_file, batch_size)
