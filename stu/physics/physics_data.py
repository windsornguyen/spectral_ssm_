import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


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


def get_dataloader(
    inputs, 
    targets, 
    batch_size,
    device,
    shuffle=False, 
    distributed=True, 
    rank=0, 
    num_replicas=1, 
    num_workers=1, 
    pin_memory=True
) -> DataLoader:
    """Create a DataLoader for the given dataset."""
    dataset = PhysicsDataset(inputs, targets)

    sampler = DistributedSampler(
        dataset=dataset,
        num_replicas=num_replicas,
        rank=rank
    ) if distributed else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=(sampler is None),
    )

# # Example usage
# input_file = 'input_data.npy'
# target_file = 'target_data.npy'
# batch_size = 10

# dataloader = get_dataloader(input_file, target_file, batch_size)
