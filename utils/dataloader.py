# =============================================================================#
# Authors: Windsor Nguyen, Yagiz Devre, Isabel Liu
# File: dataloader.py
# =============================================================================#

"""Custom dataloader for loading sequence data in a distributed manner."""

import torch

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.colors import Colors, colored_print


# TODO: Write generic dataset downloading and saving script for the user.
# TODO Add a mask option for mujoco-v3 task.
class Dataloader(Dataset):
    def __init__(self, data, task, preprocess=True, eps=1e-7):
        self.data = data
        self.task = task
        self.preprocess = preprocess
        self.eps = eps

        if self.preprocess:
            colored_print('Calculating data statistics...', Colors.OKBLUE)
            self._calculate_statistics()
            colored_print('Normalizing data...', Colors.OKBLUE)
            self._normalize_data()
            colored_print('Validating data normalization...', Colors.OKBLUE)
            self._validate_normalization()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.task == 'mujoco-v3':
            features = self.data[index]
        else:
            features = torch.cat(
                (self.data[index]['x_t'], self.data[index]['x_t_plus_1']),
                dim=-1,
            )

        input_frames = features[:-1]  # All frames except the last one
        target_frames = features[1:]  # All frames except the first one

        return input_frames, target_frames, index

    def _calculate_statistics(self):
        if self.task == 'mujoco-v3':
            features = torch.cat(self.data, dim=0)
        else:
            features = torch.cat(
                [
                    torch.cat(
                        (self.data[i]['x_t'], self.data[i]['x_t_plus_1']),
                        dim=-1,
                    )
                    for i in range(len(self.data))
                ],
                dim=0,
            )

        # Mean over samples and frames, for each feature
        self.mean = features.mean(dim=(0, 1), keepdim=True)

        # Std over samples and frames, for each feature
        self.std = features.std(dim=(0, 1), keepdim=True)

    def _normalize_data(self):
        for i in range(len(self.data)):
            if self.task == 'mujoco-v3':
                self.data[i] = (self.data[i] - self.mean) / (
                    self.std + self.eps
                )
            else:
                self.data[i]['x_t'] = (
                    self.data[i]['x_t']
                    - self.mean[..., : self.data[i]['x_t'].shape[-1]]
                ) / (self.std[..., : self.data[i]['x_t'].shape[-1]] + self.eps)
                self.data[i]['x_t_plus_1'] = (
                    self.data[i]['x_t_plus_1']
                    - self.mean[..., self.data[i]['x_t'].shape[-1] :]
                ) / (self.std[..., self.data[i]['x_t'].shape[-1] :] + self.eps)

    def _validate_normalization(self):
        # Check if the mean and standard deviation are close to the desired values
        if self.task == 'mujoco-v3':
            features = torch.cat(self.data, dim=0)
        else:
            features = torch.cat(
                [
                    torch.cat(
                        (self.data[i]['x_t'], self.data[i]['x_t_plus_1']),
                        dim=-1,
                    )
                    for i in range(len(self.data))
                ],
                dim=0,
            )

        normalized_mean = features.mean(dim=(0, 1))
        normalized_std = features.std(dim=(0, 1))

        assert torch.allclose(
            normalized_mean, torch.zeros_like(normalized_mean), atol=self.eps
        ), f'Normalized mean is not close to zero: {normalized_mean}'
        assert torch.allclose(
            normalized_std, torch.ones_like(normalized_std), atol=self.eps
        ), f'Normalized standard deviation is not close to one: {normalized_std}'

        # Print out mean and standard deviation
        colored_print(f'Normalized mean: {normalized_mean}', Colors.OKGREEN)
        colored_print(
            f'Normalized standard deviation: {normalized_std}', Colors.OKGREEN
        )
        colored_print(
            'Data normalization validated successfully.',
            Colors.BOLD + Colors.OKGREEN,
        )


def get_dataloader(
    data,
    task,
    batch_size,
    num_workers,
    preprocess=True,
    shuffle=True,
    pin_memory=True,
    distributed=True,
    rank=0,
    world_size=1,
    prefetch_factor=2,
    persistent_workers=True,
):
    colored_print(f'Creating dataloader for task: {task}', Colors.OKBLUE)
    dataset = Dataloader(data, task, preprocess)

    sampler = (
        DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
        )
        if distributed
        else None
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=(shuffle and sampler is None),
        pin_memory=pin_memory,
        sampler=sampler,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    colored_print('Dataloader created successfully.', Colors.OKGREEN)
    return dataloader


def split_data(dataset, train_ratio=0.8, random_seed=1337):
    data = dataset['video_representations']

    # TODO: Not sure if we need this, experiment to see if data is the same without
    # Set the random seed for reproducibility
    torch.manual_seed(random_seed)

    # Shuffle the data using PyTorch
    indices = torch.randperm(len(data))
    shuffled_data = [data[i] for i in indices]

    # Split the data into training and validation sets
    num_train = int(len(data) * train_ratio)
    train_data = shuffled_data[:num_train]
    val_data = shuffled_data[num_train:]

    colored_print(
        f'Data split into {len(train_data)} training samples and {len(val_data)} validation samples.',
        Colors.OKBLUE,
    )

    return train_data, val_data
