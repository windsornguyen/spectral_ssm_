# ==============================================================================#
# Authors: Windsor Nguyen, Dwaipayan Saha
# File: experiment.py
# ==============================================================================#

"""Utilities for running an experiment."""

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm


# TODO: Add mixed precision training (incompatible with torch.vmap it seems?)
class Experiment:
    """Initializes and maintains the experiment state."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        device: torch.device = None,
    ) -> None:
        """
        Initialize an experiment.

        Args:
            model (nn.Module): A PyTorch model.
            optimizer (torch.optim.Optimizer): A PyTorch optimizer.
            device (torch.device): The device to run the model on.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model.to(self.device)


    def step(self, inputs: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        """
        Perform a single training step.

        Args:
            inputs (torch.Tensor): A batch of input data.
            targets (torch.Tensor): A batch of target labels.

        Returns:
            dict[str, float]: A dictionary of metrics for the training step.
        """
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss, metrics = self.loss_fn(outputs, targets)
        metrics['loss'] = loss.item()
        loss.backward()

        # Compute gradient norm to monitor vanishing/exploding gradients
        total_norm = 0
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
            else:
                print(f'No gradient found: {name}')
        total_norm = total_norm ** 0.5
        metrics['grad_norm'] = total_norm

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return metrics


    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate the model over an entire dataset.

        Args:
            dataloader (DataLoader): 
              A DataLoader providing batches of data for evaluation.

        Returns:
            Dict[str, float]: 
              A Dictionary of aggregated metrics over the dataset.
        """
        self.model.eval()
        losses = []

        with torch.no_grad(), tqdm(total=len(dataloader), desc='Evaluating model...') as progress:
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                print('outputs', outputs)
                loss, _ = self.loss_fn(outputs, targets)
                losses.append(loss.item())

                progress.update(1)

        avg_loss = torch.tensor(losses).mean().item()

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(torch.tensor(avg_loss, device=self.device))
            avg_loss /= torch.distributed.get_world_size()

        self.model.train()

        return {'loss': avg_loss}
