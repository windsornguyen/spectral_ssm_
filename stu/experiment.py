# ==============================================================================#
# Authors: Windsor Nguyen, Dwaipayan Saha
# File: experiment.py
# ==============================================================================#

"""Utilities for running an experiment."""

import torch
import torch.nn as nn

from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

class Experiment:
    """
    Initializes and maintains the experiment state.
    """

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

        self.scaler = GradScaler()

    def step(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, float]:
        """
        Perform a single training step.

        Args:
            inputs (torch.Tensor): A batch of input data.
            targets (torch.Tensor): A batch of target labels.

        Returns:
            dict[str, float]: A dictionary of metrics for the training step.
        """
        metrics = {}
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        # start_total = time.time()

        self.optimizer.zero_grad()

        with autocast():
            # start_forward = time.time()
            outputs = self.model(inputs)
            # end_forward = time.time()
            # print(f'Time for forward pass: {end_forward - start_forward:.4f}s')

            # start_loss = time.time()
            loss, metrics = self.loss_fn(outputs, targets)
            # end_loss = time.time()
            # print(f'Time for loss computation: {end_loss - start_loss:.4f}s')
            metrics['loss'] = loss.item()

        # start_backward = time.time()
        self.scaler.scale(loss).backward()
        # end_backward = time.time()
        # print(f'Time for backward pass: {end_backward - start_backward:.4f}s')
        
        # Unscale the gradients to report gradient norm
        # start_unscale = time.time()
        self.scaler.unscale_(self.optimizer)
        # end_unscale = time.time()
        # print(f'Time for unscaling gradients: {end_unscale - start_unscale:.4f}s')

        # Compute gradient norm to monitor vanishing/exploding gradients
        # start_grad_norm = time.time()
        total_norm = 0
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm**2
                print(f'{name}: {param_norm}')
            else:
                print(f'No gradient found: {name}')
        total_norm = total_norm**0.5
        metrics['grad_norm'] = total_norm
        # end_grad_norm = time.time()
        # print(f'Time for computing gradient norm: {end_grad_norm - start_grad_norm:.4f}s')

        # start_step = time.time()
        self.scaler.step(self.optimizer)
        # end_step = time.time()
        # print(f'Time for optimizer step: {end_step - start_step:.4f}s')

        # start_update = time.time()
        self.scaler.update()
        # end_update = time.time()
        # print(f'Time for scaler update: {end_update - start_update:.4f}s')

        if self.scheduler is not None:
            # start_scheduler = time.time()
            self.scheduler.step()
            # end_scheduler = time.time()
            # print(f'Time for scheduler step: {end_scheduler - start_scheduler:.4f}s')

        # end_total = time.time()
        # print(f'Total time for step function: {end_total - start_total:.4f}s')

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

        with (
            torch.no_grad(),
            autocast(),
            tqdm(total=len(dataloader), desc='Evaluating model...') as progress,
        ):
            for inputs, targets in dataloader:
                inputs, targets = (
                    inputs.to(self.device),
                    targets.to(self.device),
                )

                outputs = self.model(inputs)
                loss, _ = self.loss_fn(outputs, targets)
                losses.append(loss.item())

                progress.update(1)

        avg_loss = torch.tensor(losses).mean().item()

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                torch.tensor(avg_loss, device=self.device)
            )
            avg_loss /= torch.distributed.get_world_size()

        self.model.train()

        return {'loss': avg_loss}
