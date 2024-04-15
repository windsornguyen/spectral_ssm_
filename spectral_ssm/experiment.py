# ==============================================================================#
# Authors: Windsor Nguyen, Dwaipayan Saha
# File: experiment.py
# ==============================================================================#

"""Utilities for running an experiment."""

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm


class Experiment:
    """
    Initializes and maintains the experiment state.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device = None,
    ) -> None:
        """
        Initializes an experiment.

        Args:
          model: A PyTorch model.
          optimizer: A PyTorch optimizer.
          device: The device to run the model on.
        """
        self.model = model
        self.optimizer = optimizer

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        # TODO: Write GitHub Issue for PyTorch-MPS,
        # does not yet support reductions for tensors w/ rank > 4 :(
        # https://github.com/google/jax/issues/20112
        # elif torch.backends.mps.is_available():
        #     self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def loss_fn(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Computes the loss and metrics for a batch of data.

        Args:
          outputs: The model outputs.
          targets: The target labels.

        Returns:
          A tuple of the loss and a dictionary of metrics.
        """
        loss = self.criterion(outputs, targets)

        _, preds = torch.max(outputs, dim=1)
        correct = (preds == targets).sum().item()
        total = targets.size(0)
        accuracy = 100.0 * correct / total

        metrics = {'loss': loss.item(), 'accuracy': accuracy}
        return loss, metrics

    def step(self, inputs: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        """
        Performs a single training step: forward pass, backward pass, and optimization.

        Args:
          inputs: A batch of input data.
          targets: A batch of target labels.

        Returns:
          A dictionary of metrics for the training step.
        """
        self.model.train()
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss, metrics = self.loss_fn(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return metrics

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """
        Evaluates the model over an entire dataset.

        Args:
          dataloader: A DataLoader providing batches of data for evaluation.

        Returns:
          A dictionary of aggregated metrics over the dataset.
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc="Evaluating model..."):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss, metrics = self.loss_fn(outputs, targets)

                total_loss += loss.item() * targets.size(0)
                total_correct += metrics['accuracy'] / 100.0 * targets.size(0)
                total_samples += targets.size(0)

        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples

        return {'loss': avg_loss, 'accuracy': avg_accuracy}
