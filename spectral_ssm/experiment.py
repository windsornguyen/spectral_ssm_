# ==============================================================================#
# Authors: Windsor Nguyen, Dwaipayan Saha
# File: experiment.py
# ==============================================================================#

"""Utilities for running an experiment."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


class Experiment:
    """
    Initializes and maintains the experiment state.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
    ) -> None:
        """
        Initializes an experiment.

        Args:
          model: A PyTorch model.
          optimizer: A PyTorch optimizer.
          loss_fn: A function to compute the loss given outputs and targets.
          device: The device to run the model on.
        """
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

    def loss_fn(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Computes the loss and metrics for a batch of data. (Right?)

        Args:
          inputs: A batch of inputs.
          targets: A batch of targets.
          training: Whether the model is in training mode.

        Returns:
          A tuple of the loss and a dictionary of metrics.
        """
        criterion = nn.CrossEntropyLoss(reduction='sum')
        loss = criterion(outputs, targets)

        probs = F.softmax(outputs, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        correct = torch.sum((preds == targets).float())
        count = targets.numel()

        metrics = {'loss': loss.item(), 'correct': correct.item(), 'count': count}
        return loss, metrics

    def step(self, inputs: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        """
        Takes a single step of the experiment: forward, backward, and optimize.

        Args:
          inputs: A batch of inputs.
          targets: A batch of targets.

        Returns:
          metrics: A dictionary of metrics including loss and accuracy.
        """
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss, metrics = self.loss_fn(outputs, targets)
        loss.backward()
        self.optimizer.step()
        accuracy = 100.0 * metrics['correct'] / metrics['count']
        metrics['accuracy'] = accuracy

        return metrics

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """
        Evaluates the model over an entire epoch.

        Args:
          dataloader: A DataLoader providing batches of data for evaluation.

        Returns:
          epoch_metrics: A dictionary of aggregated metrics over the epoch.
        """
        self.model.eval()
        total_count, total_loss, total_correct = 0, 0, 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss, batch_metrics = self.loss_fn(outputs, targets)

                total_count += batch_metrics['count']
                total_loss += loss.item() * batch_metrics['count']
                total_correct += batch_metrics['correct']

        avg_loss = total_loss / total_count
        accuracy = 100.0 * total_correct / total_count
        self.model.train()
        return {'count': total_count, 'loss': avg_loss, 'accuracy': accuracy}
