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
    """Initializes and maintains the experiment state."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device = None,
    ) -> None:
        """Initialize an experiment.

        Args:
            model (nn.Module): A PyTorch model.
            optimizer (torch.optim.Optimizer): A PyTorch optimizer.
            device (torch.device): The device to run the model on.
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
        """Compute the loss and metrics for a batch of data.

        Args:
            outputs (torch.Tensor): The model outputs.
            targets (torch.Tensor): The target labels.

        Returns:
            tuple[torch.Tensor, dict[str, float]]: 
              A tuple of the loss and a dictionary of metrics.
        """
        loss = self.criterion(outputs, targets)
        preds = torch.argmax(outputs, dim=1)
        accuracy = torch.mean((preds == targets).float()).item() * 100
        metrics = {'loss': loss.item(), 'accuracy': accuracy}
        return loss, metrics


    def step(self, inputs: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        """Perform a single training step.

        Args:
            inputs (torch.Tensor): A batch of input data.
            targets (torch.Tensor): A batch of target labels.

        Returns:
            dict[str, float]: A dictionary of metrics for the training step.
        """
        self.model.train()
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss, metrics = self.loss_fn(outputs, targets)
        loss.backward()
        # for name, param in self.model.named_parameters():
        #     if param.grad is None:
        #         print(name)
        self.optimizer.step()

        return metrics


    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate the model over an entire dataset.

        Args:
            dataloader (DataLoader): 
              A DataLoader providing batches of data for evaluation.

        Returns:
            dict[str, float]: 
              A dictionary of aggregated metrics over the dataset.
        """
        self.model.eval()
        losses = []
        accuracies = []

        with torch.no_grad(), tqdm(
          total=len(dataloader), desc='Evaluating model...'
        ) as progress:
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss, metrics = self.loss_fn(outputs, targets)

                losses.append(loss.item())
                accuracies.append(metrics['accuracy'])
                progress.update(1)

        avg_loss = torch.mean(torch.tensor(losses)).item()
        avg_accuracy = torch.mean(torch.tensor(accuracies)).item()

        return {'loss': avg_loss, 'accuracy': avg_accuracy}
