# ==============================================================================#
# Authors: Windsor Nguyen, Dwaipayan Saha
# File: experiment.py
# ==============================================================================#

"""Utilities for running an experiment."""

import torch
import torch.nn as nn
from typing import Tuple, Dict
from torch.utils.data import DataLoader
from tqdm import tqdm


"""Utilities for running an experiment."""

import torch
import torch.nn as nn
from typing import Tuple, Dict
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
        self.device = device
        self.model.to(self.device)


    def loss_fn(self, outputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the loss and metrics for a batch of data.

        Args:
            outputs (torch.Tensor): The model outputs.
            targets (torch.Tensor): The target labels.

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: 
            A Tuple of the loss and a Dictionary of metrics.
        """
        total_loss = torch.tensor(0.0, device=outputs.device)
        for i in range(outputs.shape[1]):
            # Print statements to debug
            print(f"Processing output index {i}")
            print(f"outputs[:, {i}]: {outputs[:, i]}")
            print(f"targets[:, {i}]: {targets[:, i]}")

            loss = (outputs[:, i] - targets[:, i]) ** 2
            print(f"Initial loss at index {i}: {loss}")

            # scaling by constant just for now
            if i in (0, 1, 2): # coordinates of the torso (center)
                loss = loss / 5
            elif i in (3, 4, 5, 6): # orientations of the torso (center)
                loss = loss / 0.2
            elif i in (7, 8, 9, 10, 11, 12, 13, 14): # angles between the torso and the links
                loss = loss / 0.5
            elif i in (15, 16, 17, 18, 19, 20): # coordinate and coordinate angular velocities of the torso (center)
                loss = loss / 2
            elif i in (21, 22, 23, 24, 25, 26, 27, 28): # angular velocities of the angles between the torso and the links
                loss = loss / 5

            print(f"Scaled loss at index {i}: {loss}")

            total_loss += loss.mean()

        total_loss = total_loss / outputs.shape[1]
        print(f"Total loss: {total_loss}")

        metrics = {'loss': total_loss.item()}

        return total_loss, metrics


    def step(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Perform a single training step.

        Args:
            inputs (torch.Tensor): A batch of input data.
            targets (torch.Tensor): A batch of target labels.

        Returns:
            Dict[str, float]: A Dictionary of metrics for the training step.
        """
        self.model.train()
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss, metrics = self.loss_fn(outputs, targets)
        loss.backward()
        self.optimizer.step()

        return metrics

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model over an entire dataset.

        Args:
            dataloader (DataLoader): 
              A DataLoader providing batches of data for evaluation.

        Returns:
            Dict[str, float]: 
              A Dictionary of aggregated metrics over the dataset.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad(), tqdm(total=len(dataloader), desc='Evaluating model...') as progress:
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss, _ = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                num_batches += 1
                progress.update(1)

        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
