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


    def loss_fn(self, outputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the loss and metrics for a batch of data.

        Args:
            outputs (torch.Tensor): The model outputs.
            targets (torch.Tensor): The target labels.

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: 
              A Tuple of the loss and a Dictionary of metrics.
        """
        # loss = self.criterion(outputs, targets)
        # preds = torch.argmax(outputs, dim=1)
        # accuracy = torch.mean((preds == targets).float()).item() * 100
        # metrics = {'loss': loss.item(), 'accuracy': accuracy}

        total_loss = torch.tensor(0.0, device=outputs.device)
        for i in range(outputs.shape[0]):
            loss = (outputs[i]-targets[i])**2
            metrics = {'loss': loss.item()}
            
            # scaling by constant just for now
            if i in (0, 1, 2): # coordinates of the torso (center)
                loss = loss/5
            elif i in (3, 4, 5, 6): # orientations of the torso (center)
                loss = loss/0.2
            elif i in (7, 8, 9, 10, 11, 12, 13, 14): # angles between the torso and the links
                loss = loss/0.5
            elif i in (15, 16, 17, 18, 19, 20): # coordinate and coordinate angular velocities of the torso (center)
                loss = loss/2
            elif i in (21, 22, 23, 24, 25, 26, 27, 28): # angular velocities of the angles between the torso and the links
                loss = loss/5
            
            total_loss += loss
            
        total_loss = total_loss/outputs.shape[0]
        
        return loss, metrics


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
        # for name, param in self.model.named_parameters():
        #     if param.grad is None:
        #         print(name)
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
        losses = []
        # accuracies = []

        with torch.no_grad(), tqdm(
          total=len(dataloader), desc='Evaluating model...'
        ) as progress:
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss, metrics = self.loss_fn(outputs, targets)

                losses.append(loss.item())
                # accuracies.append(metrics['accuracy'])
                progress.update(1)

        avg_loss = torch.mean(torch.tensor(losses)).item()
        # avg_accuracy = torch.mean(torch.tensor(accuracies)).item()

        return {'loss': avg_loss}
