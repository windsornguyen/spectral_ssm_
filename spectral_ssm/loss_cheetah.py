
# ==============================================================================#
# Authors: Isabel Liu
# File: loss_cheetah.py
# ==============================================================================#

"""Customized Loss for HalfCheetah-v1 Task."""

import torch
from typing import Tuple, Dict

def loss_fn(outputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
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
            loss = (outputs[:, i] - targets[:, i]) ** 2

            # scaling by constant just for now
            if i in (0, 1):  # coordinates of the front tip
                loss = loss / 2.5
                print(f'Index {i}, Coordinate Loss Scale /2.5: {loss.mean().item()}')
            elif i in (2, 3, 4, 5, 6, 7, 8):  # angles of the front tip and limbs
                loss = loss / 0.5
                print(f'Index {i}, Angle Loss Scale /0.5: {loss.mean().item()}')
            elif i in (9, 10):  # coordinate velocities of the front tip
                loss = loss / 2
                print(f'Index {i}, Coordinate Velocity Loss Scale /2: {loss.mean().item()}')
            elif i in (11, 12, 13, 14, 15, 16, 17):  # angular velocities of the front tip and limbs
                loss = loss / 2.5
                print(f'Index {i}, Angular Velocity Loss Scale /2.5: {loss.mean().item()}')

            total_loss += loss.mean()

        total_loss = total_loss / outputs.shape[1]
        metrics = {'loss': total_loss.item()}
        print(f'Total Scaled Loss: {total_loss.item()}')

        return total_loss, metrics