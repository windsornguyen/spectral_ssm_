
# ==============================================================================#
# Authors: Isabel Liu
# File: loss_ant.py
# ==============================================================================#

"""Customized Loss for Ant-v1 Task."""

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
            if i in (0, 1, 2):  # coordinates of the torso (center)
                loss = loss / 5
                print(f'Index {i}, Coordinate Loss Scale /5: {loss.mean().item()}')
            elif i in (3, 4, 5, 6):  # orientations of the torso (center)
                loss = loss / 0.2
                print(f'Index {i}, Orientation Loss Scale /0.2: {loss.mean().item()}')
            elif i in (7, 8, 9, 10, 11, 12, 13, 14):  # angles between the torso and the links
                loss = loss / 0.5
                print(f'Index {i}, Angle Loss Scale /0.5: {loss.mean().item()}')
            elif i in (15, 16, 17, 18, 19, 20):  # coordinate and coordinate angular velocities of the torso (center)
                loss = loss / 2
                print(f'Index {i}, Velocity Loss Scale /2: {loss.mean().item()}')
            elif i in (21, 22, 23, 24, 25, 26, 27, 28):  # angular velocities of the angles between the torso and the links
                loss = loss / 5
                print(f'Index {i}, Angular Velocity Loss Scale /5: {loss.mean().item()}')

            total_loss += loss.mean()

        total_loss = total_loss / outputs.shape[1]
        metrics = {'loss': total_loss.item()}
        print(f'Total Scaled Loss: {total_loss.item()}')

        return total_loss, metrics