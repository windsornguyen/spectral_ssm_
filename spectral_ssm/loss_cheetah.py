# ==============================================================================#
# Authors: Isabel Liu
# File: loss_cheetah.py
# ==============================================================================#

"""Customized Loss for HalfCheetah-v1 Task."""

import torch
import torch.nn as nn
from typing import Tuple, Dict

class HalfCheetahLoss(nn.Module):
    def __init__(self):
        super(HalfCheetahLoss, self).__init__()

    def forward(
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
            """
            Compute the loss and metrics for a batch of data.

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
                    coordinate_loss += loss.mean()
                elif i in (2, 3, 4, 5, 6, 7, 8):  # angles of the front tip and limbs
                    loss = loss / 0.5
                    angle_loss += loss.mean()
                elif i in (9, 10):  # coordinate velocities of the front tip
                    loss = loss / 2
                    coordinate_velocity_loss += loss.mean()
                elif i in (11, 12, 13, 14, 15, 16, 17):  # angular velocities of the front tip and limbs
                    loss = loss / 2.5
                    angular_velocity_loss += loss.mean()

                total_loss += loss.mean()

            total_loss /= outputs.shape[1]
            coordinate_loss /= 2
            angle_loss /= 7
            coordinate_velocity_loss /= 2
            angular_velocity_loss /= 7

            metrics = {'loss': total_loss.item(), 'coordinate_loss': coordinate_loss.item(), 'angle_loss': angle_loss.item(), 'coordinate_velocity_loss': coordinate_velocity_loss.item(), 'angular_velocity_loss': angular_velocity_loss.item()}

            return total_loss, metrics
