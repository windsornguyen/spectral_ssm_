# =============================================================================#
# Authors: Windsor Nguyen
# File: (Transformer) base.py
#
# Vanilla Transformer.
#
# =============================================================================#

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass
class TransformerConfig:
    n_layer: int = 6
    n_head: int = 1
    n_embd: int = 37
    d_out: int = 29
    max_len: int = 1_000
    dropout: float = 0.25
    loss_fn: nn.Module = None


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config

        self.transformer = nn.Transformer(
            d_model=config.n_embd,
            nhead=config.n_head,
            num_encoder_layers=config.n_layer,
            num_decoder_layers=config.n_layer,
            dim_feedforward=4 * config.n_embd,
            dropout=config.dropout,
            batch_first=True,
        )

        self.regression_head = nn.Linear(config.n_embd, config.d_out, bias=False)
        self.loss_fn = self.config.loss_fn

        # Report the number of parameters
        print("Model parameter count: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, inputs, targets=None):
        """
        inputs: (batch_size, seq_len, d_in)
        """
        device = inputs.device
        batch_size, seq_len, d_in = inputs.size()
        assert seq_len <= self.config.max_len, f"Cannot forward sequence of length {seq_len}, block size is only {self.config.max_len}"

        # Forward the Transformer model itself
        x = self.transformer(inputs, inputs)
        preds = self.regression_head(x)

        if targets is not None:
            loss = self.loss_fn(preds, targets)
        else:
            loss = None
        return preds, loss

    def predict(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        init: int = 0,
        steps: int = 100,
        ar_steps: int = 1,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        """
        Predicts the next states for a given set of input trajectories using vectorized operations.

        Args:
            inputs (torch.Tensor): A tensor of input trajectories with shape [num_trajectories, seq_len, d_in].
            targets (torch.Tensor): A tensor of target trajectories with shape [num_trajectories, seq_len, d_out].
            init (int): The index of the initial state to start the prediction from. Defaults to 0.
            steps (int): The number of time steps to predict. Defaults to 100.
            ar_steps (int): The number of autoregressive steps to take before using the ground truth state.
                Defaults to 1, which means the model always uses the ground truth state to predict the next state.

        Returns:
            tuple[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]]:
                - preds (torch.Tensor): A tensor of predicted states for each trajectory after `steps` time steps,
                    with shape [num_trajectories, steps, d_out].
                - loss (tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]): A tuple containing:
                    - avg_loss (torch.Tensor): The mean loss over time steps and trajectories.
                    - avg_metrics (dict[str, torch.Tensor]): A dictionary of mean losses for each metric.
                    - trajectory_losses (torch.Tensor): A tensor of losses for each trajectory at each time step,
                        with shape [num_trajectories, steps].
        """
        device = next(self.parameters()).device
        print(f'Predicting on {device}.')
        num_trajectories, seq_len, d_in = inputs.size()

        # Initialize the predicted sequences and losses
        ar_sequences = inputs.clone()
        preds = torch.zeros(num_trajectories, steps, self.config.d_out, device=device)
        trajectory_losses = torch.zeros(num_trajectories, steps, device=device)
        metrics = {
            key: torch.zeros(num_trajectories, steps, device=device) for key in 
                ['coordinate_loss', 'orientation_loss', 'angle_loss', 
                'coordinate_velocity_loss', 'angular_velocity_loss']
        }

        # Initialize initial autoregressive sequences up to `init` steps for each trajectory
        ar_sequences[:, :init+1, :] = inputs[:, :init+1, :]
        u_start = targets.shape[2]
        
        # Iterate over the specified number of time steps
        for i in tqdm(range(steps), desc='Predicting', unit='step'):
            xs = ar_sequences[:, :i + 1 + init, :]
            ys = targets[:, :i + 1 + init, :]
            
            preds_step, (step_loss, step_metrics) = self.forward(xs, ys)

            preds[:, i, :] = preds_step[:, i, :]

            # Update autoregressive sequences, keeping control vectors intact
            if i < steps - 1:
                next_input = ar_sequences[:, i + 1 + init, :]
                
                next_input[:, :u_start] = preds[:, i, :] if (i + 1) % ar_steps != 0 else inputs[:, i + 1 + init, :u_start]
                ar_sequences[:, i + 1 + init, :] = next_input

            trajectory_losses[:, i] = step_loss

            for key in metrics:
                metrics[key][:, i] = step_metrics[key]

        # Calculate average losses and metrics across trajectories
        avg_loss = trajectory_losses.mean()

        loss = (avg_loss, metrics, trajectory_losses)
        return preds, loss
