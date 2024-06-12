# ==============================================================================#
# Authors: Windsor Nguyen, Dwaipayan Saha, Isabel Liu, Yagiz Devre
# File: model.py
# ==============================================================================#

"""Spectral temporal unit (STU) block."""

import torch
import torch.nn as nn
import time

from dataclasses import dataclass
from stu import stu_utils


@dataclass
class STUConfigs:
    d_model: int = 24
    d_target: int = 18
    num_layers: int = 6
    dropout: float = 0.25
    input_len: int = 1000
    num_eigh: int = 24
    auto_reg_k_u: int = 3
    auto_reg_k_y: int = 2
    learnable_m_y: bool = True
    loss_fn: nn.Module = None  # TODO: Ideally should not be None by default?


class STU(nn.Module):
    """Simple STU Layer.

    Args:
        d_out (int): Output dimension.
        input_len (int): Input sequence length.
        num_eigh (int): Number of eigenvalues and eigenvectors to use.
        auto_reg_k_u (int): Auto-regressive depth on the input sequence.
        auto_reg_k_y (int): Auto-regressive depth on the output sequence.
        learnable_m_y (bool): Whether the m_y matrix is learnable.
    """

    def __init__(
        self,
        d_out: int = 37,
        input_len: int = 1000,
        num_eigh: int = 24,
        auto_reg_k_u: int = 3,
        auto_reg_k_y: int = 2,
        learnable_m_y: bool = True,
    ) -> None:
        super(STU, self).__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.d_out = d_out
        self.eigh = stu_utils.get_top_hankel_eigh(
            input_len, num_eigh, self.device
        )
        self.l, self.k = input_len, num_eigh
        self.auto_reg_k_u = auto_reg_k_u
        self.auto_reg_k_y = auto_reg_k_y
        self.learnable_m_y = learnable_m_y
        self.m_x_var = 1.0 / (float(self.d_out) ** 0.5)

        if learnable_m_y:
            self.m_y = nn.Parameter(
                torch.zeros([self.d_out, self.auto_reg_k_y, self.d_out])
            )
        else:
            self.register_buffer(
                'm_y', torch.zeros([self.d_out, self.auto_reg_k_y, self.d_out])
            )

        self.m_u = nn.Parameter(
            stu_utils.get_random_real_matrix(
                (self.d_out, self.d_out, self.auto_reg_k_u), self.m_x_var
            )
        )

        self.m_phi = nn.Parameter(torch.zeros(self.d_out * self.k, self.d_out))

    def apply_stu(self, inputs):
        start_time = time.time()  # Start timing
        eig_vals, eig_vecs = self.eigh

        x_tilde = stu_utils.compute_x_tilde(inputs, (eig_vals, eig_vecs))
        print(f'Time for x_tilde computation: {time.time() - start_time:.4f}s')
        start_time = time.time()  # Reset timing

        # delta_phi = x_tilde @ self.m_phi
        delta_phi = torch.einsum('lk,kd->ld', x_tilde, self.m_phi)
        print(
            f'Time for delta_phi computation: {time.time() - start_time:.4f}s'
        )
        start_time = time.time()  # Reset timing

        delta_ar_u = stu_utils.compute_ar_x_preds(self.m_u, inputs)
        print(
            f'Time for delta_ar_u computation: {time.time() - start_time:.4f}s'
        )
        start_time = time.time()  # Reset timing

        y_t = stu_utils.compute_y_t(self.m_y, delta_phi + delta_ar_u)
        print(f'Time for y_t computation: {time.time() - start_time:.4f}s')

        return y_t

    # TODO: Remove vmap and handle batch sizes manually
    def forward(self, inputs):
        start_time = time.time()
        output = torch.vmap(self.apply_stu)(inputs)
        print(
            f'Total time for STU forward pass: {time.time() - start_time:.4f}s'
        )
        return output


class Architecture(nn.Module):
    """General model architecture based on STU blocks."""

    def __init__(self, config):
        super(Architecture, self).__init__()
        self.d_model = config.d_model
        self.d_target = config.d_target
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.input_len = config.input_len
        self.num_eigh = config.num_eigh
        self.auto_reg_k_u = config.auto_reg_k_u
        self.auto_reg_k_y = config.auto_reg_k_y
        self.learnable_m_y = config.learnable_m_y

        self.embedding = nn.Linear(self.d_model, self.d_model)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.d_model),
                    STU(
                        d_out=self.d_model,
                        input_len=self.input_len,
                        num_eigh=self.num_eigh,
                        auto_reg_k_u=self.auto_reg_k_u,
                        auto_reg_k_y=self.auto_reg_k_y,
                        learnable_m_y=self.learnable_m_y,
                    ),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.d_model, 2 * self.d_model),
                    nn.GLU(dim=-1),
                    nn.Dropout(self.dropout),
                )
                for _ in range(self.num_layers)
            ]
        )
        self.projection = nn.Linear(self.d_model, self.d_target)

    def forward(self, inputs):
        start_time = time.time()  # Start timing for the embedding operation
        x = self.embedding(inputs)
        embedding_time = time.time() - start_time
        print(f'Time for embedding: {embedding_time:.4f}s')

        total_layer_time = 0

        for i, layer in enumerate(self.layers):
            start_time = time.time()  # Start timing for each layer
            z = x
            x = layer(x)
            x = x + z
            layer_time = time.time() - start_time
            total_layer_time += layer_time
            print(f'Time for layer {i}: {layer_time:.4f}s')

        print(f'Total time for all layers: {total_layer_time:.4f}s')

        start_time = time.time()  # Start timing for the final projection
        output = self.projection(x)
        projection_time = time.time() - start_time
        print(f'Time for final projection: {projection_time:.4f}s')
        return output

    def predict(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        init: int = 0,
        t: int = 1,
    ) -> tuple[list[float], tuple[torch.Tensor, dict[str, float]]]:
        """
        Predicts the next states in trajectories and computes losses against the targets.

        Args:
            inputs (torch.Tensor): A tensor of shape [num_trajectories, seq_len, d_in].
            targets (torch.Tensor): A tensor of shape [num_trajectories, seq_len, d_out].
            init (int): The index of the initial state to start at.
            t (int): The number of time steps to predict.

        Returns:
            A tuple containing the list of predicted states after `t` time steps and
            a tuple containing the total loss and a dictionary of metrics.
        """
        device = inputs.device
        num_trajectories, seq_len, d_in = inputs.size()

        predicted_sequence = []
        total_loss = torch.tensor(0.0, device=device)
        metrics = {
            'loss': [],
            'coordinate_loss': [],
            'orientation_loss': [],
            'angle_loss': [],
            'coordinate_velocity_loss': [],
            'angular_velocity_loss': [],
        }

        for i in range(t):
            current_input_state = inputs[:, init + i, :].unsqueeze(1)
            current_target_state = targets[:, init + i, :].unsqueeze(1)

            # Predict the next state using the model
            next_state = self.model(current_input_state)
            loss, metric = self.loss_fn(next_state, current_target_state)

            predicted_sequence.append(next_state.squeeze(1).tolist())

            # Accumulate the metrics
            for key in metrics:
                metrics[key].append(metric[key])

            # Accumulate the losses
            total_loss += loss.item()

        total_loss /= t

        return predicted_sequence, (total_loss, metrics)
