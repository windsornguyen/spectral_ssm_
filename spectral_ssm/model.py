# ==============================================================================#
# Authors: Windsor Nguyen, Dwaipayan Saha, Isabel Liu, Yagiz Devre
# File: model.py
# ==============================================================================#

"""Spectral temporal unit (STU) block."""

import functools
import torch
import torch.nn as nn
from spectral_ssm import stu_utils


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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.d_out = d_out
        self.eigh = stu_utils.get_top_hankel_eigh(input_len, num_eigh, self.device)
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
            stu_utils.get_random_real_matrix((d_out, d_out, auto_reg_k_u), self.m_x_var)
        )

        self.m_phi = nn.Parameter(torch.zeros(500 * d_out * num_eigh, d_out))

    def apply_stu(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the STU transformation to the input tensor.

        Args:
            inputs (torch.Tensor): Input tensor of shape (L, d_in).

        Returns:
            torch.Tensor: Output tensor of shape (L, d_out).
        """
        eig_vals, eig_vecs = self.eigh
        eig_vals = eig_vals.to(inputs.device)
        eig_vecs = eig_vecs.to(inputs.device)
        self.m_phi = self.m_phi.to(inputs.device)
        self.m_u = self.m_u.to(inputs.device)
        self.m_y = self.m_y.to(inputs.device)

        x_tilde = stu_utils.compute_x_tilde(inputs, (eig_vals, eig_vecs))
        delta_phi = x_tilde @ self.m_phi
        delta_ar_u = stu_utils.compute_ar_x_preds(self.m_u, inputs)
        y_t = stu_utils.compute_y_t(self.m_y, delta_phi + delta_ar_u)

        return y_t


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the STU layer.

        Args:
            inputs (torch.Tensor): Input tensor of shape (B, L, d_in) where B is batch size,
                L is sequence length, d_in is the number of input features (channels).

        Returns:
            torch.Tensor: Output tensor of shape (B, L, d_out).
        """
        outputs = torch.vmap(self.apply_stu)(inputs)
        return outputs[:, -1, :]  # Take the last output along the sequence dimension


class Architecture(nn.Module):
    """
    General model architecture.
    """

    def __init__(
        self,
        d_model=37,
        d_target=29,
        num_layers=6,
        dropout=0.1,
        input_len=1000,
        num_eigh=24,
        auto_reg_k_u=3,
        auto_reg_k_y=2,
        learnable_m_y=True,
    ):
        """Initialize general model architecture.

        Args:
          d_model: Dimension of the embedding.
          d_target: Dimension of the target.
          num_layers: Number of layers.
          dropout: Dropout rate.
          input_len: Input sequence length.
          num_eigh: Number of eigenvalues and eigenvectors to use.
          auto_reg_k_u: Auto-regressive depth on the input sequence.
          auto_reg_k_y: Auto-regressive depth on the output sequence.
          learnable_m_y: Whether the m_y matrix is learnable.
        """
        super(Architecture, self).__init__()
        self.embedding = nn.Linear(d_model, d_model)
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_layers)]
        )
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    STU(
                        d_model,
                        input_len,
                        num_eigh,
                        auto_reg_k_u,
                        auto_reg_k_y,
                        learnable_m_y,
                    ),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, 2 * d_model),
                    nn.GLU(dim=-1),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ]
        )

        self.projection = nn.Linear(d_model, d_target)


    def forward(self, inputs):
        """Forward pass.

        Args:
          inputs: Input tensor of shape (B, C, H, W),
            where B is the batch size, C is the number of channels,
            H is the height, and W is the width.

        Returns:
          Output tensor of shape (B, d_target), where d_target is the target dimension.
        """
        x = self.embedding(inputs)

        for i, layer in enumerate(self.layers):
            z = x
            x = self.layer_norms[i](x)
            x = layer(x)
            x = x + z

        return self.projection(x)
