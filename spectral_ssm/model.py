# ==============================================================================#
# Authors: Windsor Nguyen, Dwaipayan Saha
# File: model.py
# ==============================================================================#

"""Spectral temporal unit (STU) block."""

import functools

import torch
import torch.nn as nn

from spectral_ssm import stu_utils


@functools.partial(torch.vmap, in_dims=(None, 0, None))
def apply_stu(
    params: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    inputs: torch.Tensor,
    eigh: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Apply STU.

    Args:
      params: A tuple of parameters of shapes [d_out, d_out], [d_in, d_out, k_u],
        [d_in * k, d_out] and [d_in * k, d_out]
      inputs: Input matrix of shape [l, d_in].
      eigh: A tuple of eigenvalues [k] and circulant eigenvecs [k, l, l].

    Returns:
      A sequence of y_ts of shape [l, d_out].
    """
    m_y, m_u, m_phi = params
    x_tilde = stu_utils.compute_x_tilde(inputs, eigh).to(inputs.device)

    # Compute deltas from the spectral filters, which are of shape [l, d_out].
    delta_phi = x_tilde @ m_phi

    # Compute deltas from AR on x part
    delta_ar_u = stu_utils.compute_ar_x_preds(m_u, inputs)

    # Compute y_ts, which are of shape [l, d_out].
    return stu_utils.compute_y_t(m_y, delta_phi + delta_ar_u)


class STU(nn.Module):
    """Simple STU Layer."""

    def __init__(
        self,
        d_out: int = 256,
        input_len: int = 32 * 32,  # CIFAR-10 dimensions
        num_eigh: int = 24,
        auto_reg_k_u: int = 3,
        auto_reg_k_y: int = 2,
        learnable_m_y: bool = True,
    ) -> None:
        """Initialize STU layer.

        Args:
          d_out: Output dimension.
          input_len: Input sequence length.
          num_eigh: Tuple of eigenvalues and vectors sized (k,) and (l, k)
          auto_reg_k_u: Auto-regressive depth on the input sequence,
          auto_reg_k_y: Auto-regressive depth on the output sequence,
          learnable_m_y: m_y matrix learnable,
        """
        super(STU, self).__init__()
        self.d_out = d_out
        self.eigh = stu_utils.get_top_hankel_eigh(input_len, num_eigh)
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

        # NOTE: Assume d_in = d_out
        self.m_u = nn.Parameter(
            stu_utils.get_random_real_matrix((d_out, d_out, auto_reg_k_u), self.m_x_var)
        )
        self.m_phi = nn.Parameter(torch.zeros(d_out * num_eigh, d_out))

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
          inputs: Assumed to be of shape (B, L, H) where B is batch size, L is
            sequence length, H is number of features (channels) in the input.

        Returns:
          `torch.Tensor` of preactivations.
        """

        params = (self.m_y, self.m_u, self.m_phi)
        return apply_stu(params, inputs, self.eigh)


class Architecture(nn.Module):
    """
    General model architecture.
    """

    def __init__(
        self,
        d_model=256,
        d_target=10,
        num_layers=6,
        dropout=0.1,
        input_len=32 * 32,  # CIFAR-10 dimensions
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
        self.embedding = nn.Linear(3, d_model)
        # TODO:
        # Make sure to use nn.DataParallel or nn.DistributedDataParallel) to
        # handle synchronization of batch normalization stats across replicas
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(d_model, momentum=0.1) for _ in range(num_layers)]
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
                )
                for _ in range(num_layers)
            ]
        )
        self.embedding = nn.Linear(3, d_model)
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
        # Reshape input for embedding
        batch_size, channels, height, width = inputs.shape
        x = inputs.view(batch_size, channels, height * width).permute(0, 2, 1)

        # Embedding layer.
        x = self.embedding(x)

        for i, layer in enumerate(self.layers):
            # Saving input to layer for residual.
            z = x

            # Batch norm layer.
            x = self.batch_norms[i](x.permute(0, 2, 1)).permute(0, 2, 1)

            # Apply sequential layers + residual connection.
            x = layer(x) + z

        # Projection.
        x = x.mean(dim=1)
        return self.projection(x)
