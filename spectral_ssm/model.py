# ==============================================================================#
# Authors: Windsor Nguyen, Dwaipayan Saha
# File: model.py
# ==============================================================================#

"""Spectral temporal unit (STU) block."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from spectral_ssm import stu_utils

import functools


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
    print(f'Input shape before compute_x_tilde: {inputs.shape}')
    x_tilde = stu_utils.compute_x_tilde(inputs, eigh)
    print(f'x_tilde shape after compute_x_tilde: {x_tilde.shape}')

    # Compute deltas from the spectral filters, which are of shape [l, d_out].
    delta_phi = x_tilde @ m_phi

    # Compute deltas from AR on x part
    delta_ar_u = stu_utils.compute_ar_x_preds(m_u, inputs)

    print(f'x_tilde shape after conv: {x_tilde.shape}')
    print(f'delta_phi shape: {delta_phi.shape}')
    print(f'delta_ar_u shape: {delta_ar_u.shape}')

    # Compute y_ts, which are of shape [l, d_out].
    return stu_utils.compute_y_t(m_y, delta_phi + delta_ar_u)


class STU(nn.Module):
    """Simple STU Layer."""

    def __init__(
        self,
        d_out: int = 256,
        input_len: int = 32 * 32,
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

        print(f'Shape before operation X: {inputs.shape}')
        params = (self.m_y, self.m_u, self.m_phi)
        return apply_stu(params, inputs, self.eigh)


# class Architecture(nn.Module):
#     """
#     General model architecture.
#     """

#     def __init__(
#         self,
#         d_model=256,
#         d_target=10,
#         num_layers=6,
#         dropout=0.1,
#         input_len=1024,
#         num_eigh=24,
#         auto_reg_k_u=3,
#         auto_reg_k_y=2,
#         learnable_m_y=True,
#     ):
#         """
#         Initialize general model architecture.

#         Args:
#         name: Name of the module.
#         d_model: Dimension of the embedding.
#         d_target: Dimension of the target.
#         num_layers: Number of layers.
#         dropout: Dropout rate.
#         input_len: Input sequence length.
#         num_eigh: Number of eigen values and vecs.
#         auto_reg_k_u: Auto-regressive depth on the input sequence.
#         auto_reg_k_y: Auto-regressive depth on the output sequence.
#         learnable_m_y: m_y matrix learnable.
#         """
#         super(Architecture, self).__init__()
#         self.d_model = d_model
#         self.d_target = d_target
#         self.num_layers = num_layers
#         self.dropout = dropout
#         self.input_len = input_len
#         self.num_eigh = num_eigh
#         self.auto_reg_k_u = auto_reg_k_u
#         self.auto_reg_k_y = auto_reg_k_y
#         self.learnable_m_y = learnable_m_y

#         self.embedding = nn.Conv1d(in_channels=3, out_channels=d_model, kernel_size=1)
#         self.batch_norms = nn.ModuleList(
#             [nn.BatchNorm1d(input_len) for _ in range(num_layers)]
#         )

#         # STU modules
#         self.layers = nn.ModuleList(
#             [
#                 nn.Sequential(
#                     nn.BatchNorm1d(d_model),
#                     STU(
#                         d_out=d_model,
#                         input_len=input_len,
#                         num_eigh=num_eigh,
#                         auto_reg_k_u=auto_reg_k_u,
#                         auto_reg_k_y=auto_reg_k_y,
#                         learnable_m_y=learnable_m_y,
#                     ),
#                     nn.Linear(d_model, 2 * d_model),
#                 )
#                 for _ in range(num_layers)
#             ]
#         )

#         # Final projection layer
#         self.projection = nn.Linear(d_model, d_target)

#     def forward(self, inputs, is_training=True):
#         # Flatten input for embedding
#         batch_size, channels, height, width = inputs.shape
#         inputs = inputs.view(batch_size, channels, height * width)
#         x = self.embedding(inputs)
#         print(f'Embedding output shape: {x.shape}')
#         for i in range(self.num_layers):
#             z = x
#             x = x.permute(0, 2, 1)
#             print(f'Before BatchNorm shape: {x.shape}')
#             x = self.batch_norms[i](x)
#             print(f'After BatchNorm shape: {x.shape}')
#             x = x.permute(0, 2, 1)
#             x = self.layers[i](x)
#             print(f'After STU Layer {i}: {x.shape}')
#             x = F.gelu(x)
#             if is_training:
#                 x = F.dropout(x, p=self.dropout, training=is_training)
#             # x = self.expand_dims[i](x)
#             torch.reshape(x, (batch_size, -1, 2 * self.d_model))
#             x = F.glu(x, dim=-1)
#             if is_training:
#                 x = F.dropout(x, p=self.dropout, training=is_training)
#             x = x + z  # Residual connection

#         x = torch.mean(x, dim=1, keepdim=True)
#         x = self.projection(x.squeeze(1))
#         return x

# class Architecture(nn.Module):
#     """
#     General model architecture.
#     """

#     def __init__(
#         self,
#         d_model=256,
#         d_target=10,
#         num_layers=6,
#         dropout=0.1,
#         input_len=1024,
#         num_eigh=24,
#         auto_reg_k_u=3,
#         auto_reg_k_y=2,
#         learnable_m_y=True,
#     ):
#         """
#         Initialize general model architecture.

#         Args:
#         d_model: Dimension of the embedding.
#         d_target: Dimension of the target.
#         num_layers: Number of layers.
#         dropout: Dropout rate.
#         input_len: Input sequence length.
#         num_eigh: Number of eigen values and vecs.
#         auto_reg_k_u: Auto-regressive depth on the input sequence.
#         auto_reg_k_y: Auto-regressive depth on the output sequence.
#         learnable_m_y: m_y matrix learnable.
#         """
#         super(Architecture, self).__init__()
#         self.d_model = d_model
#         self.d_target = d_target
#         self.num_layers = num_layers
#         self.dropout = dropout
#         self.input_len = input_len
#         self.num_eigh = num_eigh
#         self.auto_reg_k_u = auto_reg_k_u
#         self.auto_reg_k_y = auto_reg_k_y
#         self.learnable_m_y = learnable_m_y
#         self.embedding = nn.Linear(3, d_model)
#         self.batch_norms = nn.ModuleList(
#             [nn.BatchNorm1d(input_len) for _ in range(num_layers)]
#         )
        
#         self.layers = nn.ModuleList(
#             [
#                 nn.Sequential(
#                     nn.BatchNorm1d(input_len),
#                     STU(
#                         d_out=d_model,
#                         input_len=input_len,
#                         num_eigh=num_eigh,
#                         auto_reg_k_u=auto_reg_k_u,
#                         auto_reg_k_y=auto_reg_k_y,
#                         learnable_m_y=learnable_m_y,
#                     ),
#                     nn.Linear(input_len, 2 * input_len),
#                 )
#                 for _ in range(num_layers)
#             ]
#         )

#         # Final projection layer
#         self.projection = nn.Linear(d_model, d_target)

#     def forward(self, inputs, is_training=True):
#         # Reshape input for embedding
#         batch_size, channels, height, width = inputs.shape
#         inputs = inputs.view(batch_size, height * width, channels)
#         x = self.embedding(inputs)
#         print(f'Embedding output shape: {x.shape}')

#         for i in range(self.num_layers):
#             z = x
#             print(f'Before BatchNorm {i} shape: {x.shape}')
#             x = self.batch_norms[i](x)
#             print(f'After BatchNorm shape: {x.shape}')
#             print(f'Before STU Layer {i} shape: {x.shape}')
#             x = self.layers[i](x)
#             print(f'After STU Layer {i} shape: {x.shape}')
#             x = F.gelu(x)
#             if is_training:
#                 x = F.dropout(x, p=self.dropout, training=is_training)
#             x = x.view(batch_size, -1, 2 * self.input_len)
#             x = F.glu(x, dim=-1)
#             if is_training:
#                 x = F.dropout(x, p=self.dropout, training=is_training)
#             x = x + z  # Residual connection

#         x = torch.mean(x, dim=1, keepdim=True)
#         x = self.projection(x.squeeze(1))
#         return x
    
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
        input_len=32 * 32,
        num_eigh=24,
        auto_reg_k_u=3,
        auto_reg_k_y=2,
        learnable_m_y=True,
    ):
        """
        Initialize general model architecture.

        Args:
        d_model: Dimension of the embedding.
        d_target: Dimension of the target.
        num_layers: Number of layers.
        dropout: Dropout rate.
        input_len: Input sequence length.
        num_eigh: Number of eigen values and vecs.
        auto_reg_k_u: Auto-regressive depth on the input sequence.
        auto_reg_k_y: Auto-regressive depth on the output sequence.
        learnable_m_y: m_y matrix learnable.
        """
        super(Architecture, self).__init__()
        self.d_model = d_model
        self.d_target = d_target
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_len = input_len
        self.num_eigh = num_eigh
        self.auto_reg_k_u = auto_reg_k_u
        self.auto_reg_k_y = auto_reg_k_y
        self.learnable_m_y = learnable_m_y
        self.embedding = nn.Linear(3, d_model)
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(d_model) for _ in range(num_layers)]
        )

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    STU(
                        d_out=d_model,
                        input_len=input_len,
                        num_eigh=num_eigh,
                        auto_reg_k_u=auto_reg_k_u,
                        auto_reg_k_y=auto_reg_k_y,
                        learnable_m_y=learnable_m_y,
                    ),
                    nn.Linear(d_model, 2 * d_model),
                )
                for _ in range(num_layers)
            ]
        )

        # Final projection layer.
        self.projection = nn.Linear(d_model, d_target)

    def forward(self, inputs, is_training=True):
        # Reshape input for embedding
        batch_size, channels, height, width = inputs.shape
        x = inputs.view(batch_size, channels, height * width).permute(0, 2, 1)

        # Embedding layer.
        x = self.embedding(x)
        print(f'Embedding output shape: {x.shape}')

        for i in range(self.num_layers):
            # Saving input to layer for residual.
            z = x
            
            # Construct pre-layer batch norm.
            print(f'Before BatchNorm {i} shape: {x.shape}')
            x = self.batch_norms[i](x.permute(0, 2, 1)).permute(0, 2, 1)
            print(f'After BatchNorm shape: {x.shape}')

            # Apply STU layer.
            print(f'Before STU Layer {i} shape: {x.shape}')
            x = self.layers[i][0](x)
            print(f'After STU Layer {i} shape: {x.shape}')

            # GeLU + Dropout.
            x = F.gelu(x)
            if is_training:
                x = F.dropout(x, p=self.dropout, training=is_training)

            # Apply linear layer.
            x = self.layers[i][1](x)
            x = F.glu(x, dim=-1)

            # Dropout.
            if is_training:
                x = F.dropout(x, p=self.dropout, training=is_training)

            # Residual connection.
            x = x + z

        # Projection.
        x = torch.mean(x, dim=1)
        x = self.projection(x)
        return x
