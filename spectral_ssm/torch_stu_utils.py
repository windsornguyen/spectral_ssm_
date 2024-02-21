# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for spectral SSM."""

import torch

def get_device():
  "Returns the CUDA device if available, else defaults to CPU."
  return torch.device('cuda') if torch.cuda_is_available() else torch.device('cpu')

@torch.jit.script
def get_hankel_matrix(
  n: int
) -> torch.Tensor:
  """Generate a spectral Hankel matrix.

  Args:
    n: Number of rows in square spectral Hankel matrix.

  Returns:
    A spectral Hankel matrix of shape [n, n].
  """
  row_indices = torch.arange(1, n + 1).unsqueeze(0)
  column_indices = torch.arange(1, n + 1).unsqueeze(1)
  return 2 / ((row_indices + column_indices) ** 3 - (row_indices + column_indices))

@torch.jit.script
def get_top_hankel_eigh(
    n: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Get top k eigenvalues and eigenvectors of spectral Hankel matrix.

  Args:
    n: Number of rows in square spectral Hankel matrix.
    k: Number of eigenvalues to return.

  Returns:
    A tuple of eigenvalues of shape [k,] and eigenvectors of shape [l, k].
  """
  hankel = get_hankel_matrix(n)
  eig_vals, eig_vecs = torch.linalg.eigh(hankel)
  top_k = torch.argsort(torch.abs(eig_vals)[-k:])
  return eig_vals[top_k], eig_vecs[:, top_k]

@torch.jit.script
def get_random_real_matrix(
    shape: tuple[int, ...],
    scaling: float,
    lower: float = -2.0,
    upper: float = 2.0,
) -> torch.Tensor:
  """Generate a random real matrix.

  Args:
    shape: Shape of matrix to generate.
    scaling: Scaling factor.
    lower: Lower trunctation of random matrix.
    upper: Upper trunctation of random matrix.

  Returns:
    A random real matrix.
  """
  random = torch.randn(shape)
  random_bounded = torch.clamp(random, min=lower, max=upper)
  return scaling * random_bounded


@torch.jit.script
def shift(
    x: torch.Tensor,
) -> torch.Tensor:
  """Shift time axis by one to align x_{t-1} and x_t.

  Args:
    x: An array of shape [l, d].

  Returns:
    An array of shape [l, d] where index [0, :] is all zeros and [i, :] is equal
    to x[i - 1, :] for i > 0.
  """
  return torch.cat([torch.zeros((1, x.shape[1]), device=x.device), x], dim=0)[:-1, :]


@torch.jit.script
def conv(
  v: torch.Tensor,
  u: torch.Tensor
) -> torch.Tensor:
  """Compute convolution to project input sequences into the spectral basis.

  Args:
      v: Top k eigenvectors of shape [l, k].
      u: Input of shape [l, d_in].

  Returns:
      A matrix of shape [l, k, d_in]
  """
  device = get_device()
  if v.device != device: 
    v = v.to(device)
  if u.device != device:
    u = u.to(device) 

  # Convolve two vectors of length l (x.shape[0]) and truncate to the l oldest
  # values.
  tr_conv = lambda x, y: torch.fft.irfft(torch.fft.rfft(x) * torch.fft.rfft(y))[
    :x.shape[0]
  ]
  
  # Convolve each sequence of length l in v with each sequence in u
  mvconv = torch.vmap(truncate_conv, in_dims=(1, None), out_dims=1)
  mmconv = torch.vmap(mvconv, in_dims=(None, 1), out_dims=-1)

  return mmconv(v, u)


@torch.jit.script
def compute_y_t(
    m_y: torch.Tensor,
    deltas: torch.Tensor,
) -> torch.Tensor:
  """Compute sequence of y_t given a series of deltas and m_y via a simple scan.

  Args:
    m_y: A matrix of shape [d_out, k, d_out] that acts as windowed transition
      matrix for the linear dynamical system evolving y_t.
    deltas: A matrix of shape [l, d_out].

  Returns:
    A matrix of shape [l, d_out].
  """
  d_out, k, _ = m_y.shape
  carry = torch.zeros((k, d_out))
  ys = torch.empty((deltas.shape[0], k, d_out))

  for i, x in enumerate(deltas):
    output = torch.tensordot(m_y, carry, dims=2) + x
    carry = torch.roll(carry, shifts=1, dims=0)
    carry[0] = output
    ys[i] = output

  return ys


@torch.jit.script
def compute_ar_x_preds(
    w: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
  """Compute the auto-regressive component of spectral SSM.

  Args:
    w: An array of shape [d_out, d_in, k].
    x: A single input sequence of shape [l, d_in].

  Returns:
    ar_x_preds: An output of shape [l, d_out].
  """
  d_out, _, k = w.shape
  l = x.shape[0]

  # Contract over `d_in`.
  o = torch.einsum('oik,li->klo', w, x)

  # For each `i` in `k`, roll the `(l, d_out)` by `i` steps.
  roll = lambda tensor, shift: torch.roll(tensor, shift=shift, dims=0)
  o = torch.vmap(roll, in_dims=(0, 0))(o, jnp.arange(k))

  # Create a mask that zeros out nothing at `k=0`, the first `(l, d_out)` at
  # `k=1`, the first two `(l, dout)`s at `k=2`, etc.
  m = torch.triu(torch.ones((k, l), device=w.device))
  m = m.unsqueeze(-1)
  m = m.expand(-1, -1, d_out)

  # Mask and sum along `k`.
  return torch.sum(o * m, axis=0)


@torch.jit.script
def compute_x_tilde(
  inputs: torch.Tensor,
  eigh: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
  """Project input sequence into spectral basis.

  Args:
    inputs: A single input sequence of shape [l, d_in].
    eigh: A tuple of eigenvalues [k] and circulant eigenvecs [k, l, l].

  Returns:
    x_tilde: An output of shape [l, k * d_in].
  """
  eig_vals, eig_vecs = eigh

  l = inputs.shape[0]
  x_tilde = conv(eig_vecs, inputs)

  # Broadcast an element-wise multiple along the k-sized axis.
  x_tilde *= torch.unsqueeze(eig_vals, axis=(0, 2)) ** 0.25

  # This shift is introduced as the rest is handled by the AR part.
  return shift(shift(x_tilde.reshape((l, -1))))
