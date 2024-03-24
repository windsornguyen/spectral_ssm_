# =============================================================================#
# Authors: Windsor Nguyen, Dwaipayan Saha
# File: stu_utils.py
# =============================================================================#

"""Utilities for spectral SSM."""

import torch
import torch.nn.functional as F

@torch.jit.script
def get_hankel_matrix(n: int) -> torch.Tensor:
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
    top_k_eig_vals = eig_vals[..., -k:]
    top_k_eig_vecs = eig_vecs[..., -k:]
    return top_k_eig_vals, top_k_eig_vecs

@torch.jit.script
def get_random_real_matrix(
    shape: list[int],
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
    """
    Shift time axis by one to align x_{t-1} and x_t.

    Args:
      x: A tensor of shape [l, d].

    Returns:
      A tensor of shape [l, d] where index [0, :] is all zeros and [i, :] is equal
      to x[i - 1, :] for i > 0.
    """
    # Pad the tensor with one row of zeros at the beginning
    padded = F.pad(x, (0, 0, 1, 0), 'constant', 0.0)
    # Remove the last row to maintain the original shape
    shifted = padded[:-1, :]
    return shifted


@torch.jit.script
def tr_conv(v, u):
    """
    Perform truncated convolution using FFT.

    Args:
        v (torch.Tensor): Tensor of shape [l,].
        u (torch.Tensor): Tensor of shape [l,].

    Returns:
        torch.Tensor: Convolution result of shape [l,].
    """
    # Convolve two vectors of length l (x.shape[0])
    v_fft = torch.fft.rfft(v, dim=0)
    u_fft = torch.fft.rfft(u, dim=0)
    output_fft = v_fft * u_fft

    # Truncate to the l oldest values.
    return torch.fft.irfft(output_fft, n=v.size(0), dim=0)[: v.size(0)]


def conv(v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    Compute convolution to project input sequences into the spectral basis.

    Args:
        v: Top k eigenvectors of shape [l, k].
        u: Input of shape [l, d_in].

    Returns:
        A matrix of shape [l, k, d_in].
    """

    # Convolve each sequence of length l in v with each sequence in u.
    mvconv = torch.vmap(tr_conv, in_dims=(1, None), out_dims=1)
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
    carry = torch.zeros((k, d_out), device=deltas.device)
    ys = []

    for x in deltas:
        output = torch.tensordot(m_y, carry, dims=2) + x
        carry = torch.roll(carry, 1, dims=0)
        # Avoid in-place operation by reconstructing the carry tensor
        carry = torch.cat((output.unsqueeze(0), carry[:-1]), dim=0)
        ys.append(output.unsqueeze(0))

    ys = torch.cat(ys, dim=0)

    return ys

@torch.jit.script
def compute_ar_x_preds(
    w: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Compute the auto-regressive component of spectral SSM.

    Args:
      w: An array of shape [d_out, d_in, k].
      x: A single input sequence of shape [seq_len, d_in].

    Returns:
      ar_x_preds: An output of shape [seq_len, d_out].
    """
    d_out, _, k = w.shape
    seq_len = x.shape[0]

    # Contract over `d_in`.
    o = torch.einsum('oik,li->klo', w, x)

    # For each `i` in `k`, roll the `(seq_len, d_out)` by `i` steps.
    # TODO: See if this can be implemented using torch.vmap
    o = torch.stack([torch.roll(o[i], shifts=i, dims=0) for i in range(k)])

    # Create a mask that zeros out nothing at `k=0`, the first `(seq_len, d_out)` at
    # `k=1`, the first two `(seq_len, dout)`s at `k=2`, etc.
    m = torch.triu(torch.ones((k, seq_len, d_out), device=w.device))
    
    # Sum along `k`.
    return torch.sum(o * m, dim=0)

def compute_x_tilde(
    inputs: torch.Tensor, eigh: tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'mps'
        if torch.backends.mps.is_available()
        else 'cpu'
    )
    eig_vals, eig_vecs = eigh
    eig_vals = eig_vals.to(device)
    eig_vecs = eig_vecs.to(device)
    inputs = inputs.to(device)
    seq_len = inputs.shape[0]
    
    x_tilde = conv(eig_vecs, inputs)

    x_tilde *= torch.unsqueeze(torch.unsqueeze(eig_vals**0.25, 0), -1)

    # This shift is introduced as the rest is handled by the AR part.
    return shift(shift(x_tilde.reshape((seq_len, -1))))
