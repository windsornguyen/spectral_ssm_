# =============================================================================#
# Authors: Windsor Nguyen, Dwaipayan Saha, Evan Dogariu
# File: stu_utils.py
# =============================================================================#

#TODO: Put @torch.compile on these (need to remove vmap first)
"""Utilities for spectral SSM."""

import torch
import torch.nn.functional as F


def get_hankel_matrix(n: int) -> torch.Tensor:
    """Generate a spectral Hankel matrix.

    Args:
        n (int): Number of rows in square spectral Hankel matrix.

    Returns:
        torch.Tensor: A spectral Hankel matrix of shape [n, n].
    """
    indices = torch.arange(1, n + 1) # -> [n]
    sum_indices = indices[:, None] + indices[None, :] # -> [n, n]
    z = 2 / (sum_indices ** 3 - sum_indices) # -> [n, n]
    return z


def get_top_hankel_eigh(
    n: int, 
    k: int, 
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get top k eigenvalues and eigenvectors of spectral Hankel matrix.

    Args:
        n (int): Number of rows in square spectral Hankel matrix.
        k (int): Number of eigenvalues to return.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple of eigenvalues of shape [k,] and 
            eigenvectors of shape [n, k].
    """
    hankel_matrix = get_hankel_matrix(n).to(device) # -> [n, n]
    eig_vals, eig_vecs = torch.linalg.eigh(hankel_matrix) # -> [n], [n, n]
    return eig_vals[-k:], eig_vecs[:, -k:] # -> ([k], [n, k])


def get_random_real_matrix(
    shape: list[int],
    scaling: float,
    lower: float = -2.0,
    upper: float = 2.0,
) -> torch.Tensor:
    """Generate a random real matrix.

    Args:
        shape (list[int]): Shape of the matrix to generate.
        scaling (float): Scaling factor for the matrix values.
        lower (float, optional): Lower bound of truncated normal distribution 
            before scaling.
        upper (float, optional): Upper bound of truncated normal distribution
            before scaling.

    Returns:
        torch.Tensor: A random real matrix scaled by the specified factor.
    """
    random = torch.randn(shape)
    clamped = torch.clamp(random, min=lower, max=upper)
    return scaling * clamped


def shift(x: torch.Tensor) -> torch.Tensor:
    """Shift time axis by one to align x_{t-1} and x_t.

    Args:
        x (torch.Tensor): A tensor of shape [seq_len, d].

    Returns:
        torch.Tensor: A tensor of shape [seq_len, d] where index [0, :] is all zeros and 
            [i, :] is equal to x[i - 1, :] for i > 0.
    """
    return torch.cat([torch.zeros_like(x[:1]), x[:-1]], dim=0)


def tr_conv(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Perform truncated convolution using FFT.
    
    Args:
        v (torch.Tensor): Tensor of shape [seq_len,].
        u (torch.Tensor): Tensor of shape [seq_len,].
    
    Returns:
        torch.Tensor: Convolution result of shape [seq_len,].
    """
    n = x.shape[0] + y.shape[0] - 1
    X = torch.fft.rfft(x, n=n)
    Y = torch.fft.rfft(y, n=n)
    Z = X * Y
    z = torch.fft.irfft(Z, n=n)
    return z[:x.shape[0]]


def conv(v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    Compute convolution to project input sequences into the spectral basis using broadcasting.
    
    Args:
        v (torch.Tensor): Top k eigenvectors of shape [l, k].
        u (torch.Tensor): Input of shape [bsz, l, d_in].
    
    Returns:
        torch.Tensor: A matrix of shape [bsz, l, k, d_in].
    """
    # TODO: Slow but works and avoids vmap bugs. Vectorize this eventually.
    bsz, l, d_in = u.shape
    k = v.shape[1]

    # Reshape and expand dimensions for broadcasting
    v = v.view(1, l, k, 1).expand(bsz, -1, -1, d_in)
    u = u.view(bsz, l, 1, d_in).expand(-1, -1, k, -1)

    # Perform convolution using tr_conv
    result = torch.zeros(bsz, l, k, d_in, device=v.device)
    for b in range(bsz):
        for i in range(k):
            for j in range(d_in):
                result[b, :, i, j] = tr_conv(v[b, :, i, j], u[b, :, i, j])

    return result


def compute_y_t(m_y: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Compute sequence of y_t given a series of deltas and m_y via a simple scan.

    Args:
        m_y (torch.Tensor): A matrix of shape [d_out, k, d_out] that acts as windowed 
            transition matrix for the linear dynamical system evolving y_t.
        deltas (torch.Tensor): A matrix of shape [seq_len, d_out].

    Returns:
        torch.Tensor: A matrix of shape [seq_len, d_out].
    """
    d_out, k, _ = m_y.shape
    bsz, seq_len, _ = deltas.shape

    device = m_y.device

    A = torch.cat([
        m_y.reshape(d_out, k * d_out).to(device),
        torch.eye((k - 1) * d_out, k * d_out, device=device, dtype=torch.float32)
    ], dim=0)

    X = torch.cat([
        deltas,
        torch.zeros(bsz, seq_len, (k - 1) * d_out, device=device, dtype=torch.float32)
    ], dim=2)

    y = X[:, 0]
    ys = [y[..., :d_out]]

    for x in X[:, 1:].transpose(0, 1):
        y = A @ y.reshape(bsz, k * d_out, 1) + x.reshape(bsz, k * d_out, 1)
        ys.append(y[:, :d_out, 0])

    return torch.stack(ys, dim=1)


def compute_ar_x_preds(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute the auto-regressive component of spectral SSM.

    Args:
        w (torch.Tensor): An array of shape [d_out, d_in, k].
        x (torch.Tensor): A single input sequence of shape [bsz, l, d_in].

    Returns:
        torch.Tensor: ar_x_preds: An output of shape [bsz, l, d_out].
    """
    bsz, l, d_in = x.shape
    d_out, _, k = w.shape

    # Contract over `d_in`
    o = torch.einsum('oik,bli->bklo', w, x)  # [bsz, k, l, d_out]

    # For each `i` in `k`, roll the `(l, d_out)` by `i` steps for each batch
    rolled_o = torch.stack([torch.roll(o[:, i], shifts=i, dims=1) for i in range(k)], dim=1)

    # Create a mask that zeros out nothing at `k=0`, the first `(l, d_out)` at
    # `k=1`, the first two `(l, dout)`s at `k=2`, etc.
    m = torch.triu(torch.ones((k, l), device=w.device)).unsqueeze(-1)

    # Mask and sum along `k`
    return torch.sum(rolled_o * m, dim=1)


def compute_x_tilde(
    inputs: torch.Tensor, eigh: tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """Compute the x_tilde component of spectral SSM.

    Args:
        inputs (torch.Tensor): A tensor of shape [seq_len, d_in].
        eigh (tuple[torch.Tensor, torch.Tensor]): A tuple of eigenvalues of shape [k,] and 
            eigenvectors of shape [seq_len, k].

    Returns:
        torch.Tensor: x_tilde: A tensor of shape [seq_len, k * d_in].
    """
    eig_vals, eig_vecs = eigh
    bsz, l, _ = inputs.shape

    x_tilde = conv(eig_vecs, inputs)
    x_tilde *= eig_vals.unsqueeze(0).unsqueeze(2) ** 0.25

    # NOTE: Shifting twice is incorrect, noted by Evan.
    # return shift_torch(shift_torch(x_tilde.reshape((seq_len, -1))))
    return x_tilde.reshape((bsz, l, -1))
