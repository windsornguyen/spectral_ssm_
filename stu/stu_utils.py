# =============================================================================#
# Authors: Windsor Nguyen, Dwaipayan Saha, Evan Dogariu
# File: stu_utils.py
# =============================================================================#

"""Utilities for spectral SSM."""

import torch


@torch.jit.script
def get_hankel_matrix(n: int) -> torch.Tensor:
    """Generate a spectral Hankel matrix.

    Args:
        n (int): Number of rows in square spectral Hankel matrix.

    Returns:
        torch.Tensor: A spectral Hankel matrix of shape [n, n].
    """
    indices = torch.arange(1, n + 1)  # -> [n]
    sum_indices = indices[:, None] + indices[None, :]  # -> [n, n]
    z = 2 / (sum_indices**3 - sum_indices)  # -> [n, n]
    return z


@torch.jit.script
def get_top_hankel_eigh(
    n: int, k: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get top k eigenvalues and eigenvectors of spectral Hankel matrix.

    Args:
        n (int): Number of rows in square spectral Hankel matrix.
        k (int): Number of eigenvalues to return.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple of eigenvalues of shape [k,] and
            eigenvectors of shape [n, k].
    """
    hankel_matrix = get_hankel_matrix(n).to(device)  # -> [n, n]
    eig_vals, eig_vecs = torch.linalg.eigh(hankel_matrix)  # -> [n], [n, n]
    return eig_vals[-k:], eig_vecs[:, -k:]  # -> ([k], [n, k])


@torch.jit.script
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


@torch.jit.script
def shift(x: torch.Tensor) -> torch.Tensor:
    """Shift time axis by one to align x_{t-1} and x_t.

    Args:
        x (torch.Tensor): A tensor of shape [seq_len, d].

    Returns:
        torch.Tensor: A tensor of shape [seq_len, d] where index [0, :] is all zeros and
            [i, :] is equal to x[i - 1, :] for i > 0.
    """
    return torch.cat([torch.zeros_like(x[:1]), x[:-1]], dim=0)


def nearest_power_of_2(x: int):
    s = bin(x)
    s = s.lstrip('-0b')
    length = len(s)
    return 1 << (length - 1) if x == 1 << (length - 1) else 1 << length


# TODO: Try to refactor to use Tri Dao's causal_conv1d lib
@torch.jit.script
def conv(v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    Compute convolution to project input sequences into the spectral basis using broadcasting.

    Args:
        v (torch.Tensor): Top k eigenvectors of shape [sl, k].
        u (torch.Tensor): Input of shape [bsz, sl, d_in].

    Returns:
        torch.Tensor: A matrix of shape [bsz, sl, k, d_in].
    """
    bsz, sl, d_in = u.shape
    _, k = v.shape
    n = nearest_power_of_2(sl * 2 - 1) # Round n to the nearest power of 2

    v = v.unsqueeze(0).unsqueeze(-1).expand(bsz, -1, -1, d_in)
    u = u.unsqueeze(2).expand(-1, -1, k, -1)

    V = torch.fft.rfft(v, n=n, dim=1)
    U = torch.fft.rfft(u, n=n, dim=1)
    Z = V * U
    z = torch.fft.irfft(Z, n=n, dim=1)

    return z[:, :sl]


@torch.jit.script
def compute_y_t(m_y: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """
    Computes a sequence of y_t given a series of deltas and a transition matrix m_y.

    Args:
        m_y (torch.Tensor): A matrix of shape [d_out, k, d_out] that acts as windowed
            transition matrix for the linear dynamical system evolving y_t.
        deltas (torch.Tensor): A matrix of shape [bsz, sl, d_out].

    Returns:
        torch.Tensor: A matrix of shape [bsz, sl, d_out].
    """
    d_out, k, _ = m_y.shape
    bsz, sl, _ = deltas.shape

    # Define the transition matrix A, and add bsz for bmm
    A = m_y.view(d_out, -1)  # Reshape m_y to [d_out, k * d_out] for concat
    eye = torch.eye((k - 1) * d_out, k * d_out, dtype=deltas.dtype, device=deltas.device)
    A = torch.cat([A, eye], dim=0)
    A = A.unsqueeze(0).expand(bsz, -1, -1) # -> [bsz, k * d_out, k * d_out]

    # Add (k - 1) rows of padding to deltas
    padding = torch.zeros(
        bsz, sl, (k - 1) * d_out, dtype=deltas.dtype, device=deltas.device
    ) # -> [bsz, sl, (k - 1) * d_out]
    carry = torch.cat([deltas, padding], dim=2)  # -> [bsz, sl, k * d_out]

    # Reshape for sequential processing
    carry = carry.view(bsz, sl, k * d_out, 1) # -> [bsz, sl, k * d_out, 1]

    # Initialize y and the output list of y's
    y = carry[:, 0]  # -> [bsz, k * d_out, 1]
    ys = [y[:, :d_out, 0]] # -> [bsz, d_out]

    # Iterate through the sequence
    for i in range(1, sl):
        y = torch.bmm(A, y) + carry[:, i]
        ys.append(y[:, :d_out, 0])
    ys = torch.stack(ys, dim=1) # -> [bsz, sl, d_out]
    
    return ys


@torch.jit.script
def compute_ar_x_preds(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute the auto-regressive component of spectral SSM.

    Args:
        w (torch.Tensor): An array of shape [d_out, d_in, k].
        x (torch.Tensor): A single input sequence of shape [bsz, sl, d_in].

    Returns:
        torch.Tensor: ar_x_preds: An output of shape [bsz, sl, d_out].
    """
    bsz, sl, d_in = x.shape
    d_out, _, k = w.shape

    # Contract over `d_in`
    o = torch.einsum('oik,bli->bklo', w, x)  # [bsz, k, sl, d_out]

    # For each `i` in `k`, roll the `(sl, d_out)` by `i` steps for each batch
    rolled_o = torch.stack(
        [torch.roll(o[:, i], shifts=i, dims=1) for i in range(k)], dim=1
    )

    # Create a mask that zeros out nothing at `k=0`, the first `(sl, d_out)` at
    # `k=1`, the first two `(l, dout)`s at `k=2`, etc.
    m = torch.triu(torch.ones((k, sl), device=w.device)).unsqueeze(-1)

    # Mask and sum along `k`
    return torch.sum(rolled_o * m, dim=1)


@torch.jit.script
def compute_x_tilde(
    inputs: torch.Tensor, eigh: tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """
    Compute the x_tilde component of spectral SSM.

    Args:
        inputs (torch.Tensor): A tensor of shape [bsz, sl, d_in].
        eigh (tuple[torch.Tensor, torch.Tensor]): A tuple of eigenvalues of shape [k,] and
            eigenvectors of shape [sl, k].

    Returns:
        torch.Tensor: x_tilde: A tensor of shape [bsz, sl, k * d_in].
    """
    eig_vals, eig_vecs = eigh
    bsz, sl, _ = inputs.shape

    x_tilde = conv(eig_vecs, inputs)
    x_tilde *= eig_vals.unsqueeze(0).unsqueeze(2) ** 0.25

    # TODO: May have to adjust this once we introduce autoregressive component.
    return x_tilde.reshape((bsz, sl, -1))
