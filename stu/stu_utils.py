# =============================================================================#
# Authors: Windsor Nguyen, Dwaipayan Saha, Evan Dogariu
# File: stu_utils.py
# =============================================================================#

"""Utilities for spectral SSM."""

import torch
import torch.nn.functional as F


@torch.jit.script
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


@torch.jit.script
def tr_conv(v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    Perform truncated convolution using FFT.
    
    Args:
        v (torch.Tensor): Tensor of shape [seq_len,].
        u (torch.Tensor): Tensor of shape [seq_len,].
    
    Returns:
        torch.Tensor: Convolution result of shape [seq_len,].
    """
    # Set the device
    device = v.device

    # Calculate the sequence length and determine the target tensor
    seq_len = max(v.size(0), u.size(0))
    target_tensor = torch.tensor(2 * seq_len - 1, device=device, dtype=torch.float32)

    # Calculate the ceiling of the log base 2 of the target tensor
    ceil_log_base_2 = torch.ceil(torch.log2(target_tensor))

    # Calculate the padded length as the next power of two
    padded_len = int(2 ** ceil_log_base_2)

    # Padding for FFT efficiency (lengths that are powers of two perform best)
    v_padded = F.pad(v, (0, padded_len - seq_len))
    u_padded = F.pad(u, (0, padded_len - seq_len))

    # Perform FFT on both padded inputs
    v_fft = torch.fft.rfft(v_padded)
    u_fft = torch.fft.rfft(u_padded)

    # Element-wise multiplication in the frequency domain
    output_fft = v_fft * u_fft

    # Inverse FFT to return to the time domain
    output = torch.fft.irfft(output_fft, n=padded_len)
    
    # Truncate to the original sequence length
    return output[:seq_len]


def conv(v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    Compute convolution to project input sequences into the spectral basis using broadcasting.
    
    Args:
        v (torch.Tensor): Top k eigenvectors of shape [seq_len, k].
        u (torch.Tensor): Input of shape [seq_len, d_in].
    
    Returns:
        torch.Tensor: A matrix of shape [seq_len, k, d_in].
    """
    # Convolve each sequence of length `seq_len` in v with each sequence in u.
    mvconv = torch.vmap(tr_conv, in_dims=(1, None), out_dims=1)
    mmconv = torch.vmap(mvconv, in_dims=(None, 1), out_dims=-1)
    out = mmconv(v, u)

    return out


@torch.jit.script
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
    carry = torch.zeros((k, d_out), device=deltas.device)
    ys = []

    for x in deltas:
        output = torch.tensordot(m_y, carry, dims=2)
        output = output + x
        carry = torch.roll(carry, 1, dims=0)

        # Avoid in-place operation by reconstructing the carry tensor
        # TODO: Once torch.vmap is removed, we can modify in-place
        carry = torch.cat((output.unsqueeze(0), carry[1:]), dim=0)

        ys.append(output.unsqueeze(0))

    ys = torch.cat(ys, dim=0)

    return ys


@torch.jit.script
def compute_ar_x_preds(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute the auto-regressive component of spectral SSM.

    Args:
        w (torch.Tensor): An array of shape [d_out, d_in, k].
        x (torch.Tensor): A single input sequence of shape [seq_len, d_in].

    Returns:
        torch.Tensor: ar_x_preds: An output of shape [seq_len, d_out].
    """
    d_out, d_in, k = w.shape
    seq_len = x.shape[0]

    # Contract over `d_in`
    o = torch.einsum('oik,li->klo', w, x)

    # For each `i` in `k`, roll the `(seq_len, d_out)` by `i` steps
    o = torch.stack([torch.roll(o[i], shifts=i, dims=0) for i in range(k)])

    # Create a mask that zeros out nothing at `k=0`, the first `(seq_len, d_out)` at
    # `k=1`, the first two `(seq_len, dout)`s at `k=2`, etc.
    m = torch.triu(torch.ones((k, seq_len), device=w.device))
    m = m.unsqueeze(-1)
    m = m.expand((k, seq_len, d_out))

    # Mask and sum along `k`
    return torch.sum(o * m, dim=0)


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
    seq_len = inputs.shape[0]

    x_tilde = conv(eig_vecs, inputs)
    x_tilde *= eig_vals.unsqueeze(0).unsqueeze(2) ** 0.25

    # NOTE: Shifting twice is incorrect, noted by Evan.
    # return shift_torch(shift_torch(x_tilde.reshape((seq_len, -1))))
    return x_tilde.reshape((seq_len, -1))
