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
    print('get hankel')
    indices = torch.arange(1, n + 1)
    sum_indices = indices[:, None] + indices[None, :]
    z = 2 / (sum_indices ** 3 - sum_indices)
    print('got hankel')
    return z


@torch.jit.script
def get_top_hankel_eigh(n: int, k: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Get top k eigenvalues and eigenvectors of spectral Hankel matrix.

    Args:
        n (int): Number of rows in square spectral Hankel matrix.
        k (int): Number of eigenvalues to return.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple of eigenvalues of shape [k,] and 
            eigenvectors of shape [n, k].
    """
    print('top_eigh')
    hankel_matrix = get_hankel_matrix(n).to(device)
    eig_vals, eig_vecs = torch.linalg.eigh(hankel_matrix)
    print('top done')
    return eig_vals[-k:], eig_vecs[:, -k:]


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
    random_clamped = torch.clamp(random, min=lower, max=upper)
    return scaling * random_clamped


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
    # Sequence length
    seq_len = v.size(0)

    # Calculate the next power of two for padding length
    target_tensor = torch.tensor(2 * seq_len - 1)
    ceil_log_base_2 = torch.ceil(torch.log2(target_tensor))
    padded_len = int(2 ** ceil_log_base_2)

    # Padding for FFT efficiency (lengths that are powers of two perform best)
    v_padded = F.pad(v, (0, padded_len - seq_len))
    u_padded = F.pad(u, (0, padded_len - seq_len))

    # FFT and element-wise multiplication for convolution
    v_fft = torch.fft.rfft(v_padded)
    u_fft = torch.fft.rfft(u_padded)
    
    # Element-wise multiplication in the frequency domain
    output_fft = v_fft * u_fft

    # Inverse FFT to return to time domain
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
    seq_len, k = v.shape
    d_in = u.shape[0]
    
    # Reshape u to [seq_len, d_in]
    u = u.unsqueeze(0).expand(seq_len, d_in)
    
    # Convolve each sequence of length `seq_len` in v with each sequence in u.
    mvconv = torch.vmap(tr_conv, in_dims=(1, None), out_dims=1)
    mmconv = torch.vmap(mvconv, in_dims=(None, 1), out_dims=-1)
    out = mmconv(v, u)

    return out



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
    carry = torch.zeros((k, d_out), device=m_y.device)
    ys = []

    for x in deltas:
        output = torch.tensordot(m_y, carry, dims=2) + x
        carry = torch.cat((output.unsqueeze(0), carry[:-1]), dim=0)
        ys.append(output)
    out = torch.stack(ys)
    return out


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

    # Add a new dimension to x to match the equation
    x = x.unsqueeze(1)

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
    # print("compute_x_tilde - eig_vals:", eig_vals)
    # print("compute_x_tilde - eig_vecs:", eig_vecs)
    # print("compute_x_tilde - inputs:", inputs)
    seq_len = inputs.shape[0]
    x_tilde = conv(eig_vecs, inputs)
    x_tilde *= torch.unsqueeze(torch.unsqueeze(eig_vals**0.25, 0), -1)
    # print("compute_x_tilde - x_tilde after operations:", x_tilde)

    # This shift is introduced as the rest is handled by the AR part.
    return shift(shift(x_tilde.reshape((seq_len, -1))))
