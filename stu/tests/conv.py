import numpy as np
import torch
import jax
import jax.numpy as jnp
import jax.scipy.signal
import time
import torch.nn.functional as F


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


def conv_torch(v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    Compute convolution to project input sequences into the spectral basis using broadcasting.
    
    Args:
        v (torch.Tensor): Top k eigenvectors of shape [l, k].
        u (torch.Tensor): Input of shape [bsz, l, d_in].
    
    Returns:
        torch.Tensor: A matrix of shape [bsz, l, k, d_in].
    """
    # mvconv = torch.vmap(tr_conv, in_dims=(1, None), out_dims=1)
    # mmconv = torch.vmap(mvconv, in_dims=(None, 1), out_dims=-1)
    # bmmconv = torch.vmap(mmconv, in_dims=(None, 0), out_dims=0)
    # return bmmconv(v, u)
    
    bsz, l, d_in = u.shape
    k = v.shape[1]

    # Reshape and expand dimensions for broadcasting
    v = v.view(1, l, k, 1).expand(bsz, -1, -1, d_in)
    u = u.view(bsz, l, 1, d_in).expand(-1, -1, k, -1)

    # Perform convolution using tr_conv, TODO: vectorize this without vmap
    result = torch.zeros(bsz, l, k, d_in, device=v.device)
    for b in range(bsz):
        for i in range(k):
            for j in range(d_in):
                result[b, :, i, j] = tr_conv(v[b, :, i, j], u[b, :, i, j])

    return result


# JAX implementation using jax.jit and batch sizes
@jax.jit
def conv_jax(
    v: jnp.ndarray,
    u: jnp.ndarray,
) -> jnp.ndarray:
    """Compute convolution to project input sequences into the spectral basis.

    Args:
        v: Top k eigenvectors of shape [l, k].
        u: Input of shape [bsz, l, d_in].

    Returns:
        A matrix of shape [bsz, l, k, d_in]
    """
    # Convolve two vectors of length l (x.shape[0]) and truncate to the l oldest
    # values.
    tr_conv = lambda x, y: jax.scipy.signal.convolve(x, y, method='fft')[:x.shape[0]]

    # Convolve each sequence of length l in v with each sequence in u.
    mvconv = jax.vmap(tr_conv, in_axes=(1, None), out_axes=1)
    mmconv = jax.vmap(mvconv, in_axes=(None, 1), out_axes=-1)

    # Add an additional vmap to handle the batch dimension
    bmmconv = jax.vmap(mmconv, in_axes=(None, 0), out_axes=0)

    return bmmconv(v, u)


# Set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Use CUDA if available
device = torch.device(
    'cuda'
    if torch.cuda.is_available()
    else 'mps'
    if torch.backends.mps.is_available()
    else 'cpu'
)

# Prepare data
batch_size, seq_len, k, d_in = (
    torch.randint(1, 8, (1,)).item(),
    torch.randint(1, 32, (1,)).item(),
    torch.randint(1, 8, (1,)).item(),
    torch.randint(1, 32, (1,)).item(),
)

v = np.random.rand(seq_len, k)
u = np.random.rand(batch_size, seq_len, d_in)

v_torch = torch.tensor(v, device=device, dtype=torch.float32)
u_torch = torch.tensor(u, device=device, dtype=torch.float32)
v_jax = jnp.array(v)
u_jax = jnp.array(u)


# Warm-up JIT compilation
_ = conv_torch(v_torch, u_torch)
_ = conv_jax(v_jax, u_jax).block_until_ready()

# Benchmark and test outputs
start_time_torch = time.time()
output_torch = conv_torch(v_torch, u_torch).detach().cpu().numpy()
time_torch = time.time() - start_time_torch

start_time_jax = time.time()
output_jax = conv_jax(v_jax, u_jax).block_until_ready()
time_jax = time.time() - start_time_jax

# Check outputs and shapes
if output_torch.shape != output_jax.shape:
    print(
        f'Shape mismatch between Torch and JAX outputs: Torch shape {output_torch.shape}, JAX shape {output_jax.shape}'
    )

# Check for significant differences
i = 0
if not np.allclose(
    output_torch, output_jax, atol=1e-6
):  # Note: not accurate past 1e-4
    print('Values differ more than acceptable tolerance.')
    difference_matrix = np.abs(output_torch - output_jax)

    # Iterate over the matrix and print differences
    it = np.nditer(difference_matrix, flags=['multi_index'])
    print('Differing Values:')
    while not it.finished:
        if it[0] > 1e-6:
            idx = it.multi_index
            diff_value = it[0]
            print(
                f'Index: {idx}, Torch: {output_torch[idx]}, JAX: {output_jax[idx]}, Diff: {diff_value}'
            )
        it.iternext()
        i += 1
        if i > 10:
            break
else:
    print('Outputs are sufficiently close.')

# Output performance metrics
print(f'\nExecution Time (PyTorch): {time_torch:.6f}s')
print(f'Execution Time (JAX): {time_jax:.6f}s')
