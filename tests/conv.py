import numpy as np
import torch
import jax
import jax.numpy as jnp
import jax.scipy.signal
import time
import torch.nn.functional as F


def tr_conv(v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    Perform truncated convolution using FFT.
    
    Args:
        v (torch.Tensor): Tensor of shape [seq_len,].
        u (torch.Tensor): Tensor of shape [seq_len,].
    
    Returns:
        torch.Tensor: Convolution result of shape [seq_len,].
    """
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

    print("v_padded shape:", v_padded.shape)
    print("u_padded shape:", u_padded.shape)

    # Perform FFT on both padded inputs
    v_fft = torch.fft.rfft(v_padded)
    u_fft = torch.fft.rfft(u_padded)
    
    print("v_fft shape:", v_fft.shape)
    print("u_fft shape:", u_fft.shape)
    
    # Element-wise multiplication in the frequency domain
    output_fft = v_fft * u_fft

    # Inverse FFT to return to the time domain
    output = torch.fft.irfft(output_fft, n=padded_len)
    
    # Truncate to the original sequence length
    return output[:seq_len]

def conv_torch(v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
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
    return mmconv(v, u)

# JAX implementation using jax.jit
@jax.jit
def conv_jax(v: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    tr_conv = lambda x, y: jax.scipy.signal.convolve(x, y, method='fft')[:x.shape[0]]
    mvconv = jax.vmap(tr_conv, in_axes=(1, None), out_axes=1)
    mmconv = jax.vmap(mvconv, in_axes=(None, 1), out_axes=-1)
    return mmconv(v, u)

# Use CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare data
seq_len, k, d_in = torch.randint(1, 1024, (1,)).item(), torch.randint(1, 24, (1,)).item(), torch.randint(1, 1024, (1,)).item()
v = np.random.rand(seq_len, k)
u = np.random.rand(seq_len, d_in)
v_torch = torch.tensor(v, device=device)
u_torch = torch.tensor(u, device=device)
v_jax = jnp.array(v)
u_jax = jnp.array(u)

print(f'Testing convolution function for v of shape [{seq_len}, {k}] and u of shape [{seq_len}, {d_in}].\nDevice: {device}')

# Warm-up JIT compilation
_ = conv_torch(v_torch, u_torch)
_ = conv_jax(v_jax, u_jax).block_until_ready()

# Benchmark and test outputs
start_time_torch = time.time()
output_torch = conv_torch(v_torch, u_torch).cpu().numpy()
time_torch = time.time() - start_time_torch

start_time_jax = time.time()
output_jax = conv_jax(v_jax, u_jax).block_until_ready()
time_jax = time.time() - start_time_jax

# Check outputs and shapes
if output_torch.shape != output_jax.shape:
    print(f'Shape mismatch between Torch and JAX outputs: Torch shape {output_torch.shape}, JAX shape {output_jax.shape}')

# Check for significant differences
if not np.allclose(output_torch, output_jax, atol=1e-4): # Note: not accurate past 1e-4
    print('Values differ more than acceptable tolerance.')
    difference_matrix = np.abs(output_torch - output_jax)
    
    # Iterate over the matrix and print differences
    it = np.nditer(difference_matrix, flags=['multi_index'])
    print('Differing Values:')
    while not it.finished:
        if it[0] > 1e-4:
            idx = it.multi_index
            diff_value = it[0]
            print(f'Index: {idx}, Torch: {output_torch[idx]}, JAX: {output_jax[idx]}, Diff: {diff_value}')
        it.iternext()

# Output performance metrics
print(f'\nExecution Time (PyTorch): {time_torch:.6f}s')
print(f'Execution Time (JAX): {time_jax:.6f}s')
