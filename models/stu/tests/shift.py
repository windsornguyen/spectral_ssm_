import torch
import jax
import jax.numpy as jnp
import numpy as np
import time
import torch.autograd.profiler as profiler


def shift_torch(x: torch.Tensor) -> torch.Tensor:
    """Shift time axis by one to align x_{t-1} and x_t.

    Args:
        x (torch.Tensor): A tensor of shape [seq_len, d].

    Returns:
        torch.Tensor: A tensor of shape [seq_len, d] where index [0, :] is all zeros and
            [i, :] is equal to x[i - 1, :] for i > 0.
    """
    return torch.cat([torch.zeros_like(x[:1]), x[:-1]], dim=0)


@jax.jit
def shift_jax(x: jnp.ndarray) -> jnp.ndarray:
    """Shift time axis by one to align x_{t-1} and x_t.

    Args:
        x: An array of shape [l, d].

    Returns:
        An array of shape [l, d] where index [0, :] is all zeros and [i, :] is equal
        to x[i - 1, :] for i > 0.
    """
    return jnp.pad(x, ((1, 0), (0, 0)), mode='constant')[:-1, :]


# Set a seed for reproducibility
np.random.seed(1337)

# Prepare random data for testing
seq_len = np.random.randint(100, 1000)
d = np.random.randint(10, 100)
print(f'Testing shift function for a tensor of shape [{seq_len}, {d}].')

# Create random input data
x_np = np.random.rand(seq_len, d).astype(np.float32)
x_torch = torch.from_numpy(x_np)
x_jax = jnp.array(x_np)

# Get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_torch = x_torch.to(device)

# Warm up the JIT compilers
_ = shift_torch(x_torch)
_ = shift_jax(x_jax)

# Profile PyTorch
with profiler.profile(with_stack=True, profile_memory=True) as prof:
    _ = shift_torch(x_torch)

# Print the profiling results for PyTorch
print(prof.key_averages().table(sort_by='cpu_time_total'))

# Benchmark PyTorch
start_time_torch = time.time()
result_torch = shift_torch(x_torch)
time_torch = time.time() - start_time_torch

# Benchmark JAX
start_time_jax = time.time()
result_jax = shift_jax(x_jax)
time_jax = time.time() - start_time_jax

# Move PyTorch result back to CPU for comparison
result_torch = result_torch.cpu()

# Output performance metrics
print(f'\nExecution Time (PyTorch): {time_torch:.6f}s')
print(f'Execution Time (JAX): {time_jax:.6f}s')

# Compare the results
print('\nComparing the results...')

if np.allclose(result_torch.numpy(), result_jax, atol=1e-7):
    print('The results from PyTorch and JAX are close enough!')
else:
    print(
        'The results from PyTorch and JAX differ more than the acceptable tolerance...'
    )

    # Find the indices where the results differ
    diff_indices = np.where(np.abs(result_torch.numpy() - result_jax) > 1e-7)

    # Print the differing indices and values
    print('Differing indices and values:')
    for i in range(len(diff_indices[0])):
        index = tuple(diff_index[i] for diff_index in diff_indices)
        print(f'Index: {index}')
        print(f'PyTorch value: {result_torch.numpy()[index]}')
        print(f'JAX value: {result_jax[index]}')
        print()
        if i > 5:
            break
