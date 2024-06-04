import torch
import jax
import jax.numpy as jnp
import numpy as np
import time
import functools
import torch.autograd.profiler as profiler

@torch.jit.script
def compute_ar_x_preds_torch(
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

    # Contract over `d_in` and roll using advanced indexing
    o = torch.einsum('oik,li->klo', w, x)

    # Generate indices for the rolling effect
    arange_seq = torch.arange(seq_len, device=w.device).unsqueeze(0)
    arange_k = torch.arange(k, device=w.device).unsqueeze(1)
    shift_indices = (arange_seq + arange_k) % seq_len

    # Expand indices for batch
    expanded_indices = shift_indices.unsqueeze(-1).expand(k, seq_len, d_out)
    rolled_o = torch.gather(o, 1, expanded_indices)

    # Apply the mask and sum along `k`
    mask = torch.triu(torch.ones((k, seq_len), device=w.device)).unsqueeze(-1)
    return (rolled_o * mask).sum(dim=0)



@jax.jit
def compute_ar_x_preds_jax(
    w: jnp.ndarray,
    x: jnp.ndarray,
) -> jnp.ndarray:
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
    o = jnp.einsum('oik,li->klo', w, x)

    # For each `i` in `k`, roll the `(l, d_out)` by `i` steps.
    o = jax.vmap(functools.partial(jnp.roll, axis=0))(o, jnp.arange(k))

    # Create a mask that zeros out nothing at `k=0`, the first `(l, d_out)` at
    # `k=1`, the first two `(l, dout)`s at `k=2`, etc.
    m = jnp.triu(jnp.ones((k, l)))
    m = jnp.expand_dims(m, axis=-1)
    m = jnp.tile(m, (1, 1, d_out))

    # Mask and sum along `k`.
    return jnp.sum(o * m, axis=0)

# Set a seed
np.random.seed(42)

# Prepare random data for testing
d_out = np.random.randint(10, 100)
k = np.random.randint(1, 10)
seq_len = np.random.randint(100, 1000)
print(f'Testing compute_ar_x_preds function for w of shape [{d_out}, {d_out}, {k}] and x of shape [{seq_len}, {d_out}].')

# Create random input data
w_np = np.random.rand(d_out, d_out, k).astype(np.float32)
x_np = np.random.rand(seq_len, d_out).astype(np.float32)
w_torch = torch.from_numpy(w_np)
x_torch = torch.from_numpy(x_np)
w_jax = jnp.array(w_np)
x_jax = jnp.array(x_np)

# Get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
w_torch = w_torch.to(device)
x_torch = x_torch.to(device)

# Warm up the JIT compilers
_ = compute_ar_x_preds_torch(w_torch, x_torch)
_ = compute_ar_x_preds_jax(w_jax, x_jax)

# Profile PyTorch
with profiler.profile(with_stack=True, profile_memory=True) as prof:
    _ = compute_ar_x_preds_torch(w_torch, x_torch)

# Print the profiling results for PyTorch
print(prof.key_averages().table(sort_by="cpu_time_total"))

# Benchmark PyTorch
start_time_torch = time.time()
result_torch = compute_ar_x_preds_torch(w_torch, x_torch)
time_torch = time.time() - start_time_torch

# Benchmark JAX
start_time_jax = time.time()
result_jax = compute_ar_x_preds_jax(w_jax, x_jax)
time_jax = time.time() - start_time_jax

# Move PyTorch result back to CPU for comparison
result_torch = result_torch.cpu()

# Output performance metrics
print(f'\nExecution Time (PyTorch): {time_torch:.6f}s')
print(f'Execution Time (JAX): {time_jax:.6f}s')

# Compare the results
print('\nComparing the results...')

if np.allclose(result_torch.numpy(), result_jax, atol=1e-8):
    print('The results from PyTorch and JAX are close enough.')
else:
    print('The results from PyTorch and JAX differ more than the acceptable tolerance.')
    
    # Find the indices where the results differ
    diff_indices = np.where(np.abs(result_torch.numpy() - result_jax) > 1e-8)
    
    # Print the differing indices and values
    print('Differing indices and values:')
    for i in range(len(diff_indices[0])):
        index = tuple(diff_index[i] for diff_index in diff_indices)
        print(f'Index: {index}')
        print(f'PyTorch value: {result_torch.numpy()[index]}')
        print(f'JAX value: {result_jax[index]}')
        print()
        if i >= 5:
            break
