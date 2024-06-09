import torch
import jax
import jax.numpy as jnp
import numpy as np
import time
import torch.autograd.profiler as profiler

@torch.jit.script
def pscan_fft_simple(A, X):
    N, T, D = X.shape
    device = X.device
    A_log = torch.log(A.to(dtype=torch.cfloat))
    A_log_T = A_log.T
    A_log_T = torch.cat([A_log_T, torch.zeros(T - 1, N, device=device)], dim=0)
    mask1 = torch.where((torch.arange(2 * T - 1, device=device) <= T - 1),1,0,)
    mask1 = mask1.unsqueeze(1)
    Z1_log_rev = torch.fft.ifft(torch.fft.fft(mask1, dim=0) * torch.fft.fft(A_log_T, dim=0), n=2 * T - 1, dim=0)
    Z1_log_rev = Z1_log_rev[:T, :].T.unsqueeze(-1)
    mask2 = torch.where(torch.cat([((torch.arange(2 * T - 1, device=device) >= 1) & (torch.arange(2 * T - 1, device=device) <= t)).unsqueeze(0) for t in range(T)], dim=0,), 1, 0,)
    mask2 = mask2.unsqueeze(-1)
    Z2_log_rev = torch.fft.ifft(torch.fft.fft(mask2, dim=1) * torch.fft.fft(A_log_T.unsqueeze(0), dim=1),n=2 * T - 1,dim=1,)
    Z2_log_rev = Z2_log_rev[:, :T, :]
    Z2_log_rev = Z2_log_rev.permute(2, 0, 1)
    Z2_log_rev = torch.tril(Z2_log_rev, diagonal=0)
    Z_log = Z1_log_rev - Z2_log_rev
    Z = torch.tril(torch.exp(Z_log), diagonal=0)
    Y_ = torch.bmm(Z, X)
    Y_ = torch.cat([torch.zeros(N, 1, D, device=device), Y_[:, :-1, :]], dim=1)
    Y = Y_ + X
    return Y

@torch.jit.script
def pscan_naive(A, X):
    N, T, D = X.shape
    device = X.device
    Y = torch.zeros(N, T, D, device=device, dtype=X.dtype)
    Y[:, 0, :] = X[:, 0, :]
    for k in range(1, X.shape[1]): Y[:, k, :] = A[:, k - 1].unsqueeze(1) * Y[:, k - 1, :] + X[:, k, :]
    return Y

@torch.jit.script
def get_eig(A):
    return torch.linalg.eig(A)

# @torch.jit.script
def compute_y_t_pscan(m_y: torch.Tensor, deltas: torch.Tensor, use_fft: bool) -> torch.Tensor:
    """
    Compute sequence of y_t given a series of deltas and m_y via a simple scan.

    Args:
        m_y (torch.Tensor): A matrix of shape [d_out, k, d_out] that acts as windowed 
            transition matrix for the linear dynamical system evolving y_t.
        deltas (torch.Tensor): A matrix of shape [seq_len, d_out].

    Returns:
        torch.Tensor: A matrix of shape [seq_len, d_out].
    """

    # stack states and transitions to have a bona fide LDS. state is of dimension `k * d_out`
    # transition matrix has (a permuted) m_y in the first row, and identity on the -1 off-diagonal
    _, k, d = m_y.shape
    L, _ = deltas.shape
    identity = torch.cat([torch.eye((k - 1) * d), torch.zeros((k - 1) * d, d)], axis=1)
    reshaped_M = m_y.reshape(d, k * d)
    A = torch.cat([reshaped_M, identity], axis=0)
    X = torch.cat([deltas, torch.zeros(L, (k - 1) * d)], axis=-1)

    # diagonalize dynamics and project
    evals, evecs = get_eig(A)
    inv_evecs = torch.linalg.inv(evecs)
    X_diag = X.to(torch.complex64) @ inv_evecs.T
    # X_diag = (inv_evecs @ X.to(torch.complex64).T).T

    # run diagonal pscan. N = k * d and T = L
    A_pscan = torch.tile(evals[:, None], (1, L))
    X_pscan = X_diag.T[:, :, None]
    Y_pscan = pscan_fft_simple(A_pscan, X_pscan) if use_fft else pscan_naive(A_pscan, X_pscan)
    ys = Y_pscan[:, :, 0].T

    # undiagonalize
    # ys = (evecs @ ys.T).T.to(torch.float32)
    ys = (ys @ evecs.T).to(torch.float32)
    return ys[:, :d]

# @torch.jit.script
def compute_y_t_stack(m_y: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """
    Compute sequence of y_t given a series of deltas and m_y via a simple scan.

    Args:
        m_y (torch.Tensor): A matrix of shape [d_out, k, d_out] that acts as windowed 
            transition matrix for the linear dynamical system evolving y_t.
        deltas (torch.Tensor): A matrix of shape [seq_len, d_out].

    Returns:
        torch.Tensor: A matrix of shape [seq_len, d_out].
    """

    # stack states and transitions to have a bona fide LDS. state is of dimension `k * d_out`
    # transition matrix has (a permuted) m_y in the first row, and identity on the -1 off-diagonal
    _, k, d = m_y.shape
    L, _ = deltas.shape
    identity = torch.cat([torch.eye((k - 1) * d), torch.zeros((k - 1) * d, d)], axis=1)
    reshaped_M = m_y.reshape(d, k * d)
    A = torch.cat([reshaped_M, identity], axis=0)
    X = torch.cat([deltas, torch.zeros(L, (k - 1) * d)], axis=-1)

    y = X[0]
    ys = [y[:d]]
    for i in range(1, L):
        y = A @ y + X[i]
        ys.append(y[:d])
    return torch.stack(ys)


@torch.jit.script
def compute_y_t_torch(m_y: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """
    Compute sequence of y_t given a series of deltas and m_y via a simple scan.

    Args:
        m_y (torch.Tensor): A matrix of shape [d_out, k, d_out] that acts as windowed 
            transition matrix for the linear dynamical system evolving y_t.
        deltas (torch.Tensor): A matrix of shape [seq_len, d_out].

    Returns:
        torch.Tensor: A matrix of shape [seq_len, d_out].
    """
    # d_out, k, _ = m_y.shape
    # seq_len, _ = deltas.shape

    # # Initialize carry and ys tensor
    # carry = torch.zeros((seq_len + k, d_out), device=deltas.device)
    # ys = torch.zeros(seq_len, d_out, device=deltas.device)

    # # Fill the deltas into the appropriate positions in carry
    # carry[k:] = deltas

    # # Apply the transition matrices to the carry tensor
    # for i in range(k):
    #     # Correct the tensordot operation
    #     ys += torch.tensordot(carry[i:i+seq_len], m_y[:, i, :], dims=([1], [0]))

    # ys += deltas

    # return ys
    
    # d_out, k, _ = m_y.shape
    # batch_size = deltas.size(0)

    # # Initialize carry with zeros for the entire batch
    # carry = torch.zeros(batch_size, k, d_out, device=deltas.device)

    # # Calculate all tensor dot products at once (requires `m_y` to be compatible)
    # outputs = torch.einsum('ijk,blk->bi', m_y, carry) + deltas

    # # Define a shift matrix that shifts all outputs down and places the latest output at the top
    # shift_matrix = torch.eye(k, k, device=deltas.device)
    # shift_matrix = torch.roll(shift_matrix, 1, dims=0)

    # # Update carry for the next iteration
    # new_carry = torch.matmul(shift_matrix, carry.reshape(batch_size, k, d_out)).reshape(batch_size, k, d_out)
    # new_carry[:, 0, :] = outputs

    # # Result collection
    # ys = outputs

    # return ys

    d_out, k, _ = m_y.shape
    carry = torch.zeros((k, d_out), device=deltas.device)
    ys = []

    for x in deltas:
        output = torch.tensordot(m_y, carry, dims=2) + x
        carry = torch.roll(carry, 1, dims=0)

        # Avoid in-place operation by reconstructing the carry tensor
        carry = torch.cat((output.unsqueeze(0), carry[1:]), dim=0)

        ys.append(output.unsqueeze(0))

    ys = torch.cat(ys, dim=0)

    return ys


@jax.jit
def compute_y_t_jax(m_y: jnp.ndarray, deltas: jnp.ndarray) -> jnp.ndarray:
    """Compute sequence of y_t given a series of deltas and m_y via a simple scan.

    Args:
        m_y: A matrix of shape [d_out, k, d_out] that acts as windowed transition
            matrix for the linear dynamical system evolving y_t.
        deltas: A matrix of shape [l, d_out].

    Returns:
        A matrix of shape [l, d_out].
    """
    d_out, k, _ = m_y.shape

    def scan_op(carry, x):
        output = jnp.tensordot(m_y, carry, axes=2) + x
        carry = jnp.roll(carry, 1, axis=0)
        carry = carry.at[0].set(output)
        return carry, output

    _, ys = jax.lax.scan(scan_op, jnp.zeros((k, d_out)), deltas)
    return ys

# Set a seed
np.random.seed(42)

# Prepare random data for testing
d_out = np.random.randint(10, 100)
k = np.random.randint(1, 10)
seq_len = np.random.randint(100, 1000)
# d_out = 128
# k = 24
# seq_len = 1000
print(f'Testing compute_y_t function for m_y of shape [{d_out}, {k}, {d_out}] and deltas of shape [{seq_len}, {d_out}].')

# Create random input data
m_y_np = np.random.rand(d_out, k, d_out).astype(np.float32)
deltas_np = np.random.rand(seq_len, d_out).astype(np.float32)
m_y_torch = torch.from_numpy(m_y_np)
deltas_torch = torch.from_numpy(deltas_np)
m_y_jax = jnp.array(m_y_np)
deltas_jax = jnp.array(deltas_np)

# Get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m_y_torch = m_y_torch.to(device)
deltas_torch = deltas_torch.to(device)

# Warm up the JIT compilers
_ = compute_y_t_torch(m_y_torch, deltas_torch)
_ = compute_y_t_jax(m_y_jax, deltas_jax).block_until_ready()

# Profile PyTorch
with profiler.profile(with_stack=True, profile_memory=True) as prof:
    result_torch = compute_y_t_torch(m_y_torch, deltas_torch)
    
# Print the profiling results for PyTorch
print(prof.key_averages().table(sort_by="cpu_time_total"))

# Benchmark PyTorch
start_time_torch = time.time()
result_torch = compute_y_t_torch(m_y_torch, deltas_torch)
time_torch = time.time() - start_time_torch

# Benchmark JAX
start_time_jax = time.time()
result_jax = compute_y_t_jax(m_y_jax, deltas_jax).block_until_ready()
time_jax = time.time() - start_time_jax

# Move PyTorch result back to CPU for comparison
result_torch = result_torch.cpu()

# Output performance metrics
print(f'\nExecution Time (PyTorch): {time_torch:.6f}s')
print(f'Execution Time (JAX): {time_jax:.6f}s')

# Compare the results
print('\nComparing the results...')

atol = 1e-3
if np.allclose(result_torch.numpy(), result_jax, atol=atol):
    print('The results from PyTorch and JAX are close enough.')
else:
    print('The results from PyTorch and JAX differ more than the acceptable tolerance.')
    
    # Find the indices where the results differ
    diff_indices = np.where(np.abs(result_torch.numpy() - result_jax) > atol)
    
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
