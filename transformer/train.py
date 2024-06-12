import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from stu.physics.physics_data import get_dataloader
from transformer.model import Transformer, TransformerConfig
from losses.loss_ant import AntLoss
from losses.loss_cheetah import HalfCheetahLoss
from losses.loss_walker import Walker2DLoss

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def gaussian_kernel(size, sigma):
    """Creates a 1D Gaussian kernel using PyTorch."""
    size = int(size) // 2
    x = torch.arange(-size, size + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def smooth_curve(points, sigma=2):
    """Applies 1D Gaussian smoothing on a list of points."""
    kernel_size = int(
        4 * sigma + 1
    )  # Kernel size, covering +/- 4 standard deviations
    points = torch.tensor(points, dtype=torch.float32)
    kernel = gaussian_kernel(kernel_size, sigma).unsqueeze(0)
    # Apply padding to handle borders
    points_padded = torch.nn.functional.pad(
        points, (kernel_size // 2, kernel_size // 2), mode='reflect'
    )
    smoothed_points = torch.nn.functional.conv1d(
        points_padded.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0)
    )
    return smoothed_points.squeeze().numpy()


def plot_losses(losses, title, eval_interval=None, ylabel='Loss'):
    """Plots smoothed loss curve."""
    if eval_interval:
        x_values = [i * eval_interval for i in range(len(losses))]
    else:
        x_values = list(range(len(losses)))
    plt.plot(x_values, smooth_curve(losses, sigma=2), label=title)
    plt.xlabel('Steps')
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    device = next(model.parameters()).device
    losses = []

    # torch.cuda.synchronize()
    # t0 = time.time()

    for X, y in tqdm(loader, desc='Evaluating', unit='iter'):
        X, y = X.to(device), y.to(device)
        preds, loss = model(X, y)
        loss, _ = loss
        losses.append(loss.item())

    # torch.cuda.synchronize()
    # t1 = time.time()
    # dt = t1 - t0
    # mfu = model.estimate_mfu(loader.batch_size * len(loader), dt)
    # print(f"Evaluation MFU: {mfu:.2f}")

    model.train()
    return np.mean(losses)


def main():
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Hyperparameters
    batch_size = (
        5  # How many independent sequences will we process in parallel?
    )
    max_len = 1_000  # What is the maximum context length for predictions?
    num_epochs = 3
    eval_interval = 100
    lr = 7.5e-4
    device = (
        'mps'
        if torch.backends.mps.is_available()
        else 'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )
    print('Running on device:', device)
    n_embd = 37
    d_out = 29
    n_head = 1  # Constraint: n_embd % n_head == 0
    scale = 16  # 4 is default
    n_layer = 6
    dropout = 0.25
    bias = False
    patience = 5  # Number of validation steps to wait for improvement

    # Data loading
    controller = 'Ant-v1'  # If not Ant-v1, add /3000/ after {controller} and change the loss function
    train_inputs = f'../data/{controller}/yagiz_train_inputs.npy'
    train_targets = f'../data/{controller}/yagiz_train_targets.npy'
    val_inputs = f'../data/{controller}/yagiz_val_inputs.npy'
    val_targets = f'../data/{controller}/yagiz_val_targets.npy'
    print(f'Training on {controller} task.')

    # Get dataloaders
    train_loader = get_dataloader(
        train_inputs, train_targets, batch_size, device
    )
    val_loader = get_dataloader(val_inputs, val_targets, batch_size, device)

    # Set the loss function based on the controller
    loss_fn = (
        HalfCheetahLoss()
        if controller == 'HalfCheetah-v1'
        else Walker2DLoss()
        if controller == 'Walker2D-v1'
        else AntLoss()
    )

    configs = {
        'n_layer': n_layer,
        'n_head': n_head,
        'n_embd': n_embd,
        # 'scale': scale,
        'd_out': d_out,
        'max_len': max_len,
        # 'bias': bias,
        'dropout': dropout,
        'loss_fn': loss_fn,
    }

    configs = TransformerConfig(**configs)
    model = Transformer(configs)
    model = model.to(device)
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        print(
            'PyTorch >= 2.0 detected. '
            'Using Memory-Efficient Attention (Rabe et al., 2022) '
            'and Flash Attention (Dao et al., 2023).'
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    patience_counter = 0

    # Initialize lists to store losses
    train_losses = []
    val_losses = []
    grad_norms = []

    # Check available individual losses once before the training loop
    metric_losses = {
        'coordinate_loss': [],
        'orientation_loss': [],
        'angle_loss': [],
        'coordinate_velocity_loss': [],
        'angular_velocity_loss': [],
    }

    pbar = tqdm(
        range(num_epochs * len(train_loader)), desc='Training', unit='iter'
    )
    for epoch in range(num_epochs):
        for step, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)

            # Evaluate the loss
            preds, loss = model(xb, yb)
            loss, metrics = loss
            train_losses.append(loss.item())

            # Check if metric exists first for the given loss function
            for metric in metric_losses:
                if metric in metrics:
                    metric_losses[metric].append(metrics[metric])

            # Check if any inputs, outputs or weights contain NaNs
            if torch.isnan(preds).any() or torch.isnan(loss):
                print('NaN detected!')
                print('Inputs: ', xb)
                print('Outputs: ', preds)
                print('Loss: ', loss)
                for name, param in model.named_parameters():
                    if param is not None and torch.isnan(param).any():
                        print(f'NaN in {name}')

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Print gradients to check for NaN and compute grad norm
            grad_norm = 0.0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
                    if torch.isnan(param.grad).any():
                        print(f'NaN gradient in {name}')
            grad_norm = grad_norm**0.5
            grad_norms.append(grad_norm)

            optimizer.step()

            # Evaluate on validation set
            total_steps = epoch * len(train_loader) + step
            if (
                total_steps % eval_interval == 0
            ) or total_steps == num_epochs * len(train_loader) - 1:
                val_loss = evaluate(model, val_loader)
                val_losses.append(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(
                        model.state_dict(), f'best_{controller}.safetensors'
                    )
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(
                        f'Early stopping triggered. Best validation loss: {best_val_loss:.4f}'
                    )
                    break

            postfix_dict = {
                'tr_loss': loss.item(),
                'val_loss': val_losses[-1] if len(val_losses) > 0 else None,
                'grd_nrm': grad_norm,
            }
            for metric in metrics:
                postfix_dict[metric] = metrics[metric]

            pbar.set_postfix(postfix_dict)
            pbar.update(1)

    plt.style.use('seaborn-v0_8-whitegrid')
    if not os.path.exists('results'):
        os.makedirs('results')

    # Plot training and validation losses (main losses - losses.png)
    plt.figure(figsize=(10, 5))
    plot_losses(train_losses, 'Training Loss')
    plot_losses(val_losses, 'Validation Loss', eval_interval)
    plt.title(f'Training and Validation Losses on {controller} Task')
    plt.tight_layout()
    plt.savefig(f'results/{controller}_losses.png', dpi=300)
    plt.show()
    plt.close()

    # Plot other losses and gradient norm (other losses - details.png)
    plt.figure(figsize=(10, 5))
    for metric, losses in metric_losses.items():
        plot_losses(losses, metric)
    plot_losses(grad_norms, 'Gradient Norm', ylabel='Gradient Norm')
    plt.title(f'Other Losses, Gradient Norm Over Time on {controller} Task')
    plt.tight_layout()
    plt.savefig(f'results/{controller}_details.png', dpi=300)
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
