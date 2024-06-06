import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from data import get_dataloader
from model import Transformer, TransformerConfig


def smooth_curve(points, sigma=2):
    return gaussian_filter1d(points, sigma=sigma)

def plot_losses(losses, title, eval_interval=None, ylabel='Loss'):
    if eval_interval:
        x_values = [i * eval_interval for i in range(len(losses))]
    else:
        x_values = list(range(len(losses)))
    plt.plot(x_values, smooth_curve(losses), label=title)
    plt.xlabel('Steps')
    plt.ylabel(ylabel)
    plt.legend()


def main():
    # Set seed for reproducibility
    torch.manual_seed(1337)

    # Hyperparameters
    batch_size = 128  # how many independent sequences will we process in parallel?
    max_len = 1_000  # what is the maximum context length for predictions?
    max_iters = 3_500 // 6
    eval_interval = (max_iters / 10) // 6
    learning_rate = 1e-3
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running on device:', device)
    n_embd = 24
    d_out = 18
    n_head = 1 # Constraint: n_embd % n_head == 0
    scale = 16 # 4 is default
    n_layer = 6
    dropout = 0.0
    bias = False
    patience = 10  # Number of validation steps to wait for improvement

    # Data loading
    controller = 'HalfCheetah-v1' # If not Ant-v1, add /3000/ after {controller}
    train_inputs = f'data/{controller}/3000/train_inputs.npy'
    train_targets = f'data/{controller}/3000/train_targets.npy'
    val_inputs = f'data/{controller}/3000/val_inputs.npy'
    val_targets = f'data/{controller}/3000/val_targets.npy'

    # Get dataloaders
    train_loader = get_dataloader(train_inputs, train_targets, 'train', batch_size, device)
    val_loader = get_dataloader(val_inputs, val_targets, 'val', batch_size, device)

    @torch.no_grad()
    def evaluate(model, loader):
        print('Evaluating on validation set...')
        model.eval()
        losses = []
        for X, y in tqdm(loader, desc='Evaluating', unit='iter'):
            X, y = X.to(device), y.to(device)
            preds, loss = model(X, y)
            loss, _ = loss
            losses.append(loss.item())
        model.train()
        return np.mean(losses)

    model_args = {
        'n_layer': n_layer,
        'n_head': n_head,
        'n_embd': n_embd,
        'scale': scale,
        'd_out': d_out,
        'max_len': max_len,
        'bias': bias, 
        'dropout': dropout
    }

    config = TransformerConfig(**model_args)
    model = Transformer(config)
    m = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    patience_counter = 0

    # Initialize lists to store losses
    train_losses = []
    val_losses = []
    grad_norms = []
    coord_losses = []
    orient_losses = []
    angle_losses = []
    coord_vel_losses = []
    ang_vel_losses = []

    pbar = tqdm(range(max_iters), desc='Training', unit='iter')
    val_loss = None
    for step in pbar:
        xb, yb = next(iter(train_loader))
        xb, yb = xb.to(device), yb.to(device)
        
        # Evaluate the loss
        preds, loss = model(xb, yb)
        loss, metrics = loss

        # Store total training loss and losses specific to each observation feature
        train_losses.append(loss.item())
        coord_losses.append(metrics['coordinate_loss'])
        orient_losses.append(metrics['orientation_loss'])
        angle_losses.append(metrics['angle_loss'])
        coord_vel_losses.append(metrics['coordinate_velocity_loss'])
        ang_vel_losses.append(metrics['angular_velocity_loss'])

        # Check if any inputs, outputs or weights contain NaNs
        if torch.isnan(preds).any() or torch.isnan(loss):
            print("NaN detected!")
            print("Inputs: ", xb)
            print("Outputs: ", preds)
            print("Loss: ", loss)
            for name, param in model.named_parameters():
                if param is not None and torch.isnan(param).any():
                    print(f"NaN in {name}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Print gradients to check for NaN and compute grad norm
        grad_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
                if torch.isnan(param.grad).any():
                    print(f"NaN gradient in {name}")
        grad_norm = grad_norm ** 0.5
        grad_norms.append(grad_norm)

        optimizer.step()

        # Evaluate on validation set
        if (step != 0 and step % eval_interval == 0) or step == max_iters - 1:
            val_loss = evaluate(model, val_loader)
            val_losses.append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f'best_{controller}.safetensors')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f'Early stopping triggered. Best validation loss: {best_val_loss:.4f}')
                break

        pbar.set_postfix({
            'tr_loss': loss.item(),
            'val_loss': val_loss,
            'grd_nrm': grad_norm,
            'coord_loss': metrics['coordinate_loss'],
            'orient_loss': metrics['orientation_loss'],
            'angle_loss': metrics['angle_loss'],
            'coord_vel_loss': metrics['coordinate_velocity_loss'],
            'angle_vel_loss': metrics['angular_velocity_loss']
        })

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
    plt.close()

    # Plot other losses and gradient norm (other losses - details.png)
    plt.figure(figsize=(10, 5))
    plot_losses(coord_losses, 'Coordinate Loss')
    plot_losses(orient_losses, 'Orientation Loss')
    plot_losses(angle_losses, 'Angle Loss')
    plot_losses(coord_vel_losses, 'Coordinate Velocity Loss')
    plot_losses(ang_vel_losses, 'Angular Velocity Loss')
    plot_losses(grad_norms, 'Gradient Norm', ylabel='Gradient Norm')
    plt.title('Other Losses, Gradient Norm over Time')
    plt.tight_layout()
    plt.savefig(f'results/{controller}_details.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    main()
