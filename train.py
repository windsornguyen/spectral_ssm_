# =============================================================================#
# Authors: Isabel Liu, Yagiz Devre
# File: train.py
# =============================================================================#

"""Training loop for physics sequence prediction."""

import argparse
import numpy as np
import os
import random
import torch
import torch.distributed as dist
from typing import Tuple, Dict
from datetime import datetime
from spectral_ssm import physics_data
from spectral_ssm import experiment
from spectral_ssm import model
from spectral_ssm import optimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from spectral_ssm.loss_ant import AntLoss
from spectral_ssm.loss_cheetah import HalfCheetahLoss
from spectral_ssm.loss_walker import Walker2DLoss
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup(rank: int, world_size: int, gpus_per_node: int) -> tuple[torch.device, int, int]:
    """
    Adapts to distributed or non-distributed training environments.
    Chooses appropriate backend and device based on the available hardware and environment setup.
    Manages NCCL for NVIDIA GPUs, Gloo for CPUs, and potentially Gloo for Apple Silicon (MPS).
    """
    local_rank = rank % gpus_per_node if gpus_per_node > 0 else 0
    device = torch.device('cpu')  # Default to CPU
    backend = 'gloo'  # Default backend

    if world_size > 1 and 'SLURM_PROCID' in os.environ:
        if torch.cuda.is_available() and gpus_per_node > 0:
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
            backend = 'nccl'
            dist.init_process_group(
                backend=backend, rank=rank, world_size=world_size
            )
            print(
                f'host: {gethostname()}, rank: {rank}, local_rank: {local_rank}'
            )
            if rank == 0:
                print(f'Group initialized? {dist.is_initialized()}', flush=True)
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            dist.init_process_group(
                backend=backend, rank=rank, world_size=world_size
            )
            print(f'Using MPS on host: {gethostname()}, rank: {rank}')
            if rank == 0:
                print(f'Group initialized? {dist.is_initialized()}', flush=True)
    else:
        # Non-distributed fallback to the best available device
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')

    return 'cpu', local_rank, world_size


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


def plot_metrics(train_losses, val_losses, metric_losses, grad_norms, output_dir, controller, eval_interval):
    plt.style.use('seaborn-v0_8-whitegrid')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot training and validation losses (main losses - losses.png)
    plt.figure(figsize=(10, 5))
    plot_losses(train_losses, 'Training Loss')
    plot_losses(val_losses, 'Validation Loss', eval_interval)
    plt.title(f'Training and Validation Losses on {controller} Task')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{controller}_losses.png'), dpi=300)
    plt.show()
    plt.close()

    # Plot other losses (other losses - details.png)
    plt.figure(figsize=(10, 5))
    for metric, losses in metric_losses.items():
        plot_losses(losses, metric)
    plot_losses(grad_norms, 'Gradient Norm', ylabel='Gradient Norm')
    plt.title(f'Other Losses, Gradient Norm Over Time on {controller} Task')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{controller}_details.png'), dpi=300)
    plt.show()
    plt.close()


# To run the script: `torchrun --nproc_per_node=1 train.py`
def main() -> None:
    parser = argparse.ArgumentParser(description='Distributed Training Setup')
    parser.add_argument(
        '--local_rank', type=int, default=0
    )  # Might delete this

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('SLURM_PROCID', 0))
    gpus_per_node = int(os.environ.get('SLURM_GPUS_ON_NODE', 0))
    num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))

    device, local_rank, world_size = setup(rank, world_size, gpus_per_node)
    set_seed(42 + local_rank)
    main_process = local_rank == 0

    if main_process:
        print(
            "Lyla: Greetings! I'm Lyla, your friendly neighborhood AI training assistant."
        )

    # Hyperparameters
    train_batch_size: int = 5 // world_size # scale batch size for distributed training
    val_batch_size: int = 5 // world_size  # scale batch size for distributed training
    num_epochs: int = 3
    eval_period: int = 100
    learning_rate: float = 7.5e-4
    weight_decay: float = 1e-1
    m_y_learning_rate: float = 5e-5
    m_y_weight_decay: float = 0
    patience: int = 5
    checkpoint_dir: str = 'physics_checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    controller = 'Ant-v1'
    train_inputs = f'data/{controller}/yagiz_train_inputs.npy'
    train_targets = f'data/{controller}/yagiz_train_targets.npy'
    val_inputs = f'data/{controller}/yagiz_val_inputs.npy'
    val_targets = f'data/{controller}/yagiz_val_targets.npy'

    # Get dataloaders
    train_loader = physics_data.get_dataloader(train_inputs, train_targets, train_batch_size, device=device)
    val_loader = physics_data.get_dataloader(val_inputs, val_targets, val_batch_size, device=device)
    num_steps: int = len(train_loader) * num_epochs
    warmup_steps: int = num_steps // 10

    # Define the model
    spectral_ssm = model.Architecture(
        d_model=37,
        d_target=29,
        num_layers=6,
        dropout=0.25,
        input_len=1000,
        num_eigh=24,
        auto_reg_k_u=3,
        auto_reg_k_y=2,
        learnable_m_y=True,
    ).to(device)

    print(sum(p.numel() for p in spectral_ssm.parameters()) / 1e6, 'M parameters')

    # Wrap the model with DistributedDataParallel for distributed training
    # TODO: Distributed code is not ready yet
    if world_size > 1:
        spectral_ssm = DDP(
            spectral_ssm,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    spectral_ssm = spectral_ssm.module if world_size > 1 else spectral_ssm
    opt, scheduler = optimizer.get_optimizer(
        spectral_ssm,
        num_steps=num_steps,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        m_y_learning_rate=m_y_learning_rate,
        m_y_weight_decay=m_y_weight_decay,
    )

    loss_fn = HalfCheetahLoss() if controller == 'HalfCheetah-v1' else Walker2DLoss() if controller == 'Walker2D-v1' else AntLoss()
    exp = experiment.Experiment(model=spectral_ssm, loss_fn=loss_fn, optimizer=opt, device=device)
    msg = "Lyla: We'll be training with"

    if main_process:
        if world_size > 1:
            print(
                f'{msg} {device} on rank {rank + 1}/{world_size}, '
                f'utilizing {world_size} distributed processes.'
            )
        else:
            print(f'{msg} {device} today.')

    best_val_loss = float('inf')
    eval_periods_without_improvement = 0
    best_model_step = 0
    best_model_metrics = None
    best_checkpoint = None

    torch.autograd.set_detect_anomaly(True)

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
        'angular_velocity_loss': []
    }

    pbar = tqdm(range(num_epochs * len(train_loader)), desc='Training', unit='step') if main_process else range(num_epochs * len(train_loader))
    for epoch in range(num_epochs):
        for global_step, (inputs, targets) in enumerate(train_loader):
            train_metrics = exp.step(inputs, targets)

            # Append the losses
            train_losses.append(train_metrics["loss"])
            for metric in metric_losses:
                if metric in train_metrics:
                    metric_losses[metric].append(train_metrics[metric])

            if main_process:
                current_lrs = scheduler.get_last_lr()
                default_lr = current_lrs[0]
                m_y_lr = current_lrs[1]
                lrs = f"{default_lr:.2e}, m_y_lr={m_y_lr:.2e}"
                postfix_dict = {
                    'tr_loss': train_metrics["loss"],
                    'lr': lrs
                }
                for metric in train_metrics:
                    if metric in metric_losses:
                        postfix_dict[metric] = train_metrics[metric]  # Remove the item() call here
                pbar.set_postfix(postfix_dict)
                pbar.update(1)

            scheduler.step()

            if global_step > 0 and global_step % eval_period == 0:
                if main_process:
                    print(f"\nLyla: Lyla here! We've reached step {global_step}.")
                    print(
                        "Lyla: It's time for an evaluation update! Let's see how our model is doing..."
                    )

                val_metrics = exp.evaluate(val_loader)
                val_losses.append(val_metrics['loss'])

                if world_size > 1:
                    # Gather evaluation metrics from all processes
                    gathered_metrics = [None] * world_size
                    torch.distributed.all_gather_object(gathered_metrics, val_metrics)

                    if main_process:
                        # Aggregate metrics across all processes
                        total_loss = sum(metric['loss'] for metric in gathered_metrics) / world_size
                        print(
                            f'\nLyla: Evaluating the model on step {global_step}'
                            f' Average Loss: {total_loss:.4f}.'
                        )
                        val_metrics = {'loss': total_loss}
                else:
                    if main_process:
                        print(
                        f'\nLyla: Evaluating the model on step {global_step}'
                        f' Loss: {val_metrics["loss"]:.2f}.'
                    )

                if main_process:
                    val_loss = val_metrics['loss']
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_step = global_step
                        best_model_metrics = val_metrics
                        eval_periods_without_improvement = 0
                        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                        checkpoint_filename = f'checkpoint-step{global_step}-{timestamp}.pt'
                        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
                        best_checkpoint = checkpoint_filename
                        
                        torch.save(spectral_ssm.state_dict(), checkpoint_path)
                        print(
                            f'Lyla: Wow! We have a new personal best at step {global_step}.'
                            f' The validation loss improved to: {val_loss:.4f}!'
                            f' Checkpoint saved as {checkpoint_path}'
                        )
                    else:
                        eval_periods_without_improvement += 1
                        print(
                            f'Lyla: No improvement in validation loss for '
                            f'{eval_periods_without_improvement} eval periods. '
                            f'Current best loss: {best_val_loss:.4f}.'
                        )
                    if eval_periods_without_improvement >= patience:
                        print(
                            f'Lyla: We have reached the patience limit of {patience} '
                            f'epochs without improvement. Stopping the training early '
                            f'at step {global_step}...'
                        )
                        break
    pbar.close()

    # Load the best model checkpoint
    best_checkpoint_path = os.path.join(checkpoint_dir, best_checkpoint)
    spectral_ssm.load_state_dict(torch.load(best_checkpoint_path))

    # Print detailed information about the best model
    if main_process:
        print("\nLyla: Training completed! Nice work. Here's the best model information:")
        print(f'    Best model at step {best_model_step}')
        print(f'    Best model validation loss: {best_val_loss:.4f}')
        print(f'    Best model checkpoint saved at: {best_checkpoint_path}')

        # Save the training details to a file
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        training_details = f'training_details_{timestamp}.txt'
        with open(training_details, 'w') as f:
            f.write(f'Training completed at: {datetime.now()}\n')
            f.write(f'Best model step: {best_model_step}\n')
            f.write(f'Best model validation loss: {best_val_loss:.4f}\n')
            f.write(f'Best model checkpoint saved at: {best_checkpoint_path}\n')
        print(
            'Lyla: Congratulations on completing the training run!'
            f' Details are saved in {training_details}.'
            ' It was a pleasure assisting you. Until next time!'
        )

    # After training, plot the losses
    plot_metrics(train_losses, val_losses, metric_losses, grad_norms, 'plots/', controller, eval_period)


if __name__ == '__main__':
    main()
