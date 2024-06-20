# =============================================================================#
# Authors: Isabel Liu, Yagiz Devre, Windsor Nguyen
# File: train.py
# =============================================================================#

"""Training loop for physics sequence prediction."""

import argparse
from datetime import datetime
import os
import random
from socket import gethostname

import matplotlib.pyplot as plt
import numpy as np
import safetensors  # TODO: Replace this with safetensors.torch and change fns accordingly
import torch
import torch.nn.functional as F
import torch.distributed as dist
from safetensors.torch import save_file
from safetensors import safe_open
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import sys
sys.path.insert(0, '/scratch/gpfs/yl3030/spectral_ssm')

from losses.loss_ant import AntLoss
from losses.loss_cheetah import HalfCheetahLoss
from losses.loss_walker import Walker2DLoss
from stu.physics import physics_data
from stu import experiment as exp, optimizer as opt
from stu.model import STUConfigs, Architecture


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def colored_print(text, color):
    print(f'{color}{text}{Colors.ENDC}')


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup(
    rank: int, world_size: int, gpus_per_node: int
) -> tuple[torch.device, int, int]:
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

    return device, local_rank, world_size


def cleanup():
    dist.destroy_process_group()


def gaussian_kernel(size, sigma):
    """Create a 1D Gaussian kernel."""
    size = int(size) // 2
    x = torch.arange(-size, size + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def smooth_curve(points, sigma=2):
    """Applies 1D Gaussian smoothing on a list of points using PyTorch."""
    kernel_size = int(4 * sigma + 1)  # Kernel size, covering +/- 4 stddevs
    points = torch.tensor(points, dtype=torch.float32)

    if len(points) < kernel_size:
        return (
            points.numpy()
        )  # Return original points if not enough for smoothing

    kernel = gaussian_kernel(kernel_size, sigma).unsqueeze(0).unsqueeze(0)
    points_padded = F.pad(
        points.unsqueeze(0).unsqueeze(0),
        (kernel_size // 2, kernel_size // 2),
        mode='reflect',
    )
    smoothed_points = F.conv1d(points_padded, kernel)
    return smoothed_points.squeeze().numpy()


def plot_losses(losses, eval_period=None, ylabel='Loss'):
    """Plots smoothed loss curve using PyTorch."""
    if eval_period:
        x_values = [i * eval_period for i in range(len(losses))]
    else:
        x_values = list(range(len(losses)))
    plt.plot(x_values, smooth_curve(losses, sigma=2))
    plt.xlabel('Steps')
    plt.ylabel(ylabel)
    plt.legend()


def plot_metrics(
    train_losses,
    val_losses,
    metric_losses,
    grad_norms,
    output_dir,
    controller,
    eval_period,
):
    """Plots training and validation losses, other metric losses, and gradient norms using PyTorch."""
    plt.style.use('seaborn-v0_8-whitegrid')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot training and validation losses (main losses - losses.png)
    plt.figure(figsize=(10, 5))
    plot_losses(train_losses, 'Training Loss')
    plot_losses(val_losses, 'Validation Loss', eval_period)
    plt.title(f'Training and Validation Losses on {controller} Task')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{controller}_losses.png'), dpi=300)
    plt.close()

    # Plot other losses (other losses - details.png)
    plt.figure(figsize=(10, 5))
    for metric, losses in metric_losses.items():
        plot_losses(losses, metric)

    # TODO: Since we use AMP, prune out super high grad norms at the start
    plot_losses(grad_norms, 'Gradient Norm', ylabel='Gradient Norm')
    plt.title(f'Other Losses, Gradient Norm Over Time on {controller} Task')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{controller}_details.png'), dpi=300)
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

    # General training hyperparameters
    train_batch_size: int = (
        10 // world_size
    )  # scale batch size for distributed training
    val_batch_size: int = (
        10 // world_size
    )  # scale batch size for distributed training
    num_epochs: int = 3
    eval_period: int = 30
    patience: int = 15
    checkpoint_dir: str = 'checkpoints_halfcheetah_obs_2l_10bsz'

    # Optimizer hyperparameters
    weight_decay: float = 1e-1
    m_y_learning_rate: float = 5e-5
    m_y_weight_decay: float = 0

    # STU hyperparameters
    d_model: int = 18
    d_target: int = 18
    num_layers: int = 2
    dropout: float = 0.25
    input_len: int = 1000
    num_eigh: int = 24
    auto_reg_k_u: int = 3
    auto_reg_k_y: int = 2
    learnable_m_y: bool = True
    stu_lr: float = 7.5e-4

    if main_process:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        if not os.path.exists('plots_halfcheetah_obs_2l_10bsz/'):
            os.makedirs('plots_halfcheetah_obs_2l_10bsz/')

    controller = 'HalfCheetah-v1'
    train_inputs = f'../data/{controller}/train_inputs_obs.npy'
    train_targets = f'../data/{controller}/train_targets.npy'
    val_inputs = f'../data/{controller}/val_inputs_obs.npy'
    val_targets = f'../data/{controller}/val_targets.npy'

    # Get dataloaders
    train_loader = physics_data.get_dataloader(
        inputs=train_inputs,
        targets=train_targets,
        batch_size=train_batch_size,
        device=device,
        distributed=world_size > 1,
        rank=rank,
        num_replicas=world_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = physics_data.get_dataloader(
        inputs=val_inputs,
        targets=val_targets,
        batch_size=val_batch_size,
        device=device,
        distributed=world_size > 1,
        rank=rank,
        num_replicas=world_size,
        num_workers=num_workers,
        pin_memory=True
    )

    steps_per_epoch = len(train_loader)
    num_steps: int = steps_per_epoch * num_epochs
    warmup_steps: int = num_steps // 10

    loss_fn = {
        'HalfCheetah-v1': HalfCheetahLoss,
        'Walker2D-v1': Walker2DLoss,
        'Ant-v1': AntLoss,
    }[controller]()

    stu_configs = STUConfigs(
        d_model=d_model,
        d_target=d_target,
        num_layers=num_layers,
        dropout=dropout,
        input_len=input_len,
        num_eigh=num_eigh,
        auto_reg_k_u=auto_reg_k_u,
        auto_reg_k_y=auto_reg_k_y,
        learnable_m_y=learnable_m_y,
        loss_fn=loss_fn,
    )
    stu_model = Architecture(stu_configs).to(device)

    if world_size > 1:
        stu_model = DDP(
            stu_model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    spectral_ssm = stu_model.module if world_size > 1 else stu_model
    spectral_ssm.train()

    stu_optimizer, stu_scheduler = opt.get_optimizer(
        spectral_ssm,
        num_steps=num_steps,
        warmup_steps=warmup_steps,
        learning_rate=stu_lr,
        weight_decay=weight_decay,
        m_y_learning_rate=m_y_learning_rate,
        m_y_weight_decay=m_y_weight_decay,
    )

    experiment = exp.Experiment(
        model=spectral_ssm,
        loss_fn=loss_fn,
        optimizer=stu_optimizer,
        scheduler=stu_scheduler,
        device=device,
    )

    best_val_loss = float('inf')
    patient_counter = 0
    best_model_step = 0
    best_checkpoint = None

    train_losses = []
    val_losses = []
    val_time_steps = []
    grad_norms = []
    metric_losses = {
        'coordinate_loss': [],
        'orientation_loss': [],
        'angle_loss': [],
        'coordinate_velocity_loss': [],
        'angular_velocity_loss': [],
    }

    if main_process:
        msg = f"Lyla: We'll be training the STU model on the {controller} task with"
        if world_size > 1:
            print(
                f'{msg} {device} on rank {rank + 1}/{world_size}'
                f' utilizing {world_size} distributed processes.'
            )
        else:
            print(f'{msg} {device} today.')

    pbar = tqdm(
        range(num_epochs * steps_per_epoch),
        desc='Training',
        unit='step',
    )

    # Training loop
    for epoch in range(num_epochs):
        for step, (inputs, targets) in enumerate(train_loader):
            train_metrics = experiment.step(inputs, targets)

            if dist.is_initialized():
                # Gather metrics from all processes
                gathered_metrics = [None] * world_size
                dist.all_gather_object(gathered_metrics, train_metrics)

                # Aggregate metrics across all processes
                train_metrics = {
                    k: sum(d[k] for d in gathered_metrics) / world_size
                    for k in train_metrics.keys()
                }

            relative_time_step = step + (epoch * steps_per_epoch)
            train_losses.append(train_metrics['loss'])
            grad_norms.append(train_metrics['grad_norm'])
            for metric in metric_losses:
                if metric in train_metrics:
                    metric_losses[metric].append(train_metrics[metric])

            if main_process:
                current_lrs = experiment.scheduler.get_last_lr()
                default_lr = current_lrs[0]
                lrs = f'{default_lr:.2e}'
                postfix_dict = {
                    'tr_loss': train_metrics['loss'],
                    'val_loss': val_losses[-1] if len(val_losses) > 0 else None,
                    'grd_nrm': train_metrics['grad_norm'],
                    'lr': lrs,
                }
                for metric in train_metrics:
                    if metric in metric_losses:
                        postfix_dict[metric] = train_metrics[metric]
                pbar.set_postfix(postfix_dict)
                pbar.update(1)

            if (step > 0 and step % eval_period == 0) or (step == 0) or (step == steps_per_epoch - 1):
                if main_process:
                    colored_print(f'\nStep: {relative_time_step}', Colors.BOLD)
                    colored_print(
                        f'\nSTU - Train Loss After {eval_period} Steps: {train_losses[-1]:.4f}',
                        Colors.OKBLUE,
                    )

                val_metrics = experiment.evaluate(val_loader)

                if dist.is_initialized():
                    # Gather validation metrics from all processes
                    gathered_metrics = [None] * world_size
                    dist.all_gather_object(gathered_metrics, val_metrics)

                    # Aggregate validation metrics across all processes
                    val_metrics = {
                        k: sum(d[k] for d in gathered_metrics) / world_size
                        for k in val_metrics.keys()
                    }

                val_losses.append(val_metrics['loss'])
                val_time_steps.append(relative_time_step)

                if main_process:
                    colored_print(
                        f'\nLyla: Evaluating the STU model on step {step} Loss: {val_metrics["loss"]:.2f}.',
                        Colors.OKCYAN,
                    )

                    val_loss = val_metrics['loss']
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_step = step
                        patient_counter = 0
                        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                        checkpoint_filename = f'stu-{controller}-chkpt-step{step}-{timestamp}.safetensors'
                        checkpoint_path = os.path.join(
                            checkpoint_dir, checkpoint_filename
                        )
                        best_checkpoint = checkpoint_filename

                        if dist.is_initialized():
                            # Save the model on the main process and broadcast it to all processes
                            if main_process:
                                save_file(
                                    stu_model.module.state_dict(),
                                    checkpoint_path,
                                )
                            dist.barrier()
                            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
                            stu_model.load_state_dict(
                                load_file(checkpoint_path, device=map_location)
                            )
                        else:
                            save_file(
                                stu_model.state_dict(),
                                checkpoint_path,
                            )

                        colored_print(
                            f'Lyla: Wow! We have a new personal best for the STU model at step {step}. The validation loss improved to: {val_loss:.4f}! Checkpoint saved as {checkpoint_path}',
                            Colors.OKGREEN,
                        )
                    else:
                        patient_counter += 1
                        colored_print(
                            f'Lyla: No improvement in validation loss for the STU model for {patient_counter} eval periods. Current best loss: {best_val_loss:.4f}.',
                            Colors.WARNING,
                        )

                        if patient_counter >= patience:
                            colored_print(
                                f'Lyla: We have reached the patience limit of {patience} for the STU model. Stopping the training early at step {step}...',
                                Colors.FAIL,
                            )
                            if main_process:
                                # Save the data points to files
                                np.save(
                                    'plots_halfcheetah_obs_2l_10bsz/stu_train_losses.npy',
                                    train_losses,
                                )
                                np.save(
                                    'plots_halfcheetah_obs_2l_10bsz/stu_val_losses.npy',
                                    val_losses,
                                )
                                np.save(
                                    'plots_halfcheetah_obs_2l_10bsz/stu_val_time_steps.npy',
                                    val_time_steps,
                                )
                                np.save(
                                    'plots_halfcheetah_obs_2l_10bsz/stu_grad_norms.npy',
                                    grad_norms,
                                )
                                for metric, losses in metric_losses.items():
                                    np.save(
                                        f'plots_halfcheetah_obs_2l_10bsz/stu_{metric}.npy',
                                        losses,
                                    )

                            if dist.is_initialized():
                                dist.barrier()
                            return

    pbar.close()

    if main_process:
        if best_checkpoint:
            best_checkpoint_path = os.path.join(checkpoint_dir, best_checkpoint)

            if dist.is_initialized():
                # Load the best checkpoint on the main process and broadcast it to all processes
                if main_process:
                    with safe_open(
                        best_checkpoint_path, framework='pt', device=rank
                    ) as f:
                        state_dict = {k: f.get_tensor(k) for k in f.keys()}
                        stu_model.load_state_dict(state_dict)
                dist.barrier()
            else:
                with safe_open(
                    best_checkpoint_path, framework='pt', device='cpu'
                ) as f:
                    state_dict = {k: f.get_tensor(k) for k in f.keys()}
                    stu_model.load_state_dict(state_dict)

            print(
                f"\nLyla: Here's the best model information for the STU model:"
            )
            print(f'    Best model at step {best_model_step}')
            print(f'    Best model validation loss: {best_val_loss:.4f}')
            print(f'    Best model checkpoint saved at: {best_checkpoint_path}')

            # Save the training details to a file
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            training_details = f'training_details_stu_{timestamp}.txt'
            with open(training_details, 'w') as f:
                f.write(
                    f'Training completed for STU on {controller} at: {datetime.now()}\n'
                )
                f.write(f'Best model step: {best_model_step}\n')
                f.write(f'Best model validation loss: {best_val_loss:.4f}\n')
                f.write(
                    f'Best model checkpoint saved at: {best_checkpoint_path}\n'
                )
            print(
                f'Lyla: Congratulations on completing the training run for the STU model! Details are saved in {training_details}.'
            )
        else:
            print(
                f'\nLyla: No best checkpoint found for the STU model. The model did not improve during training.'
            )

        # Save the data points to files
        np.save('plots_halfcheetah_obs_2l_10bsz/stu_train_losses.npy', train_losses)
        np.save('plots_halfcheetah_obs_2l_10bsz/stu_val_losses.npy', val_losses)
        np.save('plots_halfcheetah_obs_2l_10bsz/stu_val_time_steps.npy', val_time_steps)
        np.save('plots_halfcheetah_obs_2l_10bsz/stu_grad_norms.npy', grad_norms)
        for metric, losses in metric_losses.items():
            np.save(f'plots_halfcheetah_obs_2l_10bsz/stu_{metric}.npy', losses)

        print('Lyla: It was a pleasure assisting you. Until next time!')


if __name__ == '__main__':
    main()
    if dist.is_initialized():
        cleanup()
