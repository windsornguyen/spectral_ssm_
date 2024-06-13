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
import safetensors
import torch
import torch.nn.functional as F
import torch.distributed as dist
from safetensors.torch import save_file
from safetensors import safe_open
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from losses.loss_ant import AntLoss
from losses.loss_cheetah import HalfCheetahLoss
from losses.loss_walker import Walker2DLoss
from stu import experiment as exp, optimizer as opt
from stu.model import STUConfigs, Architecture
from stu.physics import physics_data
from transformer.model import Transformer, TransformerConfigs
# from mamba.model import Mamba, MambaConfig
# from jamba.model import Jamba, JambaConfig


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
    """Creates a 1D Gaussian kernel using PyTorch."""
    size = int(size) // 2
    x = torch.arange(-size, size + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def smooth_curve(points, sigma=2):
    """Applies 1D Gaussian smoothing on a list of points using PyTorch."""
    kernel_size = int(
        4 * sigma + 1
    )  # Kernel size, covering +/- 4 standard deviations
    points = torch.tensor(points, dtype=torch.float32)
    kernel = gaussian_kernel(kernel_size, sigma).unsqueeze(0).unsqueeze(0)
    points_padded = F.pad(
        points.unsqueeze(0).unsqueeze(0),
        (kernel_size // 2, kernel_size // 2),
        mode='reflect',
    )
    smoothed_points = F.conv1d(points_padded, kernel)
    return smoothed_points.squeeze().numpy()


def plot_losses(losses, title, eval_period=None, ylabel='Loss'):
    """Plots smoothed loss curve using PyTorch."""
    if eval_period:
        x_values = [i * eval_period for i in range(len(losses))]
    else:
        x_values = list(range(len(losses)))
    plt.plot(x_values, smooth_curve(losses, sigma=2), label=title)
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
    plot_losses(grad_norms, 'Gradient Norm', ylabel='Gradient Norm')
    plt.title(f'Other Losses, Gradient Norm Over Time on {controller} Task')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{controller}_details.png'), dpi=300)
    plt.close()


# To run the script: `torchrun --nproc_per_node=1 train.py`
def main() -> None:
    parser = argparse.ArgumentParser(description='Distributed Training Setup')
    parser.add_argument(
        '--models',
        nargs='+',
        default=['stu'],
        choices=['stu', 'transformer', 'mamba', 'jamba'],
        help='Models to train',
    )
    args = parser.parse_args()

    # Defaults specific to the Princeton HPC cluster; modify to your own setup.
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
        16 // world_size
    )  # scale batch size for distributed training
    val_batch_size: int = (
        16 // world_size
    )  # scale batch size for distributed training
    num_epochs: int = 3
    eval_period: int = 30
    patience: int = 10
    checkpoint_dir: str = 'checkpoints'

    # Optimizer hyperparameters
    weight_decay: float = 1e-1
    m_y_learning_rate: float = 5e-5
    m_y_weight_decay: float = 0

    # STU hyperparameters
    d_model: int = 37
    d_target: int = 29
    num_layers: int = 1
    dropout: float = 0.25
    input_len: int = 1000
    num_eigh: int = 24
    auto_reg_k_u: int = 3
    auto_reg_k_y: int = 2
    learnable_m_y: bool = True
    stu_lr: float = 7.5e-4

    # Transformer hyperparameters
    n_layer: int = 1
    n_head: int = 1
    n_embd: int = 37
    scale: int = 4
    d_out: int = 29
    max_len: int = 1_000
    bias: bool = False  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    dropout: float = 0.25
    use_dilated_attn: bool = False
    transformer_lr: float = 7.5e-4

    # Mamba hyperparameters
    # TBW

    # Jamba hyperparameters
    # TBW

    if main_process:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        if not os.path.exists('plots/'):
            os.makedirs('plots/')

    controller = 'Ant-v1'
    train_inputs = f'data/{controller}/test_inputs.npy'
    train_targets = f'data/{controller}/test_targets.npy'
    val_inputs = f'data/{controller}/val_inputs.npy'
    val_targets = f'data/{controller}/val_targets.npy'

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
    )
    num_steps: int = len(train_loader) * num_epochs
    warmup_steps: int = num_steps // 10

    models = {}
    experiments = {}
    loss_fn = {
        'HalfCheetah-v1': HalfCheetahLoss,
        'Walker2D-v1': Walker2DLoss,
        'Ant-v1': AntLoss,
    }[controller]()

    # Define the models based on flags
    if 'stu' in args.models:
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

        models['stu'] = stu_model.module if world_size > 1 else stu_model
        models['stu'].train()

        stu_optimizer, stu_scheduler = opt.get_optimizer(
            models['stu'],
            num_steps=num_steps,
            warmup_steps=warmup_steps,
            learning_rate=stu_lr,
            weight_decay=weight_decay,
            m_y_learning_rate=m_y_learning_rate,
            m_y_weight_decay=m_y_weight_decay,
        )

        experiments['stu'] = exp.Experiment(
            model=models['stu'],
            loss_fn=loss_fn,
            optimizer=stu_optimizer,
            scheduler=stu_scheduler,
            device=device,
        )

    if 'transformer' in args.models:
        transformer_configs = TransformerConfigs(
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            scale=scale,
            d_out=d_out,
            max_len=max_len,
            bias=bias,
            dropout=dropout,
            use_dilated_attn=use_dilated_attn,
            loss_fn=loss_fn,
        )
        transformer_model = Transformer(transformer_configs).to(device)

        if world_size > 1:
            transformer_model = DDP(
                transformer_model,
                device_ids=[local_rank],
                output_device=local_rank,
            )

        models['transformer'] = (
            transformer_model.module if world_size > 1 else transformer_model
        )
        models['transformer'].train()

        transformer_optimizer, transformer_scheduler = opt.get_optimizer(
            models['transformer'],
            num_steps=num_steps,
            warmup_steps=warmup_steps,
            learning_rate=transformer_lr,
            weight_decay=weight_decay,
        )

        experiments['transformer'] = exp.Experiment(
            model=models['transformer'],
            loss_fn=loss_fn,
            optimizer=transformer_optimizer,
            scheduler=transformer_scheduler,
            device=device,
        )

    # TO BE ADDED!
    # if 'mamba' in args.models:
    #     mamba_configs = MambaConfig(
    #         # Add Mamba-specific configuration arguments here
    #         loss_fn=loss_fn
    #     )
    #     mamba_model = Mamba(mamba_configs).to(device)
    #     if world_size > 1:
    #         mamba_model = DDP(
    #             mamba_model,
    #             device_ids=[local_rank],
    #             output_device=local_rank
    #         )
    #     models['mamba'] = mamba_model.module if world_size > 1 else mamba_model
    #     models['mamba'].train()
    #
    #     mamba_optimizer, mamba_scheduler = opt.get_optimizer(
    #         models['mamba'],
    #         num_steps=num_steps,
    #         warmup_steps=warmup_steps,
    #         learning_rate=mamba_lr,
    #         weight_decay=weight_decay,
    #     )
    #
    #     experiments['mamba'] = exp.Experiment(
    #         model=models['mamba'],
    #         loss_fn=loss_fn,
    #         optimizer=mamba_optimizer,
    #         scheduler=mamba_scheduler,
    #         device=device
    #     )

    # if 'jamba' in args.models:
    #     jamba_configs = JambaConfig(
    #         # Add Jamba-specific configuration arguments here
    #         loss_fn=loss_fn
    #     )
    #     jamba_model = Jamba(jamba_configs).to(device)
    #     if world_size > 1:
    #         jamba_model = DDP(
    #             jamba_model,
    #             device_ids=[local_rank],
    #             output_device=local_rank
    #         )
    #     models['jamba'] = jamba_model.module if world_size > 1 else jamba_model
    #     models['jamba'].train()
    #
    #     jamba_optimizer, jamba_scheduler = opt.get_optimizer(
    #         models['jamba'],
    #         num_steps=num_steps,
    #         warmup_steps=warmup_steps,
    #         learning_rate=jamba_lr,
    #         weight_decay=weight_decay,
    #     )
    #
    #     experiments['jamba'] = exp.Experiment(
    #         model=models['jamba'],
    #         loss_fn=loss_fn,
    #         optimizer=jamba_optimizer,
    #         scheduler=jamba_scheduler,
    #         device=device
    #     )

    best_val_losses = {model_name: float('inf') for model_name in args.models}
    patient_counters = {model_name: 0 for model_name in args.models}
    best_model_step = {model_name: 0 for model_name in args.models}
    best_checkpoints = {}

    # Initialize lists to store losses and metrics for each model
    train_losses = {model_name: [] for model_name in args.models}
    val_losses = {model_name: [] for model_name in args.models}
    grad_norms = {model_name: [] for model_name in args.models}
    metric_losses = {
        model_name: {
            'coordinate_loss': [],
            'orientation_loss': [],
            'angle_loss': [],
            'coordinate_velocity_loss': [],
            'angular_velocity_loss': [],
        }
        for model_name in args.models
    }

    if main_process:
        grmr = 'models' if len(args.models) > 1 else 'model'
        models_str = (
            ', '.join(args.models[:-1]) + f', and {args.models[-1]}'
            if len(args.models) > 1
            else args.models[0]
        )
        msg = f"Lyla: We'll be training the {models_str} {grmr} on the {controller} task with"
        if world_size > 1:
            print(
                f'{msg} {device} on rank {rank + 1}/{world_size}'
                f' utilizing {world_size} distributed processes.'
            )
        else:
            print(f'{msg} {device} today.')

    # Prepare progress bars for training
    pbars = {
        model_name: tqdm(
            range(num_epochs * len(train_loader)),
            desc=f'Training {model_name}',
            unit='step',
            position=i,
        )
        for i, model_name in enumerate(models)
    }

    # TODO: Check that setting this to True doesn't break distributed training.
    torch.autograd.set_detect_anomaly(True)

    # Training loop!
    for epoch in range(num_epochs):
        for step, (inputs, targets) in enumerate(train_loader):
            for model_name in models:
                experiment = experiments[model_name]
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

                # Append the losses and metrics for each model
                train_losses[model_name].append(train_metrics['loss'])
                grad_norms[model_name].append(train_metrics['grad_norm'])
                for metric in metric_losses[model_name]:
                    if metric in train_metrics:
                        metric_losses[model_name][metric].append(
                            train_metrics[metric]
                        )

                if main_process:
                    current_lrs = experiment.scheduler.get_last_lr()
                    default_lr = current_lrs[0]
                    m_y_lr = current_lrs[1] if len(current_lrs) > 1 else None
                    lrs = f'{default_lr:.2e}'
                    if m_y_lr is not None:
                        lrs += f', m_y_lr={m_y_lr:.2e}'
                    postfix_dict = {
                        f'{model_name}_tr_loss': train_metrics['loss'],
                        f'{model_name}_val_loss': val_losses[model_name][-1]
                        if len(val_losses[model_name]) > 0
                        else None,
                        f'{model_name}_grd_nrm': train_metrics['grad_norm'],
                        f'{model_name}_lr': lrs,
                    }
                    for metric in train_metrics:
                        if metric in metric_losses[model_name]:
                            postfix_dict[f'{model_name}_{metric}'] = (
                                train_metrics[metric]
                            )
                    pbars[model_name].set_postfix(postfix_dict)

            if main_process:
                for model_name in models:
                    pbars[model_name].update(1)

            total_steps = epoch * len(train_loader) + step

            if total_steps > 0 and total_steps % 10 == 0:
                if main_process:
                    colored_print(f'\nStep: {total_steps}', Colors.BOLD)
                    for model_name in models:
                        colored_print(
                            f'{model_name} - Train Loss: {train_losses[model_name][-1]:.4f}',
                            Colors.OKBLUE,
                        )

            if total_steps > 0 and total_steps % eval_period == 0:
                for model_name, experiment in experiments.items():
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

                    val_losses[model_name].append(val_metrics['loss'])

                    if main_process:
                        colored_print(
                            f'\nLyla: Evaluating the {model_name} model on step {total_steps} Loss: {val_metrics["loss"]:.2f}.',
                            Colors.OKCYAN,
                        )

                        val_loss = val_metrics['loss']
                        if val_loss < best_val_losses[model_name]:
                            best_val_losses[model_name] = val_loss
                            best_model_step[model_name] = total_steps
                            patient_counters[model_name] = 0
                            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                            checkpoint_filename = f'{model_name}-{controller}-chkpt-step{total_steps}-{timestamp}.safetensors'
                            checkpoint_path = os.path.join(
                                checkpoint_dir, checkpoint_filename
                            )
                            best_checkpoints[model_name] = checkpoint_filename

                            if dist.is_initialized():
                                # Save the model on the main process and broadcast it to all processes
                                if main_process:
                                    save_file(
                                        models[model_name].module.state_dict(),
                                        checkpoint_path,
                                    )
                                dist.barrier()
                                map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
                                models[model_name].load_state_dict(
                                    safetensors.load_file(
                                        checkpoint_path, device=map_location
                                    )
                                )
                            else:
                                save_file(
                                    models[model_name].state_dict(),
                                    checkpoint_path,
                                )

                            colored_print(
                                f'Lyla: Wow! We have a new personal best for the {model_name} model at step {total_steps}. The validation loss improved to: {val_loss:.4f}! Checkpoint saved as {checkpoint_path}',
                                Colors.OKGREEN,
                            )
                        else:
                            patient_counters[model_name] += 1
                            colored_print(
                                f'Lyla: No improvement in validation loss for the {model_name} model for {patient_counters[model_name]} eval periods. Current best loss: {best_val_losses[model_name]:.4f}.',
                                Colors.WARNING,
                            )

                            if patient_counters[model_name] >= patience:
                                colored_print(
                                    f'Lyla: We have reached the patience limit of {patience} for the {model_name} model. Stopping the training early at step {total_steps}...',
                                    Colors.FAIL,
                                )
                                if dist.is_initialized():
                                    dist.barrier()
                                return

    for pbar in pbars.values():
        pbar.close()

    if main_process:
        for model_name in args.models:
            if model_name in best_checkpoints:
                best_checkpoint_path = os.path.join(
                    checkpoint_dir, best_checkpoints[model_name]
                )

                if dist.is_initialized():
                    # Load the best checkpoint on the main process and broadcast it to all processes
                    if main_process:
                        with safe_open(
                            best_checkpoint_path, framework='pt', device=rank
                        ) as f:
                            state_dict = {k: f.get_tensor(k) for k in f.keys()}
                            models[model_name].load_state_dict(state_dict)
                    dist.barrier()
                else:
                    with safe_open(
                        best_checkpoint_path, framework='pt', device='cpu'
                    ) as f:
                        state_dict = {k: f.get_tensor(k) for k in f.keys()}
                        models[model_name].load_state_dict(state_dict)

                print(
                    f"\nLyla: Here's the best model information for the {model_name} model:"
                )
                print(f'    Best model at step {best_model_step[model_name]}')
                print(
                    f'    Best model validation loss: {best_val_losses[model_name]:.4f}'
                )
                print(
                    f'    Best model checkpoint saved at: {best_checkpoint_path}'
                )

                # Save the training details to a file
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                training_details = (
                    f'training_details_{model_name}_{timestamp}.txt'
                )
                with open(training_details, 'w') as f:
                    f.write(
                        f'Training completed for {model_name} on {controller} at: {datetime.now()}\n'
                    )
                    f.write(f'Best model step: {best_model_step[model_name]}\n')
                    f.write(
                        f'Best model validation loss: {best_val_losses[model_name]:.4f}\n'
                    )
                    f.write(
                        f'Best model checkpoint saved at: {best_checkpoint_path}\n'
                    )
                print(
                    f'Lyla: Congratulations on completing the training run for the {model_name} model! Details are saved in {training_details}.'
                )
            else:
                print(
                    f'\nLyla: No best checkpoint found for the {model_name} model. The model did not improve during training.'
                )

        for model_name in models:
            if train_losses[model_name] and val_losses[model_name]:
                plot_metrics(
                    train_losses[model_name],
                    val_losses[model_name],
                    metric_losses[model_name],
                    grad_norms[model_name],
                    f'plots/{model_name}/',
                    controller,
                    eval_period,
                )
            else:
                print(
                    f'No training data available for plotting the {model_name} model.'
                )

        # Plot validation losses for all models
        plt.figure(figsize=(8, 4))
        for model_name in models:
            if val_losses[model_name]:
                plot_losses(val_losses[model_name], model_name, eval_period)
        plt.xlabel('Steps', fontsize=12)
        plt.ylabel('Validation Loss', fontsize=12)
        plt.title(f'Validation Losses for All Models on {controller} Task', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join('plots', f'{controller}_losses.png'), dpi=300)
        plt.close()

        print('Lyla: It was a pleasure assisting you. Until next time!')


if __name__ == '__main__':
    main()
    if dist.is_initialized():
        cleanup()
