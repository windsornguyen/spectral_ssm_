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
import torch
import torch.nn.functional as F
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from torch.nn import MSELoss
from data.mujoco.mujoco_data_all import create_dataloader, split_data
from stu import experiment as exp, optimizer as opt
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


# To run the script: `python -m transformer.train_transformer` from root dir.
def main() -> None:
    torch.multiprocessing.set_start_method('spawn')
    
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
        2 // world_size
    )  # scale batch size for distributed training
    val_batch_size: int = (
        2 // world_size
    )  # scale batch size for distributed training
    num_epochs: int = 1
    eval_period: int = 50
    patience: int = 15
    checkpoint_dir: str = 'checkpoints'

    # Optimizer hyperparameters
    weight_decay: float = 1e-1

    # Transformer hyperparameters
    EFFICIENT_NET_B6_D_OUT: int = 2304
    EFFICIENT_NET_B6_FEATURE_SIZE: int = 7
    n_layers: int = 6
    n_head: int = 6
    n_embd: int = 384  # Embedding dimension
    scale: int = 4 # GPT-2 had scale=4
    n_frames: int = 299
    max_n_frames: int = 300
    bias: bool = False  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    dropout: float = 0.25
    use_dilated_attn: bool = True
    loss_fn = MSELoss()

    use_dilated_attn: bool = False  # TODO: Finish up
    transformer_lr: float = 7.5e-4
    
    # Target dimension(2304 x 7 x 7, from EfficientNet-B6)
    d_out: int = EFFICIENT_NET_B6_D_OUT * EFFICIENT_NET_B6_FEATURE_SIZE**2

    # Input dimension
    d_in: int = d_out

    if main_process:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        if not os.path.exists('plots/'):
            os.makedirs('plots/')

    controller = 'Ant-v1'  # TODO: Use this eventually in data routing.
    feature_file = '/scratch/gpfs/mn4560/ssm/data/mujoco/ant_vectorized_final.pt'
    train_data, val_data = split_data(torch.load(feature_file))

    train_loader = create_dataloader(
        video_features=train_data,
        batch_size=train_batch_size,
        num_workers=num_workers,
        device=device,
        # num_pred_steps=5,
        preprocess=True,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = create_dataloader(
        video_features=val_data,
        batch_size=val_batch_size,
        num_workers=num_workers,
        device=device,
        # num_pred_steps=5,
        preprocess=True,
        shuffle=False,
        pin_memory=True,
    )

    num_steps: int = len(train_loader) * num_epochs
    warmup_steps: int = num_steps // 10

    transformer_configs = TransformerConfigs(
        n_layers=n_layers,
        n_head=n_head,
        n_embd=n_embd,
        scale=scale,
        d_in=d_in,
        d_out=d_out,
        n_frames=n_frames,
        max_n_frames=max_n_frames,
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

    transformer_model.train()

    transformer_optimizer, transformer_scheduler = opt.get_optimizer(
        transformer_model,
        num_steps=num_steps,
        warmup_steps=warmup_steps,
        learning_rate=transformer_lr,
        weight_decay=weight_decay,
    )

    experiment = exp.Experiment(
        model=transformer_model,
        loss_fn=loss_fn,
        optimizer=transformer_optimizer,
        scheduler=transformer_scheduler,
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
        msg = f"Lyla: We'll be training the Transformer model on the {controller} task with"
        if world_size > 1:
            print(
                f'{msg} {device} on rank {rank + 1}/{world_size}'
                f' utilizing {world_size} distributed processes.'
            )
        else:
            print(f'{msg} {device} today.')

    pbar = tqdm(
        range(num_epochs * len(train_loader)),
        desc='Training',
        unit='step',
    )

    # Training loop
    for _ in range(num_epochs):
        for step, (inputs, targets, file_names) in enumerate(train_loader):
            print('device inputs', inputs[0].device)
            print('device targets', targets[0].device)
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

            if step > 0 and step % eval_period == 0:
                if main_process:
                    colored_print(f'\nStep: {step}', Colors.BOLD)
                    colored_print(
                        f'\nTransformer - Train Loss After {eval_period} Steps: {train_losses[-1]:.4f}',
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
                val_time_steps.append(step)

                if main_process:
                    colored_print(
                        f'\nLyla: Evaluating the Transformer model on step {step} Loss: {val_metrics["loss"]:.2f}.',
                        Colors.OKCYAN,
                    )

                    val_loss = val_metrics['loss']
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_step = step
                        patient_counter = 0
                        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                        checkpoint_filename = f'transformer-{controller}-chkpt-step{step}-{timestamp}.safetensors'
                        checkpoint_path = os.path.join(
                            checkpoint_dir, checkpoint_filename
                        )
                        best_checkpoint = checkpoint_filename

                        if dist.is_initialized():
                            # Save the model on the main process and broadcast it to all processes
                            if main_process:
                                save_file(
                                    transformer_model.module.state_dict(),
                                    checkpoint_path,
                                )
                            dist.barrier()
                            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
                            transformer_model.load_state_dict(
                                load_file(checkpoint_path, device=map_location)
                            )
                        else:
                            save_file(
                                transformer_model.state_dict(),
                                checkpoint_path,
                            )

                        colored_print(
                            f'Lyla: Wow! We have a new personal best for the Transformer model at step {step}. The validation loss improved to: {val_loss:.4f}! Checkpoint saved as {checkpoint_path}',
                            Colors.OKGREEN,
                        )
                    else:
                        patient_counter += 1
                        colored_print(
                            f'Lyla: No improvement in validation loss for the Transformer model for {patient_counter} eval periods. Current best loss: {best_val_loss:.4f}.',
                            Colors.WARNING,
                        )

                        if patient_counter >= patience:
                            colored_print(
                                f'Lyla: We have reached the patience limit of {patience} for the Transformer model. Stopping the training early at step {step}...',
                                Colors.FAIL,
                            )
                            if main_process:
                                # Save the data points to files
                                np.save(
                                    'plots/transformer_train_losses.npy',
                                    train_losses,
                                )
                                np.save(
                                    'plots/transformer_val_losses.npy',
                                    val_losses,
                                )
                                np.save(
                                    'plots/transformer_val_time_steps.npy',
                                    val_time_steps,
                                )
                                np.save(
                                    'plots/transformer_grad_norms.npy',
                                    grad_norms,
                                )
                                for metric, losses in metric_losses.items():
                                    np.save(
                                        f'plots/transformer_{metric}.npy',
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
                        transformer_model.load_state_dict(state_dict)
                dist.barrier()
            else:
                with safe_open(
                    best_checkpoint_path, framework='pt', device='cpu'
                ) as f:
                    state_dict = {k: f.get_tensor(k) for k in f.keys()}
                    transformer_model.load_state_dict(state_dict)

            print(
                f"\nLyla: Here's the best model information for the Transformer model:"
            )
            print(f'    Best model at step {best_model_step}')
            print(f'    Best model validation loss: {best_val_loss:.4f}')
            print(f'    Best model checkpoint saved at: {best_checkpoint_path}')

            # Save the training details to a file
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            training_details = f'training_details_transformer_{timestamp}.txt'
            with open(training_details, 'w') as f:
                f.write(
                    f'Training completed for Transformer on {controller} at: {datetime.now()}\n'
                )
                f.write(f'Best model step: {best_model_step}\n')
                f.write(f'Best model validation loss: {best_val_loss:.4f}\n')
                f.write(
                    f'Best model checkpoint saved at: {best_checkpoint_path}\n'
                )
            print(
                f'Lyla: Congratulations on completing the training run for the Transformer model! Details are saved in {training_details}.'
            )
        else:
            print(
                f'\nLyla: No best checkpoint found for the Transformer model. The model did not improve during training.'
            )

        # Save the data points to files
        np.save('plots/transformer_train_losses.npy', train_losses)
        np.save('plots/transformer_val_losses.npy', val_losses)
        np.save('plots/transformer_val_time_steps.npy', val_time_steps)
        np.save('plots/transformer_grad_norms.npy', grad_norms)
        for metric, losses in metric_losses.items():
            np.save(f'plots/transformer_{metric}.npy', losses)

        print('Lyla: It was a pleasure assisting you. Until next time!')


if __name__ == '__main__':
    main()
    if dist.is_initialized():
        cleanup()
