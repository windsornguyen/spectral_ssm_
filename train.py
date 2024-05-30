# =============================================================================#
# Authors: Isabel Liu, Yagiz Devre
# File: train.py
# =============================================================================#

"""Training loop for physics sequence preDiction."""

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_distributed_env(local_rank: int) -> Tuple[torch.device, int, int]:
    """
    Sets up the distributed training environment for various accelerators.
    - Uses NCCL for NVIDIA GPUs (CUDA).
    - Uses Gloo for CPUs and (potentially) for Apple Silicon (MPS).
    """
    # Set Gloo as default backend since NCCL does not support MPS
    backend = 'gloo'
    device = torch.device('cpu')

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        backend = 'nccl'

    # NOTE: May have to run `export PYTORCH_ENABLE_MPS_FALLBACK=1`
    # since PyTorch-MPS is not yet fully supported.
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        backend = 'gloo'

    dist.init_process_group(backend=backend, init_method='env://')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    return device, rank, world_size


# To run the script: `torchrun --nproc_per_node=1 train.py`
def main() -> None:
    parser = argparse.ArgumentParser(description='Distributed Training Setup')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    device, rank, world_size = setup_distributed_env(args.local_rank)
    set_seed(42 + rank)
    main_process = (rank == 0)

    if main_process:
        print("Lyla: Greetings! I'm Lyla, your friendly neighborhood AI training assistant.")

    # Hyperparameters
    train_batch_size: int = 10 // world_size # scale batch size for distributed training
    eval_batch_size: int = 10 // world_size  # scale batch size for distributed training
    num_steps: int = 3_500
    eval_period: int = 20
    warmup_steps: int = 350
    learning_rate: float = 1e-3
    weight_decay: float = 1e-1
    m_y_learning_rate: float = 5e-5
    m_y_weight_decay: float = 0
    patience: int = 10
    checkpoint_dir: str = 'checkpoints'
    if main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Define the model
    spectral_ssm = model.Architecture(
        d_model=256,
        d_target=29,
        num_layers=6,
        dropout=0.1,
        input_len=1000 * 37,
        num_eigh=24,
        auto_reg_k_u=3,
        auto_reg_k_y=2,
        learnable_m_y=True,
    ).to(device)

    # Wrap the model with DistributedDataParallel for distributed training
    spectral_ssm = torch.nn.SyncBatchNorm.convert_sync_batchnorm(spectral_ssm)
    spectral_ssm = DDP(
        spectral_ssm,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        
        # TODO: There is a dead neuron for some reason...
            # module.layers.0.0.m_u
        # TODO: Removing this parameter results in error in distributed training
        find_unused_parameters=True,
    )

    opt, scheduler = optimizer.get_optimizer(
        spectral_ssm.module,
        num_steps=num_steps,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        m_y_learning_rate=m_y_learning_rate,
        m_y_weight_decay=m_y_weight_decay,
    )
    exp = experiment.Experiment(model=spectral_ssm, optimizer=opt, device=device)
    msg = "Lyla: We'll be training with"

    if main_process:
        if world_size > 1:
            print(
                f'{msg} {device} on rank {rank + 1}/{world_size}, '
                f'utilizing {world_size} distributed processes.'
            )
        else:
            print(f'{msg} {device} today.')

    train_loader = physics_data.get_dataloader('input_data.npy', 'target_data.npy', 10)

    eval_loader = physics_data.get_dataloader('input_data_eval.npy', 'target_data_eval.npy', 10)

    if main_process:
        print(
            "Lyla: All set! Everything's loaded up and ready to go. "
            'May the compute Gods be by our sides...'
        )

    best_val_loss = float('inf')
    eval_periods_without_improvement = 0
    best_model_step = 0
    best_model_metrics = None
    best_checkpoint = None

    torch.autograd.set_detect_anomaly(True)

    pbar = tqdm(range(num_steps), desc='Training Progress', unit='step') if main_process else range(num_steps)
    for global_step in pbar:
        inputs, targets = next(iter(train_loader))
        train_metrics = exp.step(inputs, targets)

        if main_process:
            current_lrs = scheduler.get_last_lr()
            default_lr = current_lrs[0]
            m_y_lr = current_lrs[1]
            lrs = f"{default_lr:.2e}, m_y_lr={m_y_lr:.2e}"
            pbar.set_postfix(
                {
                    # 'train_acc': f'{train_metrics["accuracy"]:.4f}%',
                    'train_loss': f'{train_metrics["loss"]:.4f}',
                    'lr': lrs
                }
            )

        scheduler.step()

        if global_step > 0 and global_step % eval_period == 0:
            if main_process:
                print(f"\nLyla: Lyla here! We've reached step {global_step}.")
                print(
                    "Lyla: It's time for an evaluation update! Let's see how our model is doing..."
                )

            epoch_metrics = exp.evaluate(eval_loader)
    
            if world_size > 1:
                # Gather evaluation metrics from all processes
                gathered_metrics = [None] * world_size
                torch.distributed.all_gather_object(gathered_metrics, epoch_metrics)
                
                if main_process:
                    # Aggregate metrics across all processes
                    total_loss = sum(metric['loss'] for metric in gathered_metrics) / world_size
                    # total_accuracy = sum(metric['accuracy'] for metric in gathered_metrics) / world_size
                    print(
                        f'\nLyla: Evaluating the model on step {global_step}'
                        # f' -- Average Accuracy: {total_accuracy:.4f}%,'
                        f' Average Loss: {total_loss:.4f}.'
                    )
                    epoch_metrics = {'loss': total_loss}
            else:
                if main_process:
                    print(
                    f'\nLyla: Evaluating the model on step {global_step}'
                    # f' -- Accuracy: {epoch_metrics["accuracy"]:.2f}%,'
                    f' Loss: {epoch_metrics["loss"]:.2f}.'
                )

            if main_process:
                val_loss = epoch_metrics['loss']
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_step = global_step
                    best_model_metrics = epoch_metrics
                    eval_periods_without_improvement = 0
                    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                    checkpoint_filename = f'checkpoint-step{global_step}-{timestamp}.pt'
                    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
                    best_checkpoint = checkpoint_filename
                    
                    torch.save(spectral_ssm.module.state_Dict(), checkpoint_path)
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

    # Load the best model checkpoint
    best_checkpoint_path = os.path.join(checkpoint_dir, best_checkpoint)
    spectral_ssm.module.load_state_Dict(torch.load(best_checkpoint_path))

    # Print detailed information about the best model
    if main_process:
        # Print detailed information about the best model
        print("\nLyla: Training completed! Nice work. Here's the best model information:")
        print(f'    Best model at step {best_model_step}')
        print(f'    Best model validation loss: {best_val_loss:.4f}')
        # print(f'    Best model validation accuracy: {best_model_metrics["accuracy"]:.4f}%')
        print(f'    Best model checkpoint saved at: {best_checkpoint_path}')

        # Save the training details to a file
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        training_details = f'training_details_{timestamp}.txt'
        with open(training_details, 'w') as f:
            f.write(f'Training completed at: {datetime.now()}\n')
            f.write(f'Best model step: {best_model_step}\n')
            f.write(f'Best model validation loss: {best_val_loss:.4f}\n')
            # f.write(
            #     f'Best model validation accuracy: {best_model_metrics["accuracy"]:.4f}%\n'
            # )
            f.write(f'Best model checkpoint saved at: {best_checkpoint_path}\n')
        print(
            'Lyla: Congratulations on completing the training run!'
            f' Details are saved in {training_details}.'
            ' It was a pleasure assisting you. Until next time!'
        )

if __name__ == '__main__':
    main()
