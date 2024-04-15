# =============================================================================#
# Authors: Windsor Nguyen, Dwaipayan Saha
# File: example.py
# =============================================================================#

"""Example training loop."""

import argparse
import os
import torch
import torch.distributed as dist
from datetime import datetime
from spectral_ssm import cifar10
from spectral_ssm import experiment
from spectral_ssm import model
from spectral_ssm import optimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


def setup_distributed_env(local_rank: int) -> tuple[torch.device, int, int]:
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


# To run the script: `torchrun --nproc_per_node=1 example.py`
def main() -> None:
    parser = argparse.ArgumentParser(description='Distributed Training Setup')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    device, rank, world_size = setup_distributed_env(args.local_rank)

    if rank == 0:
        print("Lyla: Greetings! I'm Lyla, your friendly neighborhood AI training assistant.")

    # Hyperparameters
    train_batch_size: int = 48 // world_size # scale batch size for distributed training
    eval_batch_size: int = 48 // world_size  # scale batch size for distributed training
    num_steps: int = 3_500
    eval_period: int = 35
    warmup_steps: int = 360
    learning_rate: float = 5e-4
    weight_decay: float = 1e-1
    m_y_learning_rate: float = 5e-5
    m_y_weight_decay: float = 0
    patience: int = 300
    checkpoint_dir: str = f'checkpoints/rank{rank}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Define the model
    spectral_ssm = model.Architecture(
        d_model=256,
        d_target=10,
        num_layers=6,
        dropout=0.1,
        input_len=32 * 32,
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
        find_unused_parameters=True
    )

    opt, scheduler = optimizer.get_optimizer(
        spectral_ssm.module,  # Access the original module when wrapped in DDP
        num_steps=num_steps,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        m_y_learning_rate=m_y_learning_rate,
        m_y_weight_decay=m_y_weight_decay,
    )
    exp = experiment.Experiment(model=spectral_ssm, optimizer=opt, device=device)
    msg = "Lyla: We'll be training with"

    if rank == 0:
        if world_size > 1:
            print(
                f'{msg} {device} on rank {rank + 1}/{world_size}, '
                f'utilizing {world_size} distributed processes.'
            )
        else:
            print(f'{msg} {device} today.')

    train_dataset = cifar10.get_dataset('train')
    eval_dataset = cifar10.get_dataset('test')

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        sampler=DistributedSampler(train_dataset),
    )
    eval_loader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=eval_batch_size,
        sampler=DistributedSampler(eval_dataset),
    )

    if rank == 0:
        print(
            "Lyla: All set! Everything's loaded up and ready to go. "
            'May the compute Gods be by our sides...'
        )

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_step = 0
    best_model_metrics = None

    torch.autograd.set_detect_anomaly(True)

    pbar = tqdm(range(num_steps), desc='Training Progress', unit='step') if rank == 0 else range(num_steps)
    for global_step in pbar:
        inputs, targets = next(iter(train_loader))
        train_metrics = exp.step(inputs, targets)

        if rank == 0:
            pbar.set_postfix(
                {
                    'train_acc': f'{train_metrics["accuracy"]:.2f}%',
                    'train_loss': f'{train_metrics["loss"]:.2f}',
                    'lr': scheduler.get_last_lr()[0],
                }
            )

        scheduler.step()

        if global_step > 0 and global_step % eval_period == 0:
            if rank == 0:
                print(f"\nLyla: Lyla here! We've reached step {global_step}.")
                print(
                    "Lyla: It's time for an evaluation update! Let's see how our model is doing..."
                )
            epoch_metrics = exp.evaluate(eval_loader)
            
            if rank == 0:
                print(
                    f'\nLyla: Evaluating the model on step {global_step}'
                    f' -- Accuracy: {epoch_metrics["accuracy"]:.2f}%,'
                    f' Loss: {epoch_metrics["loss"]:.2f}.'
                )
            val_loss = epoch_metrics['loss']
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_step = global_step
                best_model_metrics = epoch_metrics
                epochs_without_improvement = 0
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                checkpoint_path = os.path.join(
                    checkpoint_dir, f'checkpoint-step{global_step}-{timestamp}.pt'
                )
                
                if rank == 0:
                    torch.save(spectral_ssm.module.state_dict(), checkpoint_path)
                    print(
                        f'Lyla: Wow! We have a new personal best at step {global_step}.'
                        f' The validation loss improved to: {val_loss:.2f}!'
                        f' Checkpoint saved as {checkpoint_path}'
                    )
            else:
                epochs_without_improvement += 1
                if rank == 0:
                    print(
                        f'Lyla: No improvement in validation loss for epoch '
                        f'{epochs_without_improvement} epochs. '
                        f'Current best loss: {best_val_loss:.2f}.'
                    )

            if epochs_without_improvement >= patience:
                if rank == 0:
                    print(
                        f'Lyla: We have reached the patience limit of {patience} '
                        f'epochs without improvement. Stopping the training early '
                        f'at step {global_step}...'
                    )
                break

    # Load the best model checkpoint
    best_checkpoint_path = os.path.join(
        checkpoint_dir, f'checkpoint-{best_model_step}.pt'
    )
    spectral_ssm.module.load_state_dict(torch.load(best_checkpoint_path))

    # Print detailed information about the best model
    if rank == 0:
        print("\nLyla: Training completed! Nice work. Here's the best model information:")
        print(f'    Best model at step {best_model_step}')
        print(f'    Best model validation loss: {best_val_loss:.2f}')
        print(f'    Best model validation accuracy: {best_model_metrics["accuracy"]:.2f}%')
        print(f'    Best model checkpoint saved at: {best_checkpoint_path}')

    # Save the training details to a file
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    training_details = f'training_details_{timestamp}.txt'
    with open(training_details, 'w') as f:
        f.write(f'Training completed at: {datetime.now()}\n')
        f.write(f'Best model step: {best_model_step}\n')
        f.write(f'Best model validation loss: {best_val_loss:.2f}\n')
        f.write(
            f'Best model validation accuracy: {best_model_metrics["accuracy"]:.2f}%\n'
        )
        f.write(f'Best model checkpoint saved at: {best_checkpoint_path}\n')
    if rank == 0:
        print(
            'Lyla: Congratulations on completing the training run!'
            f' Details are saved in {training_details}.'
            ' It was a pleasure assisting you. Until next time!'
        )

if __name__ == '__main__':
    main()
