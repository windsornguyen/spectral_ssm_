# =============================================================================#
# Authors: Windsor Nguyen, Dwaipayan Saha
# File: example.py
# =============================================================================#

"""Example training loop."""

import argparse
import torch
import torch.distributed as dist
from spectral_ssm import torch_cifar10
from spectral_ssm import torch_experiment
from spectral_ssm import torch_model
from spectral_ssm import torch_optimizer
from tqdm import tqdm


def setup_distributed_env(local_rank: int):
    """Sets up the distributed training environment for both GPU and CPU."""
    if torch.cuda.is_available():
        # Setup for GPU-based distributed training
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        # Setup for CPU-based distributed training
        device = torch.device('cpu')

    # Initialize the process group for both GPU and CPU
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend=backend, init_method='env://')

    # Regardless of GPU or CPU, obtain rank and world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    return device, rank, world_size


def main():
    # Distributed training:
    # torchrun --nproc_per_node=1 torch_example.py
    parser = argparse.ArgumentParser(description='Distributed Training Setup')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    device, rank, world_size = setup_distributed_env(args.local_rank)
    print(f'Running on rank {rank}/{world_size}, device: {device}')

    # Hyperparameters
    train_batch_size = 16
    eval_batch_size = 16
    num_steps = 10
    eval_period = 5
    warmup_steps = 1
    learning_rate = 1e-3
    weight_decay = 1e-4
    m_y_learning_rate = 1e-4
    m_y_weight_decay = 0
    patience = 5
    checkpoint_path = 'checkpoint.pt'

    # Define the model
    spectral_ssm = torch_model.Architecture(
        d_model=1024,
        d_target=10,
        num_layers=6,
        dropout=0.1,
        input_len=1024,
        num_eigh=24,
        auto_reg_k_u=3,
        auto_reg_k_y=2,
        learnable_m_y=True,
    ).to(device)

    opt, scheduler = torch_optimizer.get_optimizer(
        spectral_ssm,
        num_steps=num_steps,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        m_y_learning_rate=m_y_learning_rate,
        m_y_weight_decay=m_y_weight_decay,
    )

    exp = torch_experiment.Experiment(model=spectral_ssm, optimizer=opt)

    training_loader = torch_cifar10.get_dataset('train', batch_size=train_batch_size)
    eval_loader = torch_cifar10.get_dataset('test', batch_size=eval_batch_size)

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    pbar = tqdm(total=num_steps, desc='Training Progress')
    for global_step, (inputs, targets) in enumerate(training_loader, 1):
        # Right after fetching a batch
        print(f'Batch shapes - Inputs: {inputs.shape}, Targets: {targets.shape}')

        # Before calling exp.step or exp.evaluate
        print('Sending batch to experiment step/evaluate')
        metrics = exp.step(inputs, targets)

        pbar.update(1)
        pbar.set_description(
            f'Step {global_step} - train/acc: {metrics["accuracy"]:.2f} train/loss: {metrics["loss"]:.2f}'
        )
        scheduler.step()

        if global_step > 0 and global_step % eval_period == 0:
            epoch_metrics = exp.evaluate(eval_loader)
            print(
                f"Eval {global_step}: acc: {epoch_metrics['accuracy']:.2f}, loss: {epoch_metrics['loss']:.2f}"
            )

            val_loss = epoch_metrics['loss']
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(spectral_ssm.state_dict(), checkpoint_path)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f'Early stopping at step {global_step}')
                break

        if global_step >= num_steps:
            break

    # Load the best model checkpoint
    spectral_ssm.load_state_dict(torch.load(checkpoint_path))
    print('Training completed. Best model loaded.')


if __name__ == '__main__':
    main()
