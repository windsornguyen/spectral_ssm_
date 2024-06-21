# =============================================================================#
# Authors: Isabel Liu, Yagiz Devre, Windsor Nguyen
# File: train.py
# =============================================================================#

"""Training loop for Transformer sequence prediction."""

import argparse
from datetime import datetime
import os

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import sys
sys.path.insert(0, '/scratch/gpfs/mn4560/ssm')

from torch.nn import MSELoss
from losses.loss_ant import AntLoss
from losses.loss_cheetah import HalfCheetahLoss
from losses.loss_walker import Walker2DLoss
from utils.dataloader import get_dataloader, split_data
from utils import experiment as exp, optimizer as opt
from models.transformer.model import Transformer, TransformerConfigs
from utils.colors import Colors, colored_print
from utils.dist import set_seed, setup, cleanup


# TODO: Change this to be the correct command.
# To run the script: `python -m transformer.train_transformer` from root dir.
def main() -> None:
    torch.set_float32_matmul_precision('high')  # Enable CUDA TensorFloat-32

    # Process command line flags
    parser = argparse.ArgumentParser(
        description='Training script for sequence prediction'
    )
    parser.add_argument(
        '--controller',
        type=str,
        default='Ant-v1',
        choices=['Ant-v1', 'HalfCheetah-v1', 'Walker2D-v1'],
        help='Controller to use for the MuJoCo environment',
    )
    parser.add_argument(
        '--task',
        type=str,
        default='mujoco-v1',
        choices=[
            'mujoco-v1',  # Predict state trajectories, incl. controls as input
            'mujoco-v2',  # Predict state trajectories, w/o incl. controls as input
            'mujoco-v3',  # Predict state trajectories using a unified representation
        ],
        help='Task to train on',
    )
    parser.add_argument(
        '--della',
        type=bool,
        default=True,
        help='Training on the Princeton Della cluster',
        # NOTE: You MUST run with `torchrun` for this to work in the general setting.
    )

    args = parser.parse_args()

    controller = args.controller
    task = {
        'mujoco-v1': args.task == 'mujoco-v1',
        'mujoco-v2': args.task == 'mujoco-v2',
        'mujoco-v3': args.task == 'mujoco-v3',
    }

    # TODO: Is this needed if we re-write the dataloader?
    # torch.multiprocessing.set_start_method('spawn')

    # Defaults specific to the Princeton HPC cluster; modify to your own setup.
    device, local_rank, rank, world_size, num_workers, main_process = setup(args)
    set_seed(1337 + local_rank, main_process)

    if main_process:
        colored_print(
            "Lyla: Greetings! I'm Lyla, your friendly neighborhood AI training assistant.",
            Colors.OKBLUE,
        )

    # Prepare directories for training and plotting
    checkpoint_dir: str = 'checkpoints'
    if main_process:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        if not os.path.exists('plots/'):
            os.makedirs('plots/')

    # Data loader hyperparameters
    # TODO: Add accumulated gradients to this
    # TODO: Make data loader better
    # TODO: Add print statement reporting our batch size and accumulated batch size
    train_batch_size: int = 2 // world_size

    # Handle val batch sizes differently?
    val_batch_size: int = 2 // world_size

    # TODO: Make this relative path after directory structure is settled
    dataset = '/scratch/gpfs/mn4560/ssm/data/frames/HalfCheetah-v1/HalfCheetah-v1_ResNet-18.safetensors'
    train_data, val_data = split_data(load_file(dataset, device=device.type))

    # TODO Make it get_dataloader regardless of task (modify physics_data)
    train_loader = get_dataloader(
        data=train_data,
        task=args.task,
        batch_size=train_batch_size,
        num_workers=num_workers,
        device=device,
        preprocess=True,
        shuffle=True,
        pin_memory=True,
    )

    val_loader = get_dataloader(
        data=val_data,
        task=args.task,
        batch_size=val_batch_size,
        num_workers=num_workers,
        device=device,
        preprocess=True,
        shuffle=False,
        pin_memory=True,
    )

    # General training hyperparameters
    num_epochs: int = 1
    steps_per_epoch = len(train_loader)
    num_steps: int = steps_per_epoch * num_epochs
    warmup_steps: int = num_steps // 10
    eval_period: int = 10

    # General training variables
    patient_counter = 0
    best_val_loss = float('inf')
    best_model_step = 0
    best_checkpoint = None

    # Number of non-improving eval periods before early stopping
    patience: int = 5

    # Shared hyperparameters
    n_layers: int = 6
    scale: int = 4
    bias: bool = False
    dropout: float = 0.10
    use_dilated_attn: bool = False  # TODO: Finish up implementation
    if not task['mujoco-v3']:
        if controller == 'Ant-v1':
            loss_fn = AntLoss
        elif controller == 'HalfCheetah-v1':
            loss_fn = HalfCheetahLoss
        elif controller == 'Walker2D-v1':
            loss_fn = Walker2DLoss
        else:
            loss_fn = None
    else:
        loss_fn = MSELoss()

    # Task-specific hyperparameters
    if task['mujoco-v1']:
        n_embd: int = 24 if controller != 'Ant-v1' else 37
        n_head: int = 8 if controller != 'Ant-v1' else 1
        sl: int = 1_000
        configs = TransformerConfigs(
            n_layers=n_layers,
            n_embd=n_embd,
            n_head=n_head,
            sl=sl,
            scale=scale,
            bias=bias,
            dropout=dropout,
            use_dilated_attn=use_dilated_attn,
            loss_fn=loss_fn,
            controls={task: 'mujoco-v1', 'controller': controller},
        )

    elif task['mujoco-v2']:
        n_embd: int = 18 if controller != 'Ant-v1' else 29
        n_head: int = 9 if controller != 'Ant-v1' else 1
        sl: int = 1_000
        configs = TransformerConfigs(
            n_layers=n_layers,
            n_embd=n_embd,
            n_head=n_head,
            sl=sl,
            scale=scale,
            bias=bias,
            dropout=dropout,
            use_dilated_attn=use_dilated_attn,
            loss_fn=loss_fn,
            controls={task: 'mujoco-v2', 'controller': controller},
        )

    elif task['mujoco-v3']:
        RESNET_D_OUT: int = 512  # ResNet-18 output dim
        RESNET_FEATURE_SIZE: int = 1
        n_embd: int = RESNET_D_OUT * RESNET_FEATURE_SIZE**2
        n_head: int = 16
        sl: int = 300

        configs = TransformerConfigs(
            n_layers=n_layers,
            n_embd=n_embd,
            n_head=n_head,
            sl=sl,
            scale=scale,
            bias=bias,
            dropout=dropout,
            use_dilated_attn=use_dilated_attn,
            loss_fn=loss_fn,
            controls={'task': 'mujoco-v3', 'controller': controller},
        )

    model = Transformer(configs).to(device)
    # model = torch.compile(model)
    if world_size > 1:
        model = DDP(
            model, device_ids=[local_rank], gradient_as_bucket_view=True
        )
    transformer_model = model.module if world_size > 1 else model

    # Optimizer hyperparameters
    weight_decay: float = 1e-1
    lr: float = 6e-4

    optimizer, scheduler = opt.get_optimizer(
        transformer_model,
        num_steps=num_steps,
        warmup_steps=warmup_steps,
        learning_rate=lr,
        weight_decay=weight_decay,
    )

    # Prepare experiment variables
    experiment = exp.Experiment(
        model=transformer_model,
        task=task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    # Lists to store losses and metrics
    train_losses = []
    val_losses = []
    val_time_steps = []
    grad_norms = []

    if not task['mujoco-v3']:
        metric_losses = {
            'coordinate_loss': [],
            'orientation_loss': [],
            'angle_loss': [],
            'coordinate_velocity_loss': [],
            'angular_velocity_loss': [],
        }

    if main_process:
        msg = f"Lyla: We'll be training the Transformer model on the {args.task} task with {controller}."
        if world_size > 1:
            colored_print(
                f'{msg} {device} on rank {rank + 1}/{world_size}'
                f' utilizing {world_size} distributed processes.',
                Colors.OKCYAN
            )
        else:
            colored_print(f'{msg} {device} today.', Colors.OKCYAN)

    pbar = tqdm(
        range(num_epochs * steps_per_epoch),
        desc='Training',
        unit='step',
    )

    # Training loop
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_loader):
            last_step = step == num_steps - 1

            if task['mujoco-v3']:
                inputs, targets, file_name = batch
                # Process the batch here
            else:
                inputs, targets = batch

            train_metrics = experiment.step(inputs, targets)
            # TODO: Make this is correct for DDP
            if dist.is_initialized():
                # Gather metrics from all processes
                gathered_metrics = [None] * world_size
                dist.all_gather_object(gathered_metrics, train_metrics)

                # Aggregate metrics across all processes
                train_metrics = {
                    k: sum(d[k] for d in gathered_metrics) / world_size
                    for k in train_metrics.keys()
                }

            relative_step = step + (epoch * steps_per_epoch)
            train_losses.append(train_metrics['loss'])
            grad_norms.append(train_metrics['grad_norm'])
            if not task['mujoco-v3']:
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
                if not task['mujoco-v3']:
                    for metric in train_metrics:
                        if metric in metric_losses:
                            postfix_dict[metric] = train_metrics[metric]
                pbar.set_postfix(postfix_dict)
                pbar.update(1)

            # Periodically evaluate the model on validation set
            if step % eval_period == 0 or last_step:
                if main_process:
                    colored_print(f'\nStep: {relative_step}', Colors.BOLD)
                    colored_print(
                        f'\nTransformer - Train Loss After {relative_step} Steps: {train_losses[-1]:.4f}',
                        Colors.OKBLUE,
                    )

                val_metrics = experiment.evaluate(val_loader)

                # TODO: Make this is correct for DDP
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
                val_time_steps.append(relative_step)

                if main_process:
                    colored_print(
                        f'\nLyla: Evaluating the Transformer model on step {step} Loss: {val_metrics["loss"]:.2f}.',
                        Colors.OKCYAN,
                    )

                    val_loss = val_metrics['loss']
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_step = relative_step
                        patient_counter = 0
                        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

                        # TODO: Add task to this depending on how you do the --task flag
                        checkpoint_filename = f'transformer-{controller}-chkpt-step{relative_step}-{timestamp}.safetensors'
                        checkpoint_path = os.path.join(
                            checkpoint_dir, checkpoint_filename
                        )
                        best_checkpoint = checkpoint_filename

                        # TODO: Is this needed if we run with torchrun?
                        # TODO: Also check that it's correct if needed and what dist.barrier() does
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

                            # Save the data points to files
                            # TODO: Change these paths after directory structure is settled
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
                            if not task['mujoco-v3']:
                                for metric, losses in metric_losses.items():
                                    np.save(
                                        f'plots/transformer_{metric}.npy',
                                        losses,
                                    )

                            if dist.is_initialized():
                                dist.barrier()
                            return

    if main_process:
        if best_checkpoint:
            best_checkpoint_path = os.path.join(checkpoint_dir, best_checkpoint)

            # TODO: is is_initialized() needed if run with torchrun?
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
                "\nLyla: Here's the best model information for the Transformer model:"
            )
            print(f'    Best model at step {best_model_step}')
            print(f'    Best model validation loss: {best_val_loss:.4f}')
            print(f'    Best model checkpoint saved at: {best_checkpoint_path}')

            # Save the training details to a file
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            training_details = f'training_details_transformer_{timestamp}.txt'
            with open(training_details, 'w') as f:
                f.write(
                    f'Training completed for Transformer on {args.task} with {controller}at: {datetime.now()}\n'
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
            colored_print(
                '\nLyla: No best checkpoint found for the Transformer model. The model did not improve during training.',
                Colors.WARNING,
            )

        # Save the data points to files
        np.save('plots/transformer_train_losses.npy', train_losses)
        np.save('plots/transformer_val_losses.npy', val_losses)
        np.save('plots/transformer_val_time_steps.npy', val_time_steps)
        np.save('plots/transformer_grad_norms.npy', grad_norms)
        for metric, losses in metric_losses.items():
            np.save(f'plots/transformer_{metric}.npy', losses)

        colored_print(
            'Lyla: It was a pleasure assisting you. Until next time!',
            Colors.OKGREEN,
        )


if __name__ == '__main__':
    main()
    if dist.is_initialized():
        cleanup()
