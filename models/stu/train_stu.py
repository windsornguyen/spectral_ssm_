# =============================================================================#
# Authors: Isabel Liu, Yagiz Devre, Windsor Nguyen
# File: train.py
# =============================================================================#

"""Training loop for STU sequence prediction."""

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

from torch.nn import MSELoss
from losses.loss_ant import AntLoss
from losses.loss_cheetah import HalfCheetahLoss
from losses.loss_walker import Walker2DLoss
from utils.dataloader import get_dataloader, split_data
from utils import experiment as exp, optimizer as opt
from models.stu.model import SSSM, SSSMConfigs
from utils.colors import Colors, colored_print
from utils.dist import set_seed, setup, cleanup


# TODO: Change this to be the correct command.
# To run the script: `torchrun --nproc_per_node=1 train_stu.py`
def main() -> None:
    torch.set_float32_matmul_precision("high")  # Enable CUDA TensorFloat-32

    # Process command line flags
    parser = argparse.ArgumentParser(
        description="Training script for sequence prediction"
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="Ant-v1",
        choices=["Ant-v1", "HalfCheetah-v1", "Walker2D-v1"],
        help="Controller to use for the MuJoCo environment",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="mujoco-v1",
        choices=[
            "mujoco-v1",  # Predict state trajectories, incl. controls as input
            "mujoco-v2",  # Predict state trajectories, w/o incl. controls as input
            "mujoco-v3",  # Predict state trajectories using a unified representation
        ],
        help="Task to train on",
    )
    parser.add_argument(
        "--della",
        type=bool,
        default=True,
        help="Training on the Princeton Della cluster",
        # NOTE: You MUST run with `torchrun` for this to work in the general setting.
    )

    args = parser.parse_args()

    controller = args.controller
    task = {
        "mujoco-v1": args.task == "mujoco-v1",
        "mujoco-v2": args.task == "mujoco-v2",
        "mujoco-v3": args.task == "mujoco-v3",
    }

    # TODO: Is this needed if we re-write the dataloader?
    torch.multiprocessing.set_start_method("spawn")

    # Defaults specific to the Princeton HPC cluster; modify to your own setup.
    device, local_rank, rank, world_size, num_workers, main_process = setup(args)

    if main_process:
        colored_print(
            "Lyla: Greetings! I'm Lyla, your friendly neighborhood AI training assistant.",
            Colors.OKBLUE,
        )

    # Prepare directories for training and plotting
    checkpoint_dir: str = "checkpoints"
    if main_process:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        if not os.path.exists("plots/"):
            os.makedirs("plots/")

    # Shared hyperparameters
    n_layers: int = 6
    scale: int = 4
    bias: bool = False
    dropout: float = 0.10
    num_eigh: int = 24
    auto_reg_k_u: int = 3
    auto_reg_k_y: int = 32  # TODO: Maybe change this back to 2 (it was 32 in the paper for CIFAR-10 and Pathfinder?)
    learnable_m_y: bool = (True,)
    if not task["mujoco-v3"]:
        if controller == "Ant-v1":
            loss_fn = AntLoss
        elif controller == "HalfCheetah-v1":
            loss_fn = HalfCheetahLoss
        elif controller == "Walker2D-v1":
            loss_fn = Walker2DLoss
        else:
            loss_fn = None
    else:
        loss_fn = MSELoss()

    # Task-specific hyperparameters
    if task["mujoco-v1"]:
        d_model: int = 24 if controller != "Ant-v1" else 37
        d_out: int = 18 if controller != "Ant-v1" else 29
        sl: int = 1_000

        configs = SSSMConfigs(
            n_layers=n_layers,
            d_model=d_model,
            d_out=d_out,
            sl=sl,
            scale=scale,
            bias=bias,
            dropout=dropout,
            num_eigh=num_eigh,
            auto_reg_k_u=auto_reg_k_u,
            auto_reg_k_y=auto_reg_k_y,
            learnable_m_y=learnable_m_y,
            loss_fn=loss_fn,
            controls={"task": "mujoco-v1", "controller": controller},
        )

    elif task["mujoco-v2"]:
        d_model: int = 18 if controller != "Ant-v1" else 29
        d_out = d_model
        sl: int = 1_000
        configs = SSSMConfigs(
            n_layers=n_layers,
            d_model=d_model,
            d_out=d_out,
            sl=sl,
            scale=scale,
            bias=bias,
            dropout=dropout,
            num_eigh=num_eigh,
            auto_reg_k_u=auto_reg_k_u,
            auto_reg_k_y=auto_reg_k_y,
            learnable_m_y=learnable_m_y,
            loss_fn=loss_fn,
            controls={"task": "mujoco-v2", "controller": controller},
        )

    elif task["mujoco-v3"]:
        RESNET_D_OUT: int = 512  # ResNet-18 output dim
        RESNET_FEATURE_SIZE: int = 1
        d_out: int = RESNET_D_OUT * RESNET_FEATURE_SIZE**2
        d_model = d_out
        sl: int = 300

        configs = SSSMConfigs(
            n_layers=n_layers,
            d_model=d_model,
            d_out=d_out,
            sl=sl,
            scale=scale,
            bias=bias,
            dropout=dropout,
            num_eigh=num_eigh,
            auto_reg_k_u=auto_reg_k_u,
            auto_reg_k_y=auto_reg_k_y,
            learnable_m_y=learnable_m_y,
            loss_fn=loss_fn,
            controls={"task": "mujoco-v3", "controller": controller},
        )

    model = SSSM(configs).to(device)
    model = torch.compile(model)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)
    stu_model = model.module if world_size > 1 else model

    # Data loader hyperparameters
    total_bsz: int = 16
    bsz: int = 4  # Micro batch size
    grad_accum_steps = total_bsz // (bsz * sl * world_size)

    if main_process:
        print(f"Total (desired) batch size: {total_bsz}")
        print(f"=> Gradient accumulation steps: {grad_accum_steps}")

    # TODO: Put in v2 data (no controls)
    dataset = f"data/mujoco-v3/{controller}/{controller}_ResNet-18.safetensors"

    # TODO: Should we load data straight to GPU?
    # Things to consider:
    # 1. Can't use multiple workers if straight to GPU (but safetensors is fast anyway)
    # 2. Will it all fit on the GPU? Is to CPU better?
    train_data, val_data = split_data(load_file(dataset))

    train_loader = get_dataloader(
        data=train_data,
        task=args.task,
        batch_size=bsz,
        num_workers=num_workers,
        preprocess=True,
        shuffle=True,
        pin_memory=True,
        distributed=world_size > 1,
        rank=local_rank,
        world_size=world_size,
        prefetch_factor=2,
        persistent_workers=True,
    )

    val_loader = get_dataloader(
        data=val_data,
        task=args.task,
        batch_size=bsz,
        num_workers=num_workers,
        preprocess=True,
        shuffle=False,
        pin_memory=True,
        distributed=world_size > 1,
        rank=local_rank,
        world_size=world_size,
        prefetch_factor=2,
        persistent_workers=True,
    )

    # General training hyperparameters
    training_stu = True
    num_epochs: int = 1
    steps_per_epoch = len(train_loader)
    num_steps: int = steps_per_epoch * num_epochs
    warmup_steps: int = num_steps // 10
    eval_period: int = 10

    # General training variables
    patient_counter = 0
    best_val_loss = float("inf")
    best_model_step = 0
    best_checkpoint = None

    # Number of non-improving eval periods before early stopping
    patience: int = 5

    # Optimizer hyperparameters
    weight_decay: float = 1e-1
    max_lr: float = 6e-4
    min_lr: float = max_lr * 0.1
    optimizer_settings = (warmup_steps, num_steps, max_lr, min_lr, weight_decay)

    training_run = exp.Experiment(
        model=stu_model,
        task=task,
        loss_fn=loss_fn,
        optimizer_settings=optimizer_settings,
        training_stu=training_stu,
        bsz=bsz,
        sl=sl,
        grad_accum_steps=grad_accum_steps,
        world_size=world_size,
        main_process=main_process,
        device=device,
    )

    # Lists to store losses and metrics
    train_losses = []
    val_losses = []
    val_time_steps = []
    grad_norms = []

    if not task["mujoco-v3"]:
        metric_losses = {
            "coordinate_loss": [],
            "orientation_loss": [],
            "angle_loss": [],
            "coordinate_velocity_loss": [],
            "angular_velocity_loss": [],
        }

    if main_process:
        msg = f"Lyla: We'll be training the SSSM model on the {args.task} task with {controller}."
        if world_size > 1:
            colored_print(
                f"{msg} {device} on rank {rank + 1}/{world_size}"
                f" utilizing {world_size} distributed processes.",
                Colors.OKCYAN,
            )
        else:
            colored_print(f"{msg} {device} today.", Colors.OKCYAN)

    pbar = tqdm(
        range(num_epochs * steps_per_epoch),
        desc="Training",
        unit="step",
    )

    # Training loop
    stu_model.train()
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_loader):
            last_step = step == num_steps - 1

            if task["mujoco-v3"]:
                inputs, targets, file_name = batch
            else:
                inputs, targets = batch

            relative_step = epoch * steps_per_epoch + step

            # Periodically evaluate the model on validation set
            if step % eval_period == 0 or last_step:
                if main_process:
                    colored_print(
                        f"\nLyla: Evaluating the SSSM model on step {relative_step}.",
                        Colors.OKCYAN,
                    )

                val_metrics = training_run.evaluate(val_loader)
                val_losses.append(val_metrics["loss"])
                val_time_steps.append(relative_step)
                # if master_process:
                # # TODO: Use pbar write or print statement?
                # # pbar.write(f"Validation loss: {val_loss_accum.item():.4f}")
                # print(f'Validation loss: {val_loss_accum.item():.4f}')
                # with open(log_file, 'a') as f:
                #     f.write(f'{step} val {val_loss_accum.item():.4f}\n')
                # if step > 0 and (step % (5000 // scale) == 0 or last_step):
                #     if val_loss_accum.item() < best_val_loss:
                #         best_val_loss = val_loss_accum.item()
                #         # optionally write model checkpoints
                #         checkpoint_path = os.path.join(
                #             # TODO: Switch to safetensors?
                #             log_dir, f'model_{step:05d}.pt'
                #         )
                #         checkpoint = {
                #             'model': raw_model.state_dict(),
                #             'config': raw_model.config,
                #             'optimizer': optimizer.state_dict(),
                #             'step': step,
                #             'val_loss': val_loss_accum.item(),
                #             'rng_state_pytorch': torch.get_rng_state(),
                #             'rng_state_cuda': torch.cuda.get_rng_state(),
                #             'rng_state_numpy': np.random.get_state(),
                #         }
                #         print(
                #             f'Validation loss improved at step {step}! Saving the model to {checkpoint_path}.'
                #         )
                #         torch.save(checkpoint, checkpoint_path)

                if main_process:
                    colored_print(
                        f'\nValidation Loss: {val_metrics["loss"]:.2f}.',
                        Colors.OKCYAN,
                    )

                    val_loss = val_metrics["loss"]
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_step = relative_step
                        patient_counter = 0
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

                        # TODO: Add task to this depending on how you do the --task flag
                        checkpoint_filename = f"sssm-{controller}-chkpt-step{relative_step}-{timestamp}.safetensors"
                        checkpoint_path = os.path.join(
                            checkpoint_dir, checkpoint_filename
                        )
                        best_checkpoint = checkpoint_filename

                        # TODO: Is this needed if we run with torchrun?
                        # TODO: Also check that it's correct if needed and what dist.barrier() does
                        # TODO: TBH, not sure what this is doing.
                        if dist.is_initialized():
                            # Save the model on the main process and broadcast it to all processes
                            if main_process:
                                save_file(
                                    stu_model.module.state_dict(),
                                    checkpoint_path,
                                )
                            dist.barrier()
                            map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
                            stu_model.load_state_dict(
                                load_file(checkpoint_path, device=map_location)
                            )
                        else:
                            save_file(
                                stu_model.state_dict(),
                                checkpoint_path,
                            )
                        colored_print(
                            f"Lyla: Wow! We have a new personal best for the SSSM model at step {step}. The validation loss improved to: {val_loss:.4f}! Checkpoint saved as {checkpoint_path}",
                            Colors.OKGREEN,
                        )
                    else:
                        patient_counter += 1
                        colored_print(
                            f"Lyla: No improvement in validation loss for the SSSM model for {patient_counter} eval periods. Current best loss: {best_val_loss:.4f}.",
                            Colors.WARNING,
                        )

                        if patient_counter >= patience:
                            colored_print(
                                f"Lyla: We have reached the patience limit of {patience} for the SSSM model. Stopping the training early at step {step}...",
                                Colors.FAIL,
                            )

                            # Save the data points to files
                            # TODO: Change these paths after directory structure is settled
                            np.save(
                                f"plots/sssm/sssm-{args.task}-{controller}_train_losses.npy",
                                train_losses,
                            )
                            np.save(
                                f"plots/sssm/sssm-{args.task}-{controller}_val_losses.npy",
                                val_losses,
                            )
                            np.save(
                                f"plots/sssm/sssm-{args.task}-{controller}_val_time_steps.npy",
                                val_time_steps,
                            )
                            np.save(
                                f"plots/sssm/sssm-{args.task}-{controller}_grad_norms.npy",
                                grad_norms,
                            )
                            if not task["mujoco-v3"]:
                                for metric, losses in metric_losses.items():
                                    np.save(
                                        f"plots/sssm/sssm-{args.task}-{controller}_{metric}.npy",
                                        losses,
                                    )

                            if dist.is_initialized():
                                dist.barrier()
                            return

            train_metrics = training_run.step(inputs, targets, relative_step)

            # TODO: Report train loss, val loss ,grad norm, lr,
            # and the group losses for mujoco-v1, v2 in step(), not here.

            pbar.update(1)
    pbar.close()

    if main_process:
        if best_checkpoint:
            best_checkpoint_path = os.path.join(checkpoint_dir, best_checkpoint)

            # TODO: is is_initialized() needed if run with torchrun?
            if dist.is_initialized():
                # Load the best checkpoint on the main process and broadcast it to all processes
                if main_process:
                    with safe_open(
                        best_checkpoint_path, framework="pt", device=rank
                    ) as f:
                        state_dict = {k: f.get_tensor(k) for k in f.keys()}
                        stu_model.load_state_dict(state_dict)
                dist.barrier()
            else:
                with safe_open(best_checkpoint_path, framework="pt", device="cpu") as f:
                    state_dict = {k: f.get_tensor(k) for k in f.keys()}
                    stu_model.load_state_dict(state_dict)

            print("\nLyla: Here's the best model information for the SSSM model:")
            print(f"    Best model at step {best_model_step}")
            print(f"    Best model validation loss: {best_val_loss:.4f}")
            print(f"    Best model checkpoint saved at: {best_checkpoint_path}")

            # Save the training details to a file
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            training_details = f"training_details_sssm_{timestamp}.txt"
            with open(training_details, "w") as f:
                f.write(
                    f"Training completed for SSSM on {args.task} with {controller}at: {datetime.now()}\n"
                )
                f.write(f"Best model step: {best_model_step}\n")
                f.write(f"Best model validation loss: {best_val_loss:.4f}\n")
                f.write(f"Best model checkpoint saved at: {best_checkpoint_path}\n")
            print(
                f"Lyla: Congratulations on completing the training run for the SSSM model! Details are saved in {training_details}."
            )
        else:
            colored_print(
                "\nLyla: No best checkpoint found for the SSSM model. The model did not improve during training.",
                Colors.WARNING,
            )

        # Save the data points to files
        np.save("plots/sssm_train_losses.npy", train_losses)
        np.save("plots/sssm_val_losses.npy", val_losses)
        np.save("plots/sssm_val_time_steps.npy", val_time_steps)
        np.save("plots/sssm_grad_norms.npy", grad_norms)
        if not task["mujoco-v3"]:
            for metric, losses in metric_losses.items():
                np.save(f"plots/sssm_{metric}.npy", losses)

        colored_print(
            "Lyla: It was a pleasure assisting you. Until next time!",
            Colors.OKGREEN,
        )


if __name__ == "__main__":
    main()
    if dist.is_initialized():
        cleanup()
