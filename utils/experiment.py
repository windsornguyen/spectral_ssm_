# ==============================================================================#
# Authors: Windsor Nguyen, Dwaipayan Saha
# File: experiment.py
# ==============================================================================#

"""Utilities for running an experiment."""

import inspect
import math

import torch
import torch.nn as nn
import torch.distributed as dist

from time import time
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.optim import AdamW
from utils.colors import Colors, colored_print


# TODO: Check that the parameter names for m_y adjustments are correct for STU.
# TODO: Condition the m_y stuff on the model training on STU model
class Experiment:
    """
    Initializes and maintains the experiment state.
    """

    def __init__(
        self,
        model: nn.Module,
        task: dict[str, bool],
        loss_fn: nn.Module,
        optimizer_settings: tuple[int, int, float, float],
        training_stu: bool = False,
        bsz: int = None,
        sl: int = None,
        grad_accum_steps: int = None,
        world_size: int = 1,
        main_process: bool = False,
        device: torch.device = None,
    ) -> None:
        """
        Initialize an experiment.

        Args:
            model (nn.Module): A PyTorch model.
            optimizer (torch.optim.Optimizer): A PyTorch optimizer.
            device (torch.device): The device to run the model on.
        """
        self.model = model
        self.device = device
        self.task = task
        self.loss_fn = loss_fn
        (
            self.warmup_steps,
            self.num_steps,
            self.max_lr,
            self.min_lr,
            self.weight_decay,
        ) = optimizer_settings

        # Additional information to process
        self.bsz = bsz
        self.sl = sl
        self.grad_accum_steps = grad_accum_steps
        self.main_process = main_process
        self.world_size = world_size

        # If training STU
        if training_stu:
            self.m_y_learning_rate = 5e-5
            self.m_y_weight_decay = 0

        self.optimizer = self.get_optimizer(
            self.weight_decay, self.max_lr, self.device.type
        )

        self.model.to(self.device)

    def get_optimizer(self, weight_decay, learning_rate, device_type):
        param_groups = []
        m_y_params = []
        default_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name.startswith("m_y"):
                    m_y_params.append(param)
                else:
                    default_params.append(param)

        if m_y_params:
            param_groups.extend(
                [
                    {
                        "params": default_params,
                        "lr": self.max_lr,
                        "weight_decay": self.weight_decay,
                    },
                    {
                        "params": m_y_params,
                        "lr": self.m_y_learning_rate,
                        "weight_decay": self.m_y_weight_decay,
                    },
                ]
            )
        else:
            decay_params = [p for p in default_params if p.dim() >= 2]
            nodecay_params = [p for p in default_params if p.dim() < 2]
            param_groups.extend(
                [
                    {
                        "params": decay_params,
                        "lr": self.max_lr,
                        "weight_decay": self.weight_decay,
                    },
                    {
                        "params": nodecay_params,
                        "lr": self.max_lr,
                        "weight_decay": 0.0,
                    },
                ]
            )

        if hasattr(self, "master_process") and self.master_process:
            for i, group in enumerate(param_groups):
                print(
                    f'Optimizer | Group {i}: {len(group["params"])} tensors, '
                    f'{sum(p.numel() for p in group["params"]):,} parameters'
                )

        fused_available = "fused" in inspect.signature(AdamW).parameters
        use_fused = fused_available and self.device.type == "cuda"

        # TODO What is this the hasattr for? The Experiment class or what...?
        # If so, just use self.main_process lol
        if hasattr(self, "master_process") and self.master_process:
            print(f"Optimizer | Using fused AdamW?: {use_fused}")

        return AdamW(
            param_groups,
            lr=self.max_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused,
        )

    def get_lr(
        self,
        it,
        warmup_steps,
        num_steps,
        max_lr,
        min_lr,
    ):
        """
        Custom learning rate scheduler: linear warmup and cosine decay.
        """
        # 1. Linear warmup for warmup_steps steps
        if it < self.warmup_steps:
            return self.max_lr * (it + 1) / self.warmup_steps

        # 2. If it > lr_decay_iters, return min learning rate
        if it > self.num_steps:
            return self.min_lr

        # 3. If in between, cosine decay to down to min learning rate
        decay_ratio = (it - self.warmup_steps) / (self.num_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

    def step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        step: int,
    ) -> dict[str, float]:
        """
        Perform a single training step.

        Args:
            inputs (torch.Tensor): A batch of input data.
            targets (torch.Tensor): A batch of target labels.

        Returns:
            dict[str, float]: A dictionary of metrics for the training step.
        """
        self.optimizer.zero_grad()
        metrics = {}
        loss_accum = 0.0
        t0 = time()

        for micro_step in range(self.grad_accum_steps):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            if self.world_size > 1:
                self.model.require_backward_grad_sync = (
                    micro_step == self.grad_accum_steps - 1
                )

            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                preds, loss_info = self.model(inputs, targets)

            print(f"loss_info is definitely a tuple: {isinstance(loss_info, tuple)}")
            if isinstance(loss_info, tuple):
                loss, step_metrics = loss_info
                for k, v in step_metrics.items():
                    metrics[k] = metrics.get(k, 0) + v / self.grad_accum_steps
            else:
                loss = loss_info

            loss /= self.grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        if self.world_size > 1:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.SUM)
            for k in metrics:
                dist.all_reduce(
                    torch.tensor(metrics[k], device=self.device), op=dist.ReduceOp.SUM
                )
                metrics[k] = metrics[k].item() / self.world_size

        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        # Step the learning rate scheduler forward
        # TODO: Ensure that the learning rates for each group are being adjusted accordingly.
        for param_group in self.optimizer.param_groups:
            if "m_y" in param_group:
                param_group["lr"] = self.get_lr(
                    step,
                    self.warmup_steps,
                    self.num_steps,
                    self.m_y_learning_rate,
                    self.min_lr,
                )
            else:
                param_group["lr"] = self.get_lr(
                    step, self.warmup_steps, self.num_steps, self.max_lr, self.min_lr
                )

        # Step the optimizer forward
        self.optimizer.step()

        # Time how long this training step took
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time()
        dt = t1 - t0

        toks_processed = self.bsz * self.sl * self.grad_accum_steps * self.world_size
        toks_per_sec = toks_processed / dt

        # Add the loss and other computed values to the metrics
        metrics["loss"] = loss_accum
        metrics["lr"] = self.optimizer.param_groups[0]["lr"]  # TODO: Why [0]?
        metrics["norm"] = norm.item()
        metrics["dt"] = dt
        metrics["toks_per_sec"] = toks_per_sec

        if self.main_process:
            # Base metrics string
            metrics_str = f"step {step:5d} | loss: {metrics['loss']:.6f} | lr: {metrics['lr']:.4e} | norm: {metrics['norm']:.4f} | dt: {metrics['dt']*1000:.2f}ms | tok/sec: {metrics['toks_per_sec']:.2f}"

            # Add task-specific metrics if available
            if "coordinate_loss" in metrics:
                metrics_str += f" | coord: {metrics['coordinate_loss']:.4f}"
            if "orientation_loss" in metrics:
                metrics_str += f" | orient: {metrics['orientation_loss']:.4f}"
            if "angle_loss" in metrics:
                metrics_str += f" | angle: {metrics['angle_loss']:.4f}"
            if "coordinate_velocity_loss" in metrics:
                metrics_str += (
                    f" | coord_vel: {metrics['coordinate_velocity_loss']:.4f}"
                )
            if "angular_velocity_loss" in metrics:
                metrics_str += f" | ang_vel: {metrics['angular_velocity_loss']:.4f}"

            print(metrics_str)

        return metrics

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate the model over an entire dataset.

        Args:
            dataloader (DataLoader):
              A DataLoader providing batches of data for evaluation.

        Returns:
            Dict[str, float]:
              A Dictionary of aggregated metrics over the dataset.
        """

        # TODO: Bring distributed loss aggregation into this function.
        self.model.eval()
        total_loss = 0.0
        metrics = {}
        num_batches = 0

        with (
            torch.no_grad(),
            tqdm(total=len(dataloader), desc="Evaluating model") as pbar,
        ):
            # TODO: might have to conditional this with mujoco-v3 flag too
            for inputs, targets, filename in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    preds, loss_info = self.model(inputs, targets)

                # TODO: Make this prettier and not so ugly.
                if isinstance(loss_info, tuple):
                    if len(loss_info) == 1:
                        loss = loss_info[0]
                        step_metrics = {}
                    elif len(loss_info) == 2:
                        loss, step_metrics = loss_info
                    else:
                        raise ValueError(f"Unexpected loss_info structure: {loss_info}")
                    for k, v in step_metrics.items():
                        metrics[k] = metrics.get(k, 0) + v
                else:
                    loss = loss_info
                    step_metrics = {}

                total_loss += loss.item() if loss is not None else 0.0
                num_batches += 1
                pbar.update(1)
            pbar.close()

        # Aggregate metrics across all processes
        if self.world_size > 1:
            for k in metrics:
                dist.all_reduce(
                    torch.tensor(metrics[k], device=self.device), op=dist.ReduceOp.SUM
                )
                metrics[k] = metrics[k].item() / self.world_size

            dist.all_reduce(
                torch.tensor(total_loss, device=self.device), op=dist.ReduceOp.SUM
            )
            dist.all_reduce(
                torch.tensor(num_batches, device=self.device), op=dist.ReduceOp.SUM
            )

        avg_loss = total_loss / num_batches
        metrics["loss"] = avg_loss

        if self.main_process:
            metrics_str = f"Evaluation | loss: {avg_loss:.6f}"
            for k, v in metrics.items():
                if k != "loss":
                    metrics_str += f" | {k}: {v:.4f}"
            print(metrics_str)

        self.model.train()
        return metrics
