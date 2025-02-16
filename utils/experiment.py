# ==============================================================================#
# Authors: Windsor Nguyen
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


class Experiment:
    """
    Initializes and maintains the experiment state.
    """

    def __init__(
        self,
        model: nn.Module,
        task: dict[str, bool],
        loss_fn: nn.Module,
        bsz: int,
        sl: int,
        optimizer_settings: tuple[int, int, float, float],
        training_stu: bool = False,
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
            self.betas,
            self.eps,
            self.weight_decay,
            self.use_amsgrad,
        ) = optimizer_settings

        # Additional information to process
        self.bsz = bsz
        self.sl = sl
        self.main_process = main_process
        self.world_size = world_size

        # If training STU
        if training_stu:
            self.m_y_learning_rate = 5e-5
            self.m_y_weight_decay = 0

        self.optimizer = self.get_optimizer(
            self.max_lr, self.betas, self.eps, self.weight_decay, self.use_amsgrad
        )

        self.model.to(self.device)

    def get_optimizer(self, lr, betas, eps, weight_decay, use_amsgrad):
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
                        "name": "default",
                        "params": default_params,
                        "lr": self.max_lr,
                        "weight_decay": self.weight_decay,
                    },
                    {
                        "name": "m_y",
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
                        "name": "decay",
                        "params": decay_params,
                        "lr": self.max_lr,
                        "weight_decay": self.weight_decay,
                    },
                    {
                        "name": "no_decay",
                        "params": nodecay_params,
                        "lr": self.max_lr,
                        "weight_decay": 0.0,
                    },
                ]
            )

        if self.main_process:
            for group in param_groups:
                colored_print(
                    f'\nOptimizer | Group {group["name"]}: '
                    f'{len(group["params"])} tensors, '
                    f'{sum(p.numel() for p in group["params"]):,} parameters, '
                    f'lr: {group["lr"]}, weight_decay: {group["weight_decay"]}',
                    Colors.HEADER,
                )

        fused_available = "fused" in inspect.signature(AdamW).parameters
        use_fused = fused_available and self.device.type == "cuda"

        if self.main_process:
            colored_print(f"Optimizer | Using fused AdamW?: {use_fused}", Colors.HEADER)

        return AdamW(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=use_amsgrad,
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
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps

        # 2. If it > lr_decay_iters, return min learning rate
        if it > num_steps:
            return min_lr

        # 3. If in between, cosine decay to down to min learning rate
        decay_ratio = (it - warmup_steps) / (num_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    def step(
        self,
        inputs: torch.Tensor, 
        targets: torch.Tensor, 
        relative_step: int
    ) -> dict[str, float]:
        """
        Perform a single training step.

        Args:
            inputs (torch.Tensor): A batch of input data.
            targets (torch.Tensor): A batch of target labels.
            relative_step (int): The current step relative to the start of training.

        Returns:
            dict[str, float]: A dictionary of metrics for the training step.
        """
        self.model.train()
        self.optimizer.zero_grad()
        t0 = time()

        inputs, targets = inputs.to(self.device), targets.to(self.device)

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            preds, loss_info = self.model(inputs, targets)

        if isinstance(loss_info, tuple):
            loss, *step_metrics = loss_info
        else:
            loss = loss_info

        loss.backward()

        if self.world_size > 1:
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

        # Time how long this training step took
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time()
        dt = t1 - t0
        toks_processed = self.bsz * self.sl * self.world_size
        toks_per_sec = toks_processed / dt

        # Step the learning rate scheduler forward
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == "m_y":
                param_group["lr"] = self.get_lr(
                    relative_step,
                    self.warmup_steps,
                    self.num_steps,
                    self.m_y_learning_rate,
                    self.min_lr,
                )
            else:
                param_group["lr"] = self.get_lr(
                    relative_step,
                    self.warmup_steps,
                    self.num_steps,
                    self.max_lr,
                    self.min_lr,
                )

        metrics = {
            "loss": loss.item(),
            "grad_norm": norm.item(),
            "step_time": dt,
            "tokens_per_sec": toks_per_sec,
        }

        # Add additional metrics if available
        if isinstance(loss_info, dict):
            metrics.update({k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_info.items()})

        return metrics

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate the model over an entire validation dataset.

        Args:
            dataloader (DataLoader): A DataLoader providing batches of data for evaluation.

        Returns:
            Dict[str, float]: A Dictionary of aggregated metrics over the dataset.
        """
        self.model.eval()
        val_steps = len(dataloader)
        metrics_accum = {
            "loss": 0.0,
            "tokens_processed": 0,
            "total_time": 0.0
        }
        additional_metrics = {}

        with torch.no_grad(), tqdm(total=val_steps, desc="Validating", disable=not self.main_process) as pbar:
            for inputs, targets in dataloader:
                t0 = time()
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    preds, loss_info = self.model(inputs, targets)

                if isinstance(loss_info, tuple):
                    loss, *step_metrics = loss_info
                else:
                    loss = loss_info

                # Accumulate loss
                metrics_accum["loss"] += loss.item()

                # Accumulate additional metrics if available
                if isinstance(loss_info, dict):
                    for key, value in loss_info.items():
                        if key not in additional_metrics:
                            additional_metrics[key] = 0.0
                        additional_metrics[key] += value.item() if isinstance(value, torch.Tensor) else value

                # Time tracking
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time()
                dt = t1 - t0

                # Token processing tracking
                metrics_accum["tokens_processed"] += inputs.numel()
                metrics_accum["total_time"] += dt

                pbar.update(1)

        # Average the accumulated metrics
        metrics_avg = {
            "loss": metrics_accum["loss"] / val_steps,
            "tokens_per_sec": metrics_accum["tokens_processed"] / metrics_accum["total_time"]
        }

        # Average additional metrics
        for key, value in additional_metrics.items():
            metrics_avg[key] = value / val_steps

        # Synchronize metrics across processes if using distributed training
        if self.world_size > 1:
            for key in metrics_avg:
                dist.all_reduce(torch.tensor(metrics_avg[key]).to(self.device), op=dist.ReduceOp.AVG)
                metrics_avg[key] = metrics_avg[key].item()

        return metrics_avg
