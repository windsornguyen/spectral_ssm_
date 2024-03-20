# ==============================================================================#
# Authors: Windsor Nguyen, Dwaipayan Saha
# File: optimizer.py
# ==============================================================================#


"""AdamW with linear warmup and cosine decay."""

import torch
from torch.optim import AdamW


class WarmupCosineDecay(torch.optim.lr_scheduler._LRScheduler):
    """Cosine decay with linear warmup."""

    def __init__(
        self,
        optimizer,
        start_val,
        min_lr,
        lr,
        num_steps,
        warmup_steps,
        last_epoch=-1,
    ):
        """Initialize a cosine decay schedule with warmup.
        Args:
            start_val: The value to start at.
            min_lr: The minimum value to decay to.
            lr: The peak value to reach.
            num_steps: The total number of steps to decay over.
            warmup_steps: The number of steps to warmup for.
        """
        self.start_val = start_val
        self.min_lr = min_lr
        self.lr = lr
        self.num_steps = num_steps
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Get learning rate for a given step.
        Args:
            itr: The current step.

        Returns:
            The learning rate for the given step.
        """
        if self.last_epoch < self.warmup_steps:
            warmup_factor = self.last_epoch / self.warmup_steps
            return [
                (self.start_val + warmup_factor * (self.lr - self.start_val))
                for _ in self.base_lrs
            ]

        # Cosine annealing
        cos_factor = 0.5 * (
            1
            + torch.cos(
                torch.tensor(torch.pi
                            * (self.last_epoch - self.warmup_steps) 
                            / (self.num_steps - self.warmup_steps)
                )
            )
        )
        return [
            (self.min_lr + (self.lr - self.min_lr) * cos_factor) for _ in self.base_lrs
        ]

    # TODO: Use this somewhere?
    def get_last_lr(self):
        """Get last computed learning rate by scheduler."""
        return self._last_lr


def get_optimizer(
    model,
    num_steps=180_000,
    warmup_steps=18_000,
    learning_rate=5e-4,
    weight_decay=0.1,
    m_y_learning_rate=5e-5,
    m_y_weight_decay=0.0,
):
    m_y_params = []
    default_params = []
    for name, param in model.named_parameters():
        if name.startswith('m_y'):
            m_y_params.append(param)
        else:
            default_params.append(param)

    param_groups = [
        {'params': default_params, 'lr': learning_rate, 'weight_decay': weight_decay},
        {
            'params': m_y_params,
            'lr': m_y_learning_rate,
            'weight_decay': m_y_weight_decay,
        },
    ]

    optimizer = AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    scheduler = WarmupCosineDecay(
        optimizer,
        start_val=1e-7,
        min_lr=1e-7,
        lr=learning_rate,
        num_steps=num_steps,
        warmup_steps=warmup_steps,
    )

    return optimizer, scheduler
