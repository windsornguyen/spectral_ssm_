# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""AdamW with linear warmup and cosine decay."""

import torch

class WarmupCosineDecay:
    """Cosine decay with linear warmup."""
    def __init__(
        self,
        start_val: float,
        min_lr: float,
        lr: float,
        num_steps: int,
        warmup_steps: int,
    ) -> None:
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

        def __call__(self, itr) -> torch.Tensor:
            """Get learning rate for a given step.
            Args:
                itr: The current step.

            Returns:
                The learning rate for the given step.
            """
            if itr < self.warmup_steps:
                # Correctly calculate warmup_val within the conditional check
                warmup_val = (self.lr - self.start_val) * (itr / self.warmup_steps) + self.start_val
                return warmup_val

            cos_itr = (itr - self.warmup_steps) / (self.num_steps - self.warmup_steps)
            cos = 1 + torch.cos(np.pi * cos_itr)
            cos_val = 0.5 * (self.lr - self.min_lr) * cos + self.min_lr

            return cos_val
        
        # TODO: Implement get_optimizer!
