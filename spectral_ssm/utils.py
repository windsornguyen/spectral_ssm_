# =============================================================================#
# Authors: Windsor Nguyen, Dwaipayan Saha
# File: utils.py
# =============================================================================#

"""General-purpose utilities."""

import torch
from typing import Any, Callable


def broadcast_to_local_devices(tensor: torch.Tensor, devices: list) -> list:
    """Broadcasts a Pytree to all local devices.

    Args:
      pytree: The Pytree to broadcast.

    Returns:
      A Pytree with the same structure as `pytree`, but with values broadcasted
      to all local devices.
    """
    return [tensor.to(device) for device in devices]

def map_nested_fn(
    fn: Callable[[str, torch.Tensor], torch.Tensor],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Recursively apply `fn` to the key-value pairs of a nested dict.

    Example from optax.multi_transform for defining custom schedulers.

    Args:
      fn: local function applied to leaves mapping (k, v) to string key

    Returns:
      function mapping parameter names to key
    """

    def map_fn(nested_dict):
        return {
            k: map_fn(v) if isinstance(v, dict) else fn(k, v)
            for k, v in nested_dict.items()
        }
    return map_fn
