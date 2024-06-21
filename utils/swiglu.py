# =============================================================================#
# Authors: Windsor Nguyen
# File: swiglu.py
# =============================================================================#

# TODO: Move this to another directory that makes sense.
# TODO: In general, organize the utils directory better.

"""
The SwiGLU activation function, 
from "GLU Variants Improve Transformer" (Shazeer, 2020).

From the paper:
'We offer no explanation as to why these architectures seem to work;
we attribute their success, as all else, to __divine benevolence__.'
"""

import torch.nn.functional as F
import torch.nn as nn


class SwiGLU(nn.Module):
    """
    The SwiGLU activation function as proposed by Noam Shazeer.
    
    This module implements the SwiGLU function defined as:
    SwiGLU(x) = Swish_{1}(xW) ⊙ (xV)
    where ⊙ denotes the Hadamard product and Swish_{1} is the Swish function with β=1.

    Note: The Swish function with β=1 is equivalent to PyTorch's SiLU function.

    Args:
        d_in (int): Input dimension.
        d_out (int): Output dimension.
        bias (bool, optional): If false, an additive bias will not be learned.
    """
    def __init__(self, d_in, d_out, bias):
        super().__init__()
        self.w = nn.Linear(d_in, d_out, bias=bias)
        self.v = nn.Linear(d_in, d_out, bias=bias)

    def forward(self, x):
        return F.silu(self.w(x)) * self.v(x)
