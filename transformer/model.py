# =============================================================================#
# Authors: Isabel Liu, Windsor Nguyen
# File: (Transformer) model.py
#
# Full definition of a GPT-based Transformer language model, adapted for regression.
# References:
# 1) the official GPT-2 TensorFlow implementation released by OpenAI:
# https://github.com/openai/gpt-2/blob/master/src/model.py
# 2) huggingface/transformers PyTorch implementation:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
#
# =============================================================================#


import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from dataclasses import dataclass


class CausalSelfAttention(nn.Module):
    """
    Self-attention layer for the Transformer.
    """

    def __init__(self, config):
        super(CausalSelfAttention, self).__init__()
        assert config.n_embd % config.n_head == 0

        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(
            config.n_embd, 3 * config.n_embd, bias=config.bias
        )

        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Flash attention makes GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(
            torch.nn.functional, 'scaled_dot_product_attention'
        )
        if not self.flash:
            print(
                'WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0'
            )
            # Causal mask to ensure attention is only applied to the left in input sequence
            self.register_buffer(
                'mask',
                torch.tril(torch.ones(config.max_len, config.max_len)).view(
                    1, 1, config.max_len, config.max_len
                ),
            )

    def forward(self, x):
        """
        Performs the forward pass of the CausalSelfAttention layer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C), where B is the batch size,
                T is the sequence length, and C is the embedding dimensionality (n_embd).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, C) after applying self-attention.
        """
        # Batch size, sequence length, embedding dimensionality (n_embd)
        B, T, C = x.size()

        # Compute query, key, values for all heads in batch, and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # Causal self-attention; self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # Efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # Manual implementation of self-attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # -> (B, 1, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """
    Simple multi-layer perceptron.
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(
            config.n_embd, config.scale * config.n_embd, bias=config.bias
        )
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(
            config.scale * config.n_embd, config.n_embd, bias=config.bias
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Single block of the Transformer.
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class TransformerConfigs:
    n_layer: int = 6
    n_head: int = 1
    n_embd: int = 37  # Ant-v1 default
    scale: int = 4
    d_out: int = 29  # Ant-v1 default
    max_len: int = 1_000
    bias: bool = False  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    dropout: float = 0.25
    loss_fn: nn.Module = None


class Transformer(nn.Module):
    """
    Transformer architecture for regression, adapted from the GPT-2 implementation.
    """

    def __init__(self, config):
        super(Transformer, self).__init__()
        assert config.max_len is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wpe=nn.Embedding(config.max_len, config.n_embd),
                dropout=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.regression_head = nn.Linear(
            config.n_embd, config.d_out, bias=False
        )
        self.loss_fn = self.config.loss_fn

        # Initialize all weights
        self.apply(self._init_weights)

        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # Report the number of parameters
        print(
            'Model parameter count (excluding positional embeddings): %.2fM'
            % (self.get_num_params() / 1e6,)
        )

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, inputs, targets=None):
        """
        inputs: (batch_size, seq_len, d_in)
        """
        device = inputs.device
        batch_size, seq_len, d_in = inputs.size()
        assert (
            seq_len <= self.config.max_len
        ), f'Cannot forward sequence of length {seq_len}, block size is only {self.config.max_len}'
        pos = (
            torch.arange(0, seq_len, dtype=torch.long, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )  # shape (b, t)

        # Forward the Transformer model itself
        pos_emb = self.transformer.wpe(
            pos
        )  # Position embeddings of shape (t, n_embd)
        x = self.transformer.dropout(pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        preds = self.regression_head(x)  # -> (batch_size, seq_len, d_out)

        if targets is not None:
            loss = self.loss_fn(preds, targets)
        else:
            loss = None
        return preds, loss

    # TODO: Not sure when/where this could be used, but we'd like to use it!
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS."""
        # First, estimate the number of flops we do per iteration.
        # See the PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config

        L, H, Q, T = (
            cfg.n_layer,
            cfg.n_head,
            cfg.n_embd // cfg.n_head,
            cfg.max_len,
        )
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        # Express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def predict(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        init: int = 0,
        steps: int = 100,
        ar_steps: int = 1,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        """
        Predicts the next states for a given set of input trajectories using vectorized operations.

        Args:
            inputs (torch.Tensor): A tensor of input trajectories with shape [num_trajectories, seq_len, d_in].
            targets (torch.Tensor): A tensor of target trajectories with shape [num_trajectories, seq_len, d_out].
            init (int): The index of the initial state to start the prediction from. Defaults to 0.
            steps (int): The number of time steps to predict. Defaults to 100.
            ar_steps (int): The number of autoregressive steps to take before using the ground truth state.
                Defaults to 1, which means the model always uses the ground truth state to predict the next state.

        Returns:
            tuple[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]]:
                - preds (torch.Tensor): A tensor of predicted states for each trajectory after `steps` time steps,
                    with shape [num_trajectories, steps, d_out].
                - loss (tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]): A tuple containing:
                    - avg_loss (torch.Tensor): The mean loss over time steps and trajectories.
                    - avg_metrics (dict[str, torch.Tensor]): A dictionary of mean losses for each metric.
                    - trajectory_losses (torch.Tensor): A tensor of losses for each trajectory at each time step,
                        with shape [num_trajectories, steps].
        """

        device = next(self.parameters()).device
        print(f'Predicting on {device}.')
        num_trajectories, seq_len, d_in = inputs.size()

        # Initialize the predicted sequences and losses
        ar_sequences = inputs.clone()
        preds = torch.zeros(
            num_trajectories, steps, self.config.d_out, device=device
        )
        trajectory_losses = torch.zeros(num_trajectories, steps, device=device)
        metrics = {
            key: torch.zeros(num_trajectories, steps, device=device)
            for key in [
                'coordinate_loss',
                #  'orientation_loss',
                'angle_loss',
                'coordinate_velocity_loss',
                'angular_velocity_loss',
            ]
        }

        # Initialize initial autoregressive sequences up to `init` steps for each trajectory
        ar_sequences[:, : init + 1, :] = inputs[:, : init + 1, :]
        u_start = targets.shape[2]
        u_end = inputs.shape[2] # TODO: Unused...

        # Iterate over the specified number of time steps
        for i in tqdm(range(steps), desc='Predicting', unit='step'):
            xs = ar_sequences[:, : i + 1 + init, :]
            ys = targets[:, : i + 1 + init, :]

            preds_step, (step_loss, step_metrics) = self.forward(xs, ys)

            preds[:, i, :] = preds_step[:, -1, :]

            # Update autoregressive sequences for each trajectory independently
            if i < steps - 1:
                for traj_idx in range(num_trajectories):
                    next_input = ar_sequences[traj_idx, i + 1 + init, :].clone()
                    next_input[:u_start] = (
                        preds[traj_idx, i, :]
                        if (i + 1) % ar_steps != 0
                        else inputs[traj_idx, i + 1 + init, :u_start]
                    )
                    ar_sequences[traj_idx, i + 1 + init, :] = next_input

            trajectory_losses[:, i] = step_loss

            for key in metrics:
                metrics[key][:, i] = step_metrics[key]

        # If we've reached the end of the input sequence but still have steps to predict,
        # use the last predicted state as input (we need to hallucinate and autoregressively predict)
        for i in range(seq_len - init, steps):
            xs = ar_sequences[:, -1, :].unsqueeze(1)
            ys = None

            preds_step, (step_loss, step_metrics) = self.forward(xs, ys)

            preds[:, i, :] = preds_step[:, -1, :]

            # Update autoregressive sequences for each trajectory independently
            if i < steps - 1:
                for traj_idx in range(num_trajectories):
                    next_input = ar_sequences[traj_idx, -1, :].clone()
                    next_input[:u_start] = preds[traj_idx, i, :]
                    ar_sequences[traj_idx] = torch.cat(
                        (ar_sequences[traj_idx], next_input.unsqueeze(0)), dim=0
                    )

            trajectory_losses[:, i] = step_loss

            for key in metrics:
                metrics[key][:, i] = step_metrics[key]

        # Calculate average losses and metrics across trajectories
        avg_loss = trajectory_losses.mean()
        avg_metrics = {key: metrics[key].mean() for key in metrics}

        return preds, (avg_loss, avg_metrics, trajectory_losses)
