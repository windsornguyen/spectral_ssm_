# ==============================================================================#
# Authors: Windsor Nguyen, Dwaipayan Saha, Isabel Liu
# File: model.py
# ==============================================================================#

"""Spectral temporal unit (STU) block."""

import torch
import torch.nn as nn

from dataclasses import dataclass, field
from models.stu import stu_utils
from time import time
from utils.swiglu import SwiGLU
from tqdm import tqdm


@dataclass
class SSSMConfigs:
    n_layers: int = 6
    n_embd: int = 512
    d_out: int = 18
    sl: int = 300
    scale: int = 4
    bias: bool = False
    dropout: float = 0.10
    num_eigh: int = 24
    auto_reg_k_u: int = 3
    auto_reg_k_y: int = 32
    learnable_m_y: bool = True
    loss_fn: nn.Module = nn.MSELoss()
    controls: dict = field(
        default_factory=lambda: {"task": "mujoco-v3", "controller": "Ant-v1"}
    )


class STU(nn.Module):
    """
    A simple STU (Spectral Transform Unit) Layer.

    Args:
        d_out (int): Output dimension.
        sl (int): Input sequence length.
        num_eigh (int): Number of eigenvalues and eigenvectors to use.
        auto_reg_k_u (int): Auto-regressive depth on the input sequence.
        auto_reg_k_y (int): Auto-regressive depth on the output sequence.
        learnable_m_y (bool): Whether the m_y matrix is learnable.
    """

    def __init__(self, configs) -> None:
        super(STU, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_embd = configs.n_embd
        self.d_out = configs.d_out
        self.sl, self.k = configs.sl, configs.num_eigh
        self.eigh = stu_utils.get_top_hankel_eigh(self.sl, self.k, self.device)

        # Initialize M matrices
        self.auto_reg_k_u = configs.auto_reg_k_u
        self.auto_reg_k_y = configs.auto_reg_k_y
        self.learnable_m_y = configs.learnable_m_y
        self.m_u = nn.Parameter(torch.empty(self.d_out, self.d_out, self.auto_reg_k_u))
        self.m_phi = nn.Parameter(torch.empty(self.d_out * self.k, self.d_out))
        self.m_y = (
            nn.Parameter(torch.empty(self.d_out, self.auto_reg_k_y, self.d_out))
            if self.learnable_m_y
            else self.register_buffer(
                "m_y", torch.empty(self.d_out, self.auto_reg_k_y, self.d_out)
            )
        )

        # The output projection
        self.proj = nn.Linear(self.n_embd, self.d_out)
        self.proj.SCALE_INIT = 1

        # Regularization
        self.dropout = configs.dropout
        self.stu_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

    def forward(self, inputs):
        eig_vals, eig_vecs = self.eigh

        x_tilde = stu_utils.compute_x_tilde(inputs, (eig_vals, eig_vecs))
        x_tilde = self.stu_dropout(x_tilde)

        delta_phi = x_tilde @ self.m_phi

        delta_ar_u = stu_utils.compute_ar_x_preds(self.m_u, inputs)

        y_t = stu_utils.compute_y_t(self.m_y, delta_phi + delta_ar_u)

        return self.resid_dropout(y_t)


class FFN(nn.Module):
    """
    Simple feed-forward network.
    """

    def __init__(self, configs):
        super(FFN, self).__init__()
        # TODO: Consider implementing Squared ReLU from https://arxiv.org/pdf/2109.08668 ??
        self.swiglu = SwiGLU(
            configs.n_embd, configs.scale * configs.n_embd, bias=configs.bias
        )
        self.proj = nn.Linear(
            configs.scale * configs.n_embd, configs.d_out, bias=configs.bias
        )
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x):
        x = self.swiglu(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class STUBlock(nn.Module):
    def __init__(self, configs):
        super(STUBlock, self).__init__()
        self.ln_1 = nn.LayerNorm(configs.n_embd, bias=configs.bias)
        self.stu = STU(configs)
        self.ln_2 = nn.LayerNorm(configs.n_embd, bias=configs.bias)
        self.ffn = FFN(configs)

    def forward(self, x):
        x = x + self.stu(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class SSSM(nn.Module):
    """
    General model architecture based on STU blocks.
    """

    def __init__(self, configs):
        super(SSSM, self).__init__()
        self.configs = configs
        self.n_layers = configs.n_layers
        self.n_embd = configs.n_embd
        self.d_out = configs.d_out
        self.sl, self.k = configs.sl, configs.num_eigh

        self.bias = configs.bias
        self.dropout = configs.dropout
        self.loss_fn = configs.loss_fn
        self.controls = configs.controls
        self.task_head = nn.Linear(self.n_embd, self.d_out, bias=self.bias)

        self.emb = nn.Linear(self.n_embd, self.n_embd)
        self.stu = nn.ModuleDict(
            dict(
                # Since our tasks are continuous, we do not use token embeddings.
                wpe=nn.Embedding(self.sl, self.n_embd),
                dropout=nn.Dropout(self.dropout),
                hidden=nn.ModuleList([STUBlock(configs) for _ in range(self.n_layers)]),
                ln_f=nn.LayerNorm(self.n_embd, bias=self.bias),
            )
        )

        if self.controls["task"] == "mujoco-v1":
            if self.controls["controller"] == "Ant-v1":
                self.d_out = 29
            else:
                self.d_out = 18

        # Initialize all weights
        self.m_x = float(self.d_out) ** -0.5
        self.std = 0.02
        self.apply(self._init_weights)

        # Report the number of parameters
        print("STU Model Parameter Count: %.2fM" % (self.get_num_params() / 1e6,))

    def forward(self, inputs, targets):
        # TODO: Add docstrings to this and shape annotate each line
        device = inputs.device
        bsz, sl, n_embd = inputs.size()
        x = self.emb(inputs)

        # Generate positional embeddings for the sequence
        pos = torch.arange(0, sl, dtype=torch.long, device=device)  # -> (sl)
        pos_emb = self.stu.wpe(pos)

        # Add positional embeddings to input
        x = x + pos_emb.unsqueeze(0)
        x = self.stu.dropout(x)

        for stu_block in self.stu.hidden:
            x = stu_block(x)
        x = self.stu.ln_f(x)
        preds = self.task_head(x)

        if self.controls["task"] != "mujoco-v3":
            loss, metrics = (
                self.loss_fn(preds, targets) if targets is not None else (None, None)
            )
            return preds, (loss, metrics)
        else:
            # print("Is targets none?", targets is None)
            loss = self.loss_fn(preds, targets) if targets is not None else None
            # print("loss is", loss)
            # print('preds are', preds[:10])
            return preds, (loss,)

    def _init_weights(self, module):
        """
        Initialize the weights of the model.

        Args:
            module (nn.Module): The module to initialize.
        """
        if isinstance(module, nn.Linear):
            if hasattr(module, "SCALE_INIT"):
                # Scale by 2 to account for stu and ffn sub-layer
                self.std *= (2 * self.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
        elif isinstance(module, STU):
            # Custom initialization for m_u, m_phi, and m_y matrices
            torch.nn.init.uniform_(module.m_u, -self.m_x, self.m_x)
            torch.nn.init.xavier_normal_(module.m_phi)
            if module.learnable_m_y:
                torch.nn.init.xavier_normal_(module.m_y)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.

        Args:
            non_embedding (bool, optional):
            Whether to exclude the positional embeddings (if applicable).
            Defaults to True.

        Returns:
            int: The number of parameters in the model.
        """
        num_params = sum(p.numel() for p in self.parameters())
        return num_params

    # TODO: Not sure when/where this could be used, but we'd like to use it!
    # TODO: Also need to fix this function to make sure it's correct.
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.configs
        L, D, E, T = cfg.num_layers, cfg.n_embd, cfg.num_eigh, cfg.input_len

        # Embedding layers
        embed_flops = 2 * D * T

        # STU blocks
        stu_block_flops = 0
        for _ in range(L):
            # Layer normalization
            stu_block_flops += 2 * D * T  # ln_1 and ln_2

            # STU layer
            stu_block_flops += 2 * E * D * T  # Compute x_tilde
            stu_block_flops += 2 * D * E * D  # Apply m_phi matrix

            # FFN layer
            stu_block_flops += 2 * D * cfg.scale * D  # c_fc
            stu_block_flops += cfg.scale * D  # GELU activation
            stu_block_flops += 2 * cfg.scale * D * D  # c_proj

        # Final layer normalization
        final_ln_flops = 2 * D * T  # ln_f

        # Language model head
        lm_head_flops = 2 * D * cfg.vocab_size

        flops_per_iter = embed_flops + stu_block_flops + final_ln_flops + lm_head_flops
        flops_per_fwdbwd = flops_per_iter * fwdbwd_per_iter

        # Express flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_fwdbwd / dt  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    # TODO: Also need to fix this function to make sure it's correct.
    def flops_per_token(self):
        """Estimate the number of floating-point operations per token."""
        flops = 0
        cfg = self.configs

        # Embedding layers
        flops += 2 * cfg.n_embd * cfg.block_size  # wte and wpe embeddings

        # STU blocks
        for _ in range(cfg.num_layers):
            # Layer normalization
            flops += 2 * cfg.n_embd * cfg.block_size  # ln_1 and ln_2

            # STU layer
            flops += 2 * cfg.num_eigh * cfg.n_embd * cfg.block_size  # Compute x_tilde
            flops += 2 * cfg.n_embd * cfg.num_eigh * cfg.n_embd  # Apply m_phi matrix

            # FFN layer
            flops += 2 * cfg.n_embd * cfg.scale * cfg.n_embd  # c_fc
            flops += cfg.scale * cfg.n_embd  # GELU activation
            flops += 2 * cfg.scale * cfg.n_embd * cfg.n_embd  # c_proj

        # Final layer normalization
        flops += 2 * cfg.n_embd * cfg.block_size  # ln_f

        # Language model head
        flops += 2 * cfg.n_embd * cfg.vocab_size

        return flops

    def predict_frames(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        init: int = 100,
        steps: int = 50,
        ar_steps: int = 300,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        """
        Predicts the video frame.

        Args:
            inputs (torch.Tensor): A tensor of input videos with shape [num_videos, sl, d_in].
            targets (torch.Tensor): A tensor of target videos with shape [num_videos, sl, d_in].
            init (int): The index of the initial state to start the prediction from. Defaults to 0.
            steps (int): The number of time steps to predict. Defaults to 50.
            ar_steps (int): The number of autoregressive steps to take before using the ground truth state.
                Defaults to 1, which means the model always uses the ground truth state to predict the next state.
                If set to sl, the model always uses the last predicted state to predict the next state.

        Returns:
            tuple[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]]:
                - preds (torch.Tensor): A tensor of predicted states for each video after `steps` time steps,
                    with shape [num_videos, steps, d_out].
                - loss (tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]): A tuple containing:
                    - avg_loss (torch.Tensor): The mean loss over time steps and videos.
                    - video_losses (torch.Tensor): A tensor of losses for each video at each time step,
                        with shape [num_videos, steps].
        """
        device = next(self.parameters()).device
        print(f"Predicting on {device}.")
        num_videos, sl, d_in = inputs.size()

        # Initialize the predicted sequences and losses
        ar_sequences = inputs.clone()
        preds = torch.zeros(num_videos, steps, d_in, device=device)
        video_losses = torch.zeros(num_videos, steps, device=device)

        i = init
        with tqdm(total=steps, desc="Predicting", unit="step") as pbar:
            while i < init + steps:
                window_start = max(0, i - self.configs.sl + 1)

                input_window = ar_sequences[:, window_start : i + 1, :]
                target_window = targets[:, window_start : i + 1, :]
                preds_step, (step_loss,) = self.forward(input_window, target_window)

                preds[:, i - init, :] = preds_step[:, -1, :]
                video_losses[:, i - init] = step_loss

                # Update autoregressive sequences for the next step
                if i < init + steps - 1:
                    next_step = i + 1
                    if next_step < sl:
                        next_input = (
                            preds[:, i - init, :]
                            if (i - init + 1) % ar_steps != 0
                            else inputs[:, next_step, :]
                        )
                        ar_sequences[:, next_step, :] = next_input
                    else:
                        ar_sequences = torch.cat(
                            [
                                ar_sequences[:, 1:, :],
                                preds[:, i - init : i - init + 1, :],
                            ],
                            dim=1,
                        )

                i += 1
                pbar.update(1)

        # # If we've reached the end of the input sequence but still have steps to predict,
        # # use the last predicted state as input (we need to hallucinate and autoregressively predict)
        # for step in range(sl - init, steps):
        #     xs = ar_sequences[:, -1, :].unsqueeze(1)
        #     ys = None

        #     preds_step, step_loss = self.forward(xs, ys)

        #     preds[:, i, :] = preds_step[:, -1, :]

        #     # Update autoregressive sequences for each video independently
        #     if step < steps - 1:
        #         for video_idx in range(num_videos):
        #             next_input = ar_sequences[video_idx, -1, :].clone()
        #             next_input = preds[video_idx, i, :]
        #             ar_sequences[video_idx] = ar_sequences[video_idx, step + 1 + init, :] = next_input

        #     video_losses[:, i] = step_loss

        # Calculate average losses and metrics across videos
        avg_loss = video_losses.mean()

        return preds, (avg_loss, video_losses)
