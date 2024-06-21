# ==============================================================================#
# Authors: Windsor Nguyen, Dwaipayan Saha, Isabel Liu, Yagiz Devre
# File: model.py
# ==============================================================================#

"""Spectral temporal unit (STU) block."""

import torch
import torch.nn as nn

from dataclasses import dataclass, field
from models.stu import stu_utils
from time import time
from utils.swiglu import SwiGLU


@dataclass
class SSSMConfigs:
    n_layers: int = 6
    n_embd: int = 512
    d_model: int = 24
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
        default_factory=lambda: {'task': 'mujoco-v3', 'controller': 'Ant-v1'}
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

    def __init__(self, config) -> None:
        super(STU, self).__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.d_model = config.d_model
        self.d_out = config.d_out
        self.sl, self.k = config.sl, config.num_eigh
        self.eigh = stu_utils.get_top_hankel_eigh(self.sl, self.k, self.device)

        # Initialize M matrices
        self.auto_reg_k_u = config.auto_reg_k_u
        self.auto_reg_k_y = config.auto_reg_k_y
        self.learnable_m_y = config.learnable_m_y
        self.m_u = nn.Parameter(
            torch.empty(self.d_out, self.d_out, self.auto_reg_k_u)
        )
        self.m_phi = nn.Parameter(torch.empty(self.d_out * self.k, self.d_out))
        self.m_y = (
            nn.Parameter(torch.empty(self.d_out, self.auto_reg_k_y, self.d_out))
            if self.learnable_m_y
            else self.register_buffer(
                'm_y', torch.empty(self.d_out, self.auto_reg_k_y, self.d_out)
            )
        )

        # The output projection
        self.proj = nn.Linear(self.d_model, self.d_out)
        self.proj.SCALE_INIT = 1

        # Regularization
        self.dropout = config.dropout
        self.stu_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

    def forward(self, inputs):
        eig_vals, eig_vecs = self.eigh

        x_tilde = stu_utils.compute_x_tilde(inputs, (eig_vals, eig_vecs))
        x_tilde = self.stu_dropout(x_tilde)
        delta_phi = x_tilde @ self.m_phi

        ### AUTO-REGRESSIVE PART ###

        # # print(
        # #     f'Time for delta_phi computation: {time() - start_time:.4f}s'
        # # )
        # # start_time = time()  # Reset timing
        # delta_ar_u = stu_utils.compute_ar_x_preds(self.m_u, inputs)
        # # print(
        # #     f'Time for delta_ar_u computation: {time() - start_time:.4f}s'
        # # )
        # # start_time = time()  # Reset timing
        # y_t = stu_utils.compute_y_t(self.m_y, delta_phi + delta_ar_u)
        # # print(f'Time for y_t computation: {time() - start_time:.4f}s')

        # return y_t

        ### END AUTO-REGRESSIVE PART ###

        y_t = self.resid_dropout(delta_phi)
        return y_t


class FFN(nn.Module):
    """
    Simple feed-forward network.
    """

    def __init__(self, config):
        super(FFN, self).__init__()
        # TODO: Consider implementing Squared ReLU from https://arxiv.org/pdf/2109.08668 ??
        self.swiglu = SwiGLU(config.d_model, config.scale * config.d_model, bias=config.bias)
        self.proj = nn.Linear(config.scale * config.d_model, config.d_out, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # TODO: MAYBE consider using skip connection(s) here? Is that crazy?
        # TODO: Just test to see if skip connections help here.
        x = self.swiglu(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class STUBlock(nn.Module):
    def __init__(self, config):
        super(STUBlock, self).__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.stu = STU(config)
        self.ln_2 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.ffn = FFN(config)

    def forward(self, x):
        x = x + self.stu(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class SSSM(nn.Module):
    """
    General model architecture based on STU blocks.
    """

    def __init__(self, config):
        super(SSSM, self).__init__()
        self.n_layers = config.n_layers
        self.n_embd = config.n_embd
        self.d_out = config.d_out
        self.sl, self.k = config.sl, config.num_eigh

        self.bias = config.bias
        self.dropout = config.dropout
        self.loss_fn = config.loss_fn
        self.controls = config.controls
        self.task_head = nn.Linear(self.n_embd, self.d_out, bias=self.bias)

        self.emb = nn.Linear(self.n_embd, self.n_embd)
        self.stu = nn.ModuleDict(
            dict(
                # Since our tasks are continuous, we do not use token embeddings.
                wpe=nn.Embedding(self.sl, self.n_embd),
                dropout=nn.Dropout(self.dropout),
                hidden=nn.ModuleList(
                    [STUBlock(config) for _ in range(self.n_layers)]
                ),
                ln_f=nn.LayerNorm(self.n_embd, bias=self.bias),
            )
        )

        if self.controls['task'] == 'mujoco-v1':
            if self.controls['controller'] == 'Ant-v1':
                self.d_out = 29
            else:
                self.d_out = 18

        # Initialize all weights
        self.m_x = float(self.d_out) ** -0.5
        self.std = 0.02
        self.apply(self._init_weights)

        # Report the number of parameters
        print(
            'STU Model Parameter Count: %.2fM' % (self.get_num_params() / 1e6,)
        )

    def _init_weights(self, module):
        """
        Initialize the weights of the model.

        Args:
            module (nn.Module): The module to initialize.
        """
        if isinstance(module, nn.Linear):
            if hasattr(module, 'SCALE_INIT'):
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

    def forward(self, inputs, targets):
        # TODO: Add docstrings to this and shape annotate each line
        device = inputs.device
        bsz, sl, d_model = inputs.size()
        x = self.emb(inputs)

        # Generate positional embeddings for the sequence
        pos = torch.arange(0, sl, dtype=torch.long, device=device)  # -> (sl)
        pos_emb = self.stu.wpe(pos)

        # Add positional embeddings to input
        x = x + self.stu.dropout(pos_emb.unsqueeze(0).expand(bsz, -1, -1))
        for stu_block in self.stu.hidden:
            x = x + stu_block(x)
        x = self.stu.ln_f(x)
        preds = self.task_head(x)
        
        # TODO: I am pretty sure preds is wrong for task v3.
        if self.controls['task'] != 'mujoco-v3':
            loss, metrics = (
                self.loss_fn(preds, targets) if targets is not None else (None, None)
            )
            print(f'preds shape {preds.shape}')
            return preds, (loss, metrics)
        else:
            loss = self.loss_fn(preds, targets) if targets is not None else None
            print(f'preds shape {preds.shape}')
            return preds, (loss,)


    # TODO: Not sure when/where this could be used, but we'd like to use it!
    # TODO: Also need to fix this function to make sure it's correct.
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
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

        flops_per_iter = (
            embed_flops + stu_block_flops + final_ln_flops + lm_head_flops
        )
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
        cfg = self.config

        # Embedding layers
        flops += 2 * cfg.n_embd * cfg.block_size  # wte and wpe embeddings

        # STU blocks
        for _ in range(cfg.num_layers):
            # Layer normalization
            flops += 2 * cfg.n_embd * cfg.block_size  # ln_1 and ln_2

            # STU layer
            flops += (
                2 * cfg.num_eigh * cfg.n_embd * cfg.block_size
            )  # Compute x_tilde
            flops += (
                2 * cfg.n_embd * cfg.num_eigh * cfg.n_embd
            )  # Apply m_phi matrix

            # FFN layer
            flops += 2 * cfg.n_embd * cfg.scale * cfg.n_embd  # c_fc
            flops += cfg.scale * cfg.n_embd  # GELU activation
            flops += 2 * cfg.scale * cfg.n_embd * cfg.n_embd  # c_proj

        # Final layer normalization
        flops += 2 * cfg.n_embd * cfg.block_size  # ln_f

        # Language model head
        flops += 2 * cfg.n_embd * cfg.vocab_size

        return flops

    def predict(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        init: int = 0,
        t: int = 1,
    ) -> tuple[list[float], tuple[torch.Tensor, dict[str, float]]]:
        """
        Predicts the next states in trajectories and computes losses against the targets.

        Args:
            inputs (torch.Tensor): A tensor of shape [num_trajectories, seq_len, d_in].
            targets (torch.Tensor): A tensor of shape [num_trajectories, seq_len, d_out].
            init (int): The index of the initial state to start at.
            t (int): The number of time steps to predict.

        Returns:
            A tuple containing the list of predicted states after `t` time steps and
            a tuple containing the total loss and a dictionary of metrics.
        """
        device = inputs.device
        num_trajectories, seq_len, d_in = inputs.size()

        predicted_sequence = []
        total_loss = torch.tensor(0.0, device=device)
        metrics = {
            'loss': [],
            'coordinate_loss': [],
            'orientation_loss': [],
            'angle_loss': [],
            'coordinate_velocity_loss': [],
            'angular_velocity_loss': [],
        }

        for i in range(t):
            current_input_state = inputs[:, init + i, :].unsqueeze(1)
            current_target_state = targets[:, init + i, :].unsqueeze(1)

            # Predict the next state using the model
            next_state = self.model(current_input_state)
            loss, metric = self.loss_fn(next_state, current_target_state)

            predicted_sequence.append(next_state.squeeze(1).tolist())

            # Accumulate the metrics
            for key in metrics:
                metrics[key].append(metric[key])

            # Accumulate the losses
            total_loss += loss.item()

        total_loss /= t

        return predicted_sequence, (total_loss, metrics)
