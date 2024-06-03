"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass
from typing import Tuple, Dict

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super(CausalSelfAttention, self).__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.max_len, config.max_len))
                                        .view(1, 1, config.max_len, config.max_len))

    def forward(self, x):
        """
        x: input tensor of shape (seq_len, n_embd)
        """
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # k = k.view(T, self.n_head, C // self.n_head).transpose(0, 1) # (B, nh, T, hs)
        # q = q.view(T, self.n_head, C // self.n_head).transpose(0, 1) # (B, nh, T, hs)
        # v = v.view(T, self.n_head, C // self.n_head).transpose(0, 1) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C) # -> (B, 1, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, config.scale * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(config.scale * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class AntLoss(nn.Module):
    def __init__(self):
        super(AntLoss, self).__init__()

    def forward(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the loss and metrics for a batch of data.

        Args:
            outputs (torch.Tensor): The model outputs of shape (batch_size, seq_len, d_xt)
            targets (torch.Tensor): The target labels of shape (batch_size, seq_len, d_xt)

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: 
            A Tuple of the loss and a dictionary of metrics.
        """
        total_loss = torch.tensor(0.0, device=outputs.device)
        for i in range(outputs.shape[1]):
            loss = (outputs[:, i] - targets[:, i]) ** 2

            # scaling by constant just for now
            if i in (0, 1, 2):  # coordinates of the torso (center)
                loss /= 5
                # print(f'Index {i}, Coordinate Loss Scale /5: {loss.mean().item()}')
            elif i in (3, 4, 5, 6):  # orientations of the torso (center)
                loss /= 0.2
                # print(f'Index {i}, Orientation Loss Scale /0.2: {loss.mean().item()}')
            elif i in (7, 8, 9, 10, 11, 12, 13, 14):  # angles between the torso and the links
                loss /= 0.5
                # print(f'Index {i}, Angle Loss Scale /0.5: {loss.mean().item()}')
            elif i in (15, 16, 17, 18, 19, 20):  # coordinate and coordinate angular velocities of the torso (center)
                loss /= 2
                # print(f'Index {i}, Velocity Loss Scale /2: {loss.mean().item()}')
            elif i in (21, 22, 23, 24, 25, 26, 27, 28):  # angular velocities of the angles between the torso and the links
                loss /= 5
                # print(f'Index {i}, Angular Velocity Loss Scale /5: {loss.mean().item()}')

            total_loss += loss.mean()

        total_loss = total_loss / outputs.shape[1]
        metrics = {'loss': total_loss.item()}
        # print(f'Total Scaled Loss: {total_loss.item()}')

        return total_loss, metrics


@dataclass
class TransformerConfig:
    n_layer: int = 6
    n_head: int = 1
    n_embd: int = 37
    scale: int = 4
    d_out: int = 29
    max_len: int = 1_000
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    dropout: float = 0.0


class Transformer(nn.Module):

    def __init__(self, config):
        super(Transformer, self).__init__()
        assert config.max_len is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wpe = nn.Embedding(config.max_len, config.n_embd),
            dropout = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.regression_head = nn.Linear(config.n_embd, config.d_out, bias=False)
        self.loss_fn = AntLoss()

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

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

    def forward(self, idx, targets=None):
        """
        idx: (batch_size, seq_len, d_in)
        """
        device = idx.device
        batch_size, seq_len, d_in = idx.size()
        assert seq_len <= self.config.max_len, f"Cannot forward sequence of length {t}, block size is only {self.config.max_len}"
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1) # shape (b, t)

        # forward the GPT model itself
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.dropout(pos_emb)
        for block in self.transformer.h:
            x = block(x)
        layer_norm = self.transformer.ln_f(x)
        x = self.transformer.ln_f(x)

        stuff = self.regression_head(x)
        preds = self.regression_head(x).squeeze(-1)

        if targets is not None:
            loss = self.loss_fn(preds, targets)
        else:
            loss = None

        return preds, loss

    def crop_max_len(self, max_len):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert max_len <= self.config.max_len
        self.config.max_len = max_len
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:max_len])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:max_len,:max_len]

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.max_len
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

