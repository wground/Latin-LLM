"""
Full definition of a GPT Language Model, all of it in this single file.
Modernized architecture based on nanoChat and current LLM best practices.

Architecture (vs original nanoGPT):
- RoPE (Rotary Position Embeddings) instead of learned absolute positional embeddings
- Parameterless RMSNorm instead of LayerNorm
- SwiGLU MLP activation instead of GELU
- Grouped Query Attention (GQA) with configurable KV heads
- QK normalization for training stability
- Logit soft-capping for numerical stability
- No bias terms anywhere

References:
1) Andrej Karpathy's nanoChat: https://github.com/karpathy/nanochat
2) Andrej Karpathy's nanoGPT: https://github.com/karpathy/nanoGPT
3) LLaMA architecture: RoPE, SwiGLU, RMSNorm, GQA
4) Muon optimizer: https://github.com/KellerJordan/Muon

Author: Willow Groundwater-Schuldt
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# --- Rotary Position Embeddings (RoPE) ---

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    """
    Precompute cos/sin frequencies for RoPE.
    Returns tensor of shape (seq_len, dim//2, 2) where last dim is [cos, sin].
    Uses sin/cos directly (no complex numbers) for MPS compatibility.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (seq_len, dim//2)
    return torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)  # (seq_len, dim//2, 2)


def apply_rotary_emb(xq, xk, freqs_cis):
    """
    Apply rotary position embeddings to query and key tensors.
    xq, xk: (B, T, n_head, head_dim)
    freqs_cis: (T, head_dim//2, 2)
    """
    # Reshape to pairs: (..., head_dim) -> (..., head_dim//2, 2)
    xq_r = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_r = xk.float().reshape(*xk.shape[:-1], -1, 2)

    # Broadcast freqs: (1, T, 1, head_dim//2)
    freqs_cos = freqs_cis[..., 0].unsqueeze(0).unsqueeze(2)
    freqs_sin = freqs_cis[..., 1].unsqueeze(0).unsqueeze(2)

    # Rotation: (x0, x1) -> (x0*cos - x1*sin, x0*sin + x1*cos)
    xq_out = torch.stack([
        xq_r[..., 0] * freqs_cos - xq_r[..., 1] * freqs_sin,
        xq_r[..., 0] * freqs_sin + xq_r[..., 1] * freqs_cos,
    ], dim=-1).flatten(-2)

    xk_out = torch.stack([
        xk_r[..., 0] * freqs_cos - xk_r[..., 1] * freqs_sin,
        xk_r[..., 0] * freqs_sin + xk_r[..., 1] * freqs_cos,
    ], dim=-1).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


# --- Normalization ---

class RMSNorm(nn.Module):
    """Root Mean Square Normalization without learnable parameters (nanoChat style)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).type_as(x)


# --- Attention with GQA, QK-Norm, and RoPE ---

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep = config.n_head // config.n_kv_head  # GQA repetition factor
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Separate Q and KV projections for GQA
        self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # QK normalization for stability
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        # Regularization
        self.resid_dropout = nn.Dropout(config.dropout)

        # Flash attention
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, freqs_cis):
        B, T, C = x.size()

        # Project Q, K, V
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)

        # QK normalization (before RoPE, per nanoChat)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE to Q and K
        q, k = apply_rotary_emb(q, k, freqs_cis[:T])

        # GQA: expand KV heads to match Q heads by repeating
        if self.n_rep > 1:
            k = k.unsqueeze(3).expand(B, T, self.n_kv_head, self.n_rep, self.head_dim)
            k = k.reshape(B, T, self.n_head, self.head_dim)
            v = v.unsqueeze(3).expand(B, T, self.n_kv_head, self.n_rep, self.head_dim)
            v = v.reshape(B, T, self.n_head, self.head_dim)

        # Transpose for attention: (B, n_head, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


# --- SwiGLU MLP ---

class SwiGLUMLP(nn.Module):
    """
    MLP with SwiGLU activation (LLaMA-style).
    SwiGLU(x) = (SiLU(xW_gate) * xW_up) W_down
    Uses 3 projections but with reduced hidden dim (8/3 * n_embd) for parameter parity.
    """

    def __init__(self, config):
        super().__init__()
        hidden_dim = config.intermediate_size
        self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# --- Transformer Block ---

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLUMLP(config)

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.ln_1(x), freqs_cis)
        x = x + self.mlp(self.ln_2(x))
        return x


# --- Configuration ---

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # Overridden by tokenizer config
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 0       # 0 = same as n_head (standard MHA); < n_head enables GQA
    n_embd: int = 768
    intermediate_size: int = 0  # 0 = auto-compute (8/3 * n_embd rounded to multiple of 64)
    dropout: float = 0.0
    softcap: float = 30.0    # Output logit soft-capping (0 = disabled)
    rope_theta: float = 10000.0  # RoPE frequency base

    def __post_init__(self):
        # Default KV heads to full MHA if not specified
        if self.n_kv_head == 0:
            self.n_kv_head = self.n_head
        # Auto-compute SwiGLU hidden dimension for parameter parity with 4x GELU MLP
        if self.intermediate_size == 0:
            raw = int(8 / 3 * self.n_embd)
            self.intermediate_size = ((raw + 63) // 64) * 64
        # Pad vocab for tensor core efficiency
        if self.vocab_size % 64 != 0:
            self.vocab_size = ((self.vocab_size // 64) + 1) * 64


# --- Main Model ---

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying (saves ~vocab_size * n_embd params, good for small models)
        self.transformer.wte.weight = self.lm_head.weight

        # Precompute RoPE frequencies (registered as buffer, moves with model)
        head_dim = config.n_embd // config.n_head
        self.register_buffer("freqs_cis",
            precompute_freqs_cis(head_dim, config.block_size, config.rope_theta),
            persistent=False
        )

        # Initialize weights
        self.apply(self._init_weights)
        # Scale residual projections per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('down_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self):
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Token embeddings only (no positional embeddings — RoPE handles position)
        x = self.transformer.drop(self.transformer.wte(idx))

        # Forward through transformer blocks
        for block in self.transformer.h:
            x = block(x, self.freqs_cis)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            # Soft-cap output logits for numerical stability
            if self.config.softcap > 0:
                logits = self.config.softcap * torch.tanh(logits / self.config.softcap)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference: only compute logits for the last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        """Reduce the block size (e.g. when loading a larger checkpoint for smaller inference)."""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        # Recompute RoPE frequencies for new block size
        head_dim = self.config.n_embd // self.config.n_head
        self.freqs_cis = precompute_freqs_cis(
            head_dim, block_size, self.config.rope_theta
        ).to(self.freqs_cis.device)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Configure AdamW optimizer with weight decay separation."""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # 2D params (weights, embeddings) get weight decay; 1D params (norms) don't
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def get_param_groups(self):
        """
        Get parameter groups for Muon + AdamW hybrid optimizer.
        Returns dict with 'muon_params', 'adamw_decay_params', 'adamw_nodecay_params'.
        """
        muon_params = []        # 2D non-embedding weights -> Muon
        adamw_decay_params = []  # Embedding weights -> AdamW with decay
        adamw_nodecay_params = [] # 1D params (norms) -> AdamW without decay

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2 and 'wte' not in name and 'lm_head' not in name:
                muon_params.append(param)
            elif param.dim() >= 2:
                adamw_decay_params.append(param)
            else:
                adamw_nodecay_params.append(param)

        return {
            'muon_params': muon_params,
            'adamw_decay_params': adamw_decay_params,
            'adamw_nodecay_params': adamw_nodecay_params,
        }

    def estimate_mfu(self, fwdbwd_per_iter, dt, peak_flops=None):
        """Estimate model flops utilization (MFU) relative to hardware peak FLOPS."""
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = peak_flops if peak_flops is not None else 312e12  # fallback to A100
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
