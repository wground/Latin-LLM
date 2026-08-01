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
- Optional looped/recurrent depth (Ouro): iterate the shared block stack n_loops
  times for effective depth n_layer*n_loops at no extra parameter cost, with input
  re-injection and per-step deep supervision

References:
1) Andrej Karpathy's nanoChat: https://github.com/karpathy/nanochat
2) Andrej Karpathy's nanoGPT: https://github.com/karpathy/nanoGPT
3) LLaMA architecture: RoPE, SwiGLU, RMSNorm, GQA
4) Muon optimizer: https://github.com/KellerJordan/Muon
5) Ouro (looped language models): https://arxiv.org/abs/2510.25741

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

    def forward(self, x, freqs_cis, cache=None, slot=None):
        """
        freqs_cis must already be sliced to this call's absolute positions, so incremental
        decoding (where T == 1 but the true position is far along) rotates correctly.

        When `cache` is given, the *unexpanded* K/V for these positions are appended to
        cache slot `slot` and the full history is returned. Caching before the GQA expansion
        is what makes GQA actually save memory: only n_kv_head heads are stored.
        """
        B, T, C = x.size()

        # Project Q, K, V
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)

        # QK normalization (before RoPE, per nanoChat)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE to Q and K
        q, k = apply_rotary_emb(q, k, freqs_cis)

        if cache is not None:
            k, v = cache.update(slot, k, v)

        kv_len = k.size(1)

        # GQA: expand KV heads to match Q heads by repeating
        if self.n_rep > 1:
            k = k.unsqueeze(3).expand(B, kv_len, self.n_kv_head, self.n_rep, self.head_dim)
            k = k.reshape(B, kv_len, self.n_head, self.head_dim)
            v = v.unsqueeze(3).expand(B, kv_len, self.n_kv_head, self.n_rep, self.head_dim)
            v = v.reshape(B, kv_len, self.n_head, self.head_dim)

        # Transpose for attention: (B, n_head, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # A single query attending to the whole cached history needs no mask; a full
        # prefill does. (Partial prefill against a non-empty cache is not used here.)
        is_causal = T > 1
        if T > 1 and kv_len != T:
            raise NotImplementedError(
                "Multi-token forward against a non-empty KV cache is not supported; "
                "reset the cache and re-prefill instead."
            )

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=is_causal
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if is_causal:
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

    def forward(self, x, freqs_cis, cache=None, slot=None):
        x = x + self.attn(self.ln_1(x), freqs_cis, cache=cache, slot=slot)
        x = x + self.mlp(self.ln_2(x))
        return x


# --- KV cache ---

class KVCache:
    """Per-(loop step, layer) key/value cache for incremental decoding.

    A looped model applies the same n_layer blocks n_loops times, and each application sees
    a different residual stream, so each needs its own cache slot: n_layer * n_loops total.
    K/V are stored unexpanded (n_kv_head heads) as (B, T, n_kv_head, head_dim).
    """

    def __init__(self, n_slots: int):
        self.n_slots = n_slots
        self.k = [None] * n_slots
        self.v = [None] * n_slots

    def __len__(self):
        """Number of positions currently cached."""
        return 0 if self.k[0] is None else self.k[0].size(1)

    def update(self, slot, k, v):
        if self.k[slot] is None:
            self.k[slot], self.v[slot] = k, v
        else:
            self.k[slot] = torch.cat([self.k[slot], k], dim=1)
            self.v[slot] = torch.cat([self.v[slot], v], dim=1)
        return self.k[slot], self.v[slot]

    def reset(self):
        self.k = [None] * self.n_slots
        self.v = [None] * self.n_slots


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

    # --- Looped / recurrent-depth computation (Ouro, arXiv:2510.25741) ---
    # The n_layer unique blocks are iterated n_loops times, giving an effective depth of
    # n_layer * n_loops with NO extra parameters. This buys capability in a
    # data-constrained corpus without the overfitting risk of widening the model.
    n_loops: int = 1                 # 1 = standard transformer (no recurrence)
    loop_input_injection: bool = True  # re-add the token embedding at each loop iteration
    per_step_loss: bool = True       # deep supervision: LM loss after every loop step
    # "linear" up-weights later steps, which matches the fact that inference only ever uses
    # the final readout (arXiv:2606.24898). "uniform" is the old behaviour; "final_only"
    # disables deep supervision's contribution to the objective entirely.
    loop_loss_weighting: str = "linear"  # "uniform" | "linear" | "final_only"

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
        # Scale residual projections per GPT-2 paper. Use EFFECTIVE depth
        # (n_layer * n_loops): with recurrence a shared block contributes to the residual
        # stream n_loops times, so scale init by the total number of residual additions to
        # keep the residual-stream variance controlled.
        effective_depth = config.n_layer * config.n_loops
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('down_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * effective_depth))

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

    def _readout(self, x, targets):
        """Apply final norm + (tied) head with optional soft-cap, and CE loss if targets."""
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            if self.config.softcap > 0:
                logits = self.config.softcap * torch.tanh(logits / self.config.softcap)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
            return logits, loss
        # Inference: only compute logits for the last position
        logits = self.lm_head(x[:, [-1], :])
        if self.config.softcap > 0:
            logits = self.config.softcap * torch.tanh(logits / self.config.softcap)
        return logits, None

    def new_kv_cache(self):
        """Allocate a cache sized for this model's effective depth."""
        return KVCache(self.config.n_layer * self.config.n_loops)

    def forward(self, idx, targets=None, cache=None, pos_offset=0):
        """
        Returns ``(logits, loss, aux)``.

        ``loss`` is the training objective (the loop-weighted average when deep supervision
        is on). ``aux['final_loss']`` is the cross-entropy of the FINAL readout alone --
        the only readout inference actually uses. Reporting and checkpoint selection must
        use ``final_loss``; optimizing uses ``loss``. Keeping the averaged objective and the
        final-readout metric distinct is what makes the reported number comparable to a
        conventional (non-looped) model's perplexity.
        """
        b, t = idx.size()
        cache_len = len(cache) if cache is not None else 0
        total = cache_len + t
        assert total <= self.config.block_size, \
            f"Cannot forward sequence of length {total}, block size is only {self.config.block_size}"

        # Token embeddings only (no positional embeddings — RoPE handles position)
        h0 = self.transformer.drop(self.transformer.wte(idx))
        # Slice RoPE frequencies at the ABSOLUTE positions of these tokens, so a cached
        # decode step at position 300 rotates as position 300 rather than position 0.
        start = pos_offset if cache is None else cache_len
        freqs = self.freqs_cis[start:start + t]
        n_loops = self.config.n_loops
        n_layer = self.config.n_layer

        x = h0
        step_losses = []
        for r in range(n_loops):
            # Re-inject the input at each iteration so the shared stack can keep
            # re-reading the prompt (Universal Transformer / Ouro style).
            if self.config.loop_input_injection and r > 0:
                x = x + h0
            for li, block in enumerate(self.transformer.h):
                x = block(x, freqs, cache=cache, slot=r * n_layer + li)
            # Deep supervision: read out and accumulate loss after every loop step.
            if targets is not None and self.config.per_step_loss and r < n_loops - 1:
                _, loss_r = self._readout(x, targets)
                step_losses.append(loss_r)

        # Final readout (always from the last loop step).
        logits, final_loss = self._readout(x, targets)

        if targets is None:
            return logits, None, {}

        if self.config.per_step_loss and step_losses:
            all_losses = step_losses + [final_loss]
            losses = torch.stack(all_losses)
            if self.config.loop_loss_weighting == "linear":
                # Up-weight later steps: only the final readout is used at inference, so
                # uniform averaging spends capacity on intermediate predictions that are
                # thrown away ("readout blind spot", arXiv:2606.24898).
                w = torch.arange(1, len(losses) + 1, device=losses.device, dtype=losses.dtype)
                loss = (losses * w).sum() / w.sum()
            elif self.config.loop_loss_weighting == "final_only":
                loss = final_loss
            else:  # uniform
                loss = losses.mean()
        else:
            loss = final_loss

        aux = {"final_loss": final_loss.detach(),
               "step_losses": [l.detach() for l in step_losses] + [final_loss.detach()]}
        return logits, loss, aux

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
        # Looping runs the same blocks n_loops times, so per-token compute scales with it.
        flops_per_token = (6 * N + 12 * L * H * Q * T) * cfg.n_loops
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = peak_flops if peak_flops is not None else 312e12  # fallback to A100
        mfu = flops_achieved / flops_promised
        return mfu

    @staticmethod
    def _filter_logits(logits, top_k=None, top_p=None, min_p=None):
        """Apply top-k / top-p (nucleus) / min-p truncation to a (B, vocab) logit tensor."""
        if top_k is not None and top_k > 0:
            k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, k)
            logits = logits.masked_fill(logits < v[:, [-1]], -float('Inf'))

        if min_p is not None and min_p > 0:
            # Keep tokens whose probability is at least min_p * p(most likely token).
            probs = F.softmax(logits, dim=-1)
            threshold = min_p * probs.max(dim=-1, keepdim=True).values
            logits = logits.masked_fill(probs < threshold, -float('Inf'))

        if top_p is not None and 0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            # Cumulative mass *excluding* each token: drop it only if the mass ahead of it
            # already covers top_p. Keeps the minimal set reaching top_p.
            cum_before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
            remove = cum_before > top_p
            remove[:, 0] = False
            logits = logits.masked_fill(remove.scatter(1, sorted_idx, remove), -float('Inf'))

        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None,
                 min_p=None, repetition_penalty=1.0, repetition_window=64,
                 eos_token_id=None, use_cache=True):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.

        Uses a KV cache: the prompt is encoded once, then each new token costs one
        single-position forward instead of re-running the whole prefix.

        Args:
            repetition_penalty: Divide logits of recently-used tokens by this value (1.0 = off,
                the default). It penalizes whole BPE pieces, including Latin inflectional
                endings, so leave it off for evaluation.
            eos_token_id: Stop once every sequence in the batch has emitted this token.
            use_cache: Set False to fall back to full-prefix recomputation (reference path).
        """
        block_size = self.config.block_size
        cache = self.new_kv_cache() if use_cache else None

        # Prefill on the (possibly cropped) prompt.
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        logits, _, _ = self(idx_cond, cache=cache)

        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)

        for _ in range(max_new_tokens):
            logits = logits[:, -1, :] / temperature

            # Penalize tokens that appeared in the recent window
            if repetition_penalty > 1.0 and idx.size(1) > 0:
                window = idx[:, -repetition_window:]
                for b in range(idx.size(0)):
                    seen = set(window[b].tolist())
                    for token_id in seen:
                        if logits[b, token_id] > 0:
                            logits[b, token_id] /= repetition_penalty
                        else:
                            logits[b, token_id] *= repetition_penalty

            logits = self._filter_logits(logits, top_k=top_k, top_p=top_p, min_p=min_p)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            if eos_token_id is not None:
                # Once a sequence is done, keep emitting EOS so the batch stays rectangular.
                idx_next = torch.where(finished.unsqueeze(1),
                                       torch.full_like(idx_next, eos_token_id), idx_next)
                finished = finished | (idx_next.squeeze(1) == eos_token_id)

            idx = torch.cat((idx, idx_next), dim=1)

            if eos_token_id is not None and bool(finished.all()):
                break

            if cache is not None:
                # The cache holds absolute positions, so it must be rebuilt when the window
                # slides past block_size rather than silently rotating tokens wrongly.
                if len(cache) >= block_size:
                    cache.reset()
                    logits, _, _ = self(idx[:, -block_size:], cache=cache)
                else:
                    logits, _, _ = self(idx_next, cache=cache)
            else:
                idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
                logits, _, _ = self(idx_cond)

        return idx
