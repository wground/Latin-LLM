"""
LatinLLM Training Script
Trains a modernized GPT model on Latin text corpus using system-optimized configurations.
Uses detect_system.py output for optimal hardware configuration.

Architecture: RoPE, SwiGLU, GQA, RMSNorm, QK-norm (nanoChat-inspired), optional
              looped/recurrent depth (Ouro, arXiv:2510.25741)
Optimizer: Muon (with weight decay) + AdamW hybrid (CUDA) or AdamW (MPS/CPU)
LR Schedule: Warmup-Stable-Decay (WSD)
Checkpoints: ckpt.pt (latest, resumable) + ckpt_best.pt (best val loss) + ckpt_final.pt

Usage:
    python3 train_latin.py --init scratch [--batch_size SIZE] [--max_iters ITERS]
    python3 train_latin.py --init resume  [--out-dir DIR]
    python3 train_latin.py [--n_loops N] [--n_layer L] [--n_embd D] [--dropout P]

Note: --init is REQUIRED to be explicit. Earlier versions inferred "resume" from the mere
presence of out-dir/ckpt.pt, so any run -- including a smoke test -- would silently resume
the real training run and overwrite its checkpoint.

Author: Willow Groundwater-Schuldt & Claude
"""

import os
import time
import math
import json
import pickle
import argparse
import shutil
import subprocess
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import paths
from model import GPTConfig, GPT


# --- Muon Optimizer ---

class Muon(torch.optim.Optimizer):
    """
    Muon optimizer for 2D weight matrices.
    Orthogonalizes gradient updates using Newton-Schulz iterations for ~2x
    compute efficiency over AdamW on matrix parameters.

    Only used for attention/MLP weight matrices. Embeddings and 1D params use AdamW.

    Reference: https://github.com/KellerJordan/Muon
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5,
                 weight_decay=0.01):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @staticmethod
    def _orthogonalize(G, steps=5):
        """Newton-Schulz iteration to find nearest orthogonal matrix."""
        a, b, c = (3.4445, -4.7750, 2.0315)
        compute_dtype = torch.bfloat16 if G.is_cuda else torch.float32
        X = G.to(compute_dtype)
        X = X / (X.norm() + 1e-7)
        for _ in range(steps):
            A = X @ X.T
            X = a * X + b * (A @ X) + c * (A @ (A @ X))
        return X.to(G.dtype)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)

                if group['nesterov']:
                    update = g + momentum * buf
                else:
                    update = buf

                # Orthogonalize 2D weight updates
                if update.ndim >= 2:
                    update = self._orthogonalize(update, steps=group['ns_steps'])
                    update *= max(1, update.shape[0] / update.shape[1]) ** 0.5

                # Decoupled weight decay (AdamW-style). Per "Muon is Scalable for LLM
                # Training" (arXiv:2502.16982), decay is needed for stable scaling.
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                p.add_(update, alpha=-lr)


# --- Configuration Loading ---

def load_system_config(config_path=None) -> Dict[str, Any]:
    """Load system configuration from detect_system.py output."""
    config_path = Path(config_path) if config_path is not None else paths.SYSTEM_CONFIG
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found!")
        print("Run 'python3 detect_system.py' first to generate system config.")
        print("Using default CPU configuration...")
        return {
            "recommended_config": {
                "device": "cpu",
                "dtype": "float32",
                "compile": False,
                "backend": "cpu",
                "multi_gpu": False,
                "recommended_batch_size": 4,
                "recommended_block_size": 256,
                "use_fused_adamw": False,
                "enable_tf32": False
            }
        }

    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"Loaded system config from {config_path}")
    return config


def load_tokenizer_config(data_dir=None) -> Dict[str, Any]:
    """Load custom tokenizer configuration and metadata."""
    data_dir = Path(data_dir) if data_dir is not None else paths.DATA_DIR
    meta_path = data_dir / paths.META_NAME

    if not meta_path.exists():
        print(f"Tokenizer metadata not found at {meta_path}")
        print("You must run 'python3 prepare_corpus.py' first to build the dataset")
        exit(1)

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    print(f"Loaded custom Latin tokenizer metadata")
    print(f"   Vocabulary size: {meta['vocab_size']}")
    print(f"   Tokenizer type: {meta['tokenizer_config']['type']}")
    if "data_stats" in meta:
        print(f"   Training tokens: {meta['data_stats']['train_tokens']:,}")
        print(f"   Validation tokens: {meta['data_stats']['val_tokens']:,}")
    if not meta.get("eos_separated", False):
        print("   ⚠️  This dataset has NO document separators (<|endoftext|>). The model "
              "cannot learn where documents end. Rebuild with prepare_corpus.py.")

    return {
        "vocab_size": meta["vocab_size"],
        "tokenizer_type": meta["tokenizer_config"]["type"],
        "data_stats": meta.get("data_stats", {}),
        "meta": meta,
    }


def setup_training_config(system_config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Setup training configuration based on system capabilities and user args."""
    rec_config = system_config["recommended_config"]

    tokenizer_config = load_tokenizer_config(args.data_dir)
    vocab_size = tokenizer_config["vocab_size"]
    data_stats = tokenizer_config.get("data_stats", {})
    train_tokens = data_stats.get("train_tokens", 0)

    # Adaptive model configuration based on vocab size
    if vocab_size > 12000:
        n_layer = 8
        n_head = 8
        n_kv_head = 4  # GQA 2:1 ratio
        n_embd = 512
        batch_size_multiplier = 0.8
    elif vocab_size > 8000:
        n_layer = 7
        n_head = 7
        n_kv_head = 7  # MHA (no GQA for odd head counts)
        n_embd = 448
        batch_size_multiplier = 0.9
    else:
        n_layer = 6
        n_head = 6
        n_kv_head = 3  # GQA 2:1 ratio
        n_embd = 384
        batch_size_multiplier = 1.0

    # Adaptive training parameters based on dataset size
    if train_tokens > 10_000_000:  # Large dataset (10M+)
        eval_interval = 500
        gradient_accumulation_steps = 4
        warmup_iters = 500
    elif train_tokens > 1_000_000:  # Medium dataset
        eval_interval = 350
        gradient_accumulation_steps = 5
        warmup_iters = 400
    else:
        eval_interval = 250
        gradient_accumulation_steps = 6
        warmup_iters = 300

    config = {
        # I/O Configuration
        "out_dir": str(args.out_dir),
        "data_dir": str(args.data_dir),
        "eval_interval": eval_interval,
        "log_interval": 10,
        "eval_iters": 150,
        "eval_seed": 1337,   # fixed: evaluation windows must not move between runs
        # "weighted" applies the corpus tier multipliers via sampling (see _get_doc_sampler);
        # "uniform" ignores them, which is the control condition for a mixture ablation.
        "sampling": "weighted",
        "eval_only": False,
        "always_save_checkpoint": True,

        # Dataset Configuration
        "dataset": "latin",
        "data_stats": data_stats,
        "gradient_accumulation_steps": gradient_accumulation_steps,

        # Model Configuration (modernized architecture)
        "n_layer": n_layer,
        "n_head": n_head,
        "n_kv_head": n_kv_head,
        "n_embd": n_embd,
        "intermediate_size": 0,  # Auto-compute SwiGLU hidden dim
        "dropout": 0.05,  # Lowered from 0.15: data-rich/repeated regime over-regularizes at 0.15
        "softcap": 15.0,
        "rope_theta": 10000.0,

        # Looped / recurrent-depth computation (Ouro-style, arXiv:2510.25741).
        # n_loops > 1 iterates the shared block stack to add effective depth WITHOUT
        # adding parameters — the right lever for a data-constrained corpus.
        "n_loops": 1,                 # 1 = standard transformer (no recurrence)
        "loop_input_injection": True, # re-inject token embedding at each loop iteration
        "per_step_loss": True,        # deep supervision: LM loss after every loop step
        # Inference only ever uses the final readout, so weight later steps higher rather
        # than averaging all readouts equally (arXiv:2606.24898).
        "loop_loss_weighting": "linear",  # "uniform" | "linear" | "final_only"

        # Optimizer Configuration
        "learning_rate": 3e-4 if vocab_size > 12000 else 4e-4,
        "muon_lr": 0.02,  # Muon base LR (only used on CUDA)
        "muon_weight_decay": 0.01,  # decoupled weight decay for Muon matrices
        "max_iters": args.max_iters,
        "weight_decay": 0.05,
        "beta1": 0.9,
        "beta2": 0.95,
        "grad_clip": 1.0,

        # WSD Learning Rate Schedule
        "decay_lr": True,
        "warmup_iters": warmup_iters,
        "min_lr": 1e-4,
        "decay_fraction": 0.3,  # Final 30% of training is decay phase

        # Weights & Biases
        "wandb_log": False,
        "wandb_project": "latin-llm",
        "wandb_run_name": f"latin-gpt-v{vocab_size // 1000}k",

        # Early Stopping Configuration
        # Default OFF for the lowest-loss objective: the WSD decay phase delivers the
        # biggest val-loss drop, so we want to always run through it. When enabled, the
        # patience counter is only armed during the decay phase (see training loop), so a
        # plateau in the long constant-LR stable phase can't truncate the run early.
        "early_stopping": False,
        "patience": 15,
        "min_delta": 0.005,

        # DDP Configuration
        "backend": "nccl" if rec_config["backend"] == "cuda" else "gloo",
    }

    # Apply system-optimized settings with batch size adjustment
    optimal_batch_size = int(rec_config["recommended_batch_size"] * batch_size_multiplier)
    config.update({
        "device": rec_config["device"],
        "dtype": rec_config["dtype"],
        "compile": rec_config["compile"],
        "batch_size": args.batch_size or optimal_batch_size,
        "block_size": rec_config["recommended_block_size"],
        "use_fused_adamw": rec_config["use_fused_adamw"],
        "enable_tf32": rec_config["enable_tf32"],
    })

    config["vocab_size"] = vocab_size

    # Initialization mode is EXPLICIT. It used to be inferred from the mere existence of
    # out-dir/ckpt.pt, which meant any invocation -- including a smoke test -- silently
    # resumed the real run and then overwrote its checkpoint.
    config["init_from"] = args.init

    return config


# --- Data Loading ---

# Cache one read-only memmap per split. The previous code re-opened the .bin file on
# every micro-step, which is a measurable stall in the training loop; the memmap header
# only needs to be parsed once and the OS page cache handles the actual reads.
_DATA_CACHE: Dict[str, np.memmap] = {}
# Fixed evaluation windows, built once per (split, block_size, batch_size, eval_iters).
_EVAL_WINDOWS: Dict[str, np.ndarray] = {}


def _get_split_data(split: str, config: Dict[str, Any]) -> np.memmap:
    """Memmap for a split, resolved from the configured data dir (never from cwd)."""
    if split not in _DATA_CACHE:
        data_dir = Path(config.get("data_dir") or paths.DATA_DIR)
        filename = data_dir / f"{split}.bin"
        if not filename.exists():
            raise FileNotFoundError(
                f"Data file {filename} not found. Run prepare_corpus.py first."
            )
        _DATA_CACHE[split] = np.memmap(filename, dtype=np.uint16, mode='r')
    return _DATA_CACHE[split]


def _windows_to_batch(data: np.memmap, starts: np.ndarray, block_size: int,
                      config: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather (x, y) for the given window start offsets."""
    idx = starts[:, None] + np.arange(block_size + 1, dtype=np.int64)[None, :]
    seq = torch.from_numpy(data[idx].astype(np.int64))
    # .contiguous(): the slices below are non-contiguous views; the model's loss does
    # targets.view(-1), and pin_memory()/non_blocking H2D both want contiguous tensors.
    x = seq[:, :-1].contiguous()
    y = seq[:, 1:].contiguous()

    if config["device"] == 'cuda':
        x = x.pin_memory().to(config["device"], non_blocking=True)
        y = y.pin_memory().to(config["device"], non_blocking=True)
    else:
        x = x.to(config["device"])
        y = y.to(config["device"])

    return x, y


_SAMPLER_CACHE: Dict[str, Any] = {}


def _get_doc_sampler(config: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
    """Load the per-document index and weights written by prepare_corpus.py.

    Corpus weighting used to be physical: a tier-15 document was written into train.bin
    fifteen times. Now train.bin holds each document once and the mixture is applied here,
    by sampling window starts from documents in proportion to their weight. Same effective
    exposure, but the mixture is a run-time choice rather than a property of a 1.3 GB file.
    """
    if 'doc' in _SAMPLER_CACHE:
        return _SAMPLER_CACHE['doc']

    data_dir = Path(config.get("data_dir") or paths.DATA_DIR)
    index_f, weights_f = data_dir / "train_index.npy", data_dir / "train_weights.npy"
    if not (index_f.exists() and weights_f.exists()):
        _SAMPLER_CACHE['doc'] = None
        return None

    index = np.load(index_f)          # (n_docs, 2): [start_offset, length]
    weights = np.load(weights_f).astype(np.float64)
    if config.get("sampling", "weighted") == "uniform":
        weights = np.ones_like(weights)
    # Weight by tier * length: a window start is uniform within a document, so without the
    # length factor a one-page hymn would be sampled as often as a 900-page series.
    mass = weights * index[:, 1]
    total = mass.sum()
    _SAMPLER_CACHE['doc'] = {
        'starts': index[:, 0],
        'lengths': index[:, 1],
        'cdf': np.cumsum(mass / total),
    }
    return _SAMPLER_CACHE['doc']


def get_batch(split: str, config: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a batch of contiguous windows from the corpus.

    The memmap is cached across calls and the batch is built with a single vectorized
    gather (no Python per-sample loop). The next-batch copy is issued non-blocking on
    CUDA; because the training loop fetches the next batch before calling backward(),
    that copy overlaps compute.
    """
    data = _get_split_data(split, config)
    block_size = config["block_size"]
    batch_size = config["batch_size"]
    high = len(data) - block_size - 1

    sampler = _get_doc_sampler(config) if split == 'train' else None
    if sampler is not None:
        # torch RNG keeps the data stream reproducible and resumable.
        u = torch.rand(batch_size).numpy()
        doc = np.searchsorted(sampler['cdf'], u)
        doc = np.clip(doc, 0, len(sampler['starts']) - 1)
        # Uniform start within the chosen document; windows may run past its EOS into the
        # next document, which is exactly how the model learns document transitions.
        within = (torch.rand(batch_size).numpy() * sampler['lengths'][doc]).astype(np.int64)
        starts = np.clip(sampler['starts'][doc] + within, 0, high)
    else:
        starts = torch.randint(high, (batch_size,)).numpy()

    return _windows_to_batch(data, starts, block_size, config)


def get_eval_windows(split: str, config: Dict[str, Any]) -> np.ndarray:
    """Immutable evaluation window offsets for a split.

    Drawn once from a dedicated ``np.random.Generator`` seeded with ``eval_seed``. This
    matters twice over: evaluation now scores the SAME text every time (so a "best"
    checkpoint is chosen on a real improvement rather than on which windows it happened to
    draw), and it no longer consumes the global torch RNG, so evaluating does not shift the
    subsequent training batches.
    """
    key = f"{split}:{config['block_size']}:{config['batch_size']}:{config['eval_iters']}:{config['eval_seed']}"
    if key not in _EVAL_WINDOWS:
        data = _get_split_data(split, config)
        n_windows = config["eval_iters"] * config["batch_size"]
        high = len(data) - config["block_size"]
        if high <= 0:
            raise ValueError(f"Split '{split}' is shorter than block_size {config['block_size']}")
        rng = np.random.default_rng(config["eval_seed"] + (0 if split == "train" else 1))
        _EVAL_WINDOWS[key] = rng.integers(0, high, size=n_windows, dtype=np.int64)
    return _EVAL_WINDOWS[key]


def build_provenance(config: Dict[str, Any]) -> Dict[str, Any]:
    """Hashes identifying the data, tokenizer and code behind a run.

    The .bin files are multi-GB, so they are fingerprinted by size plus a hash of their
    first 64 MB -- enough to detect "this is a different dataset" without a full read.
    """
    data_dir = Path(config.get("data_dir") or paths.DATA_DIR)
    prov: Dict[str, Any] = {}

    for name, key in ((paths.TRAIN_BIN, 'train_bin'), (paths.VAL_BIN, 'val_bin')):
        f = data_dir / name
        if f.exists():
            prov[f'{key}_sha1'] = paths.file_sha1(f, max_bytes=64 << 20)
            prov[f'{key}_bytes'] = f.stat().st_size

    try:
        meta_path = data_dir / paths.META_NAME
        with open(meta_path, 'rb') as fh:
            meta = pickle.load(fh)
        vocab_file, merges_file = paths.tokenizer_files(meta, meta_path)
        prov['tokenizer_sha1'] = paths.dir_sha1([vocab_file, merges_file])
    except Exception:
        pass

    try:
        prov['git_rev'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=str(paths.SRC_DIR),
            stderr=subprocess.DEVNULL, text=True).strip()
        prov['git_dirty'] = bool(subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=str(paths.SRC_DIR),
            stderr=subprocess.DEVNULL, text=True).strip())
    except Exception:
        pass

    return prov


@torch.no_grad()
def estimate_loss(model, config: Dict[str, Any], ctx) -> Dict[str, float]:
    """Estimate loss on fixed train/val windows.

    Returns both the training objective (``<split>``) and the final-readout cross-entropy
    (``<split>_final``). For a looped model these differ: the objective averages the loop
    readouts, but only the final readout is used at inference, so ``*_final`` is the number
    that is comparable to a conventional model's loss. Also reports bits/byte, which stays
    comparable across tokenizer changes.
    """
    out = {}
    model.eval()

    bytes_per_token = config.get("bytes_per_token")

    for split in ['train', 'val']:
        data = _get_split_data(split, config)
        windows = get_eval_windows(split, config)
        batch_size = config["batch_size"]
        losses = torch.zeros(config["eval_iters"])
        final_losses = torch.zeros(config["eval_iters"])

        for k in range(config["eval_iters"]):
            starts = windows[k * batch_size:(k + 1) * batch_size]
            X, Y = _windows_to_batch(data, starts, config["block_size"], config)
            with ctx:
                _, loss, aux = model(X, Y)
            losses[k] = loss.item()
            final_losses[k] = aux.get("final_loss", loss).item()

        out[split] = losses.mean()
        out[f"{split}_final"] = final_losses.mean()
        if bytes_per_token:
            # bits/byte = CE_nats / ln(2) / bytes_per_token
            out[f"{split}_bpb"] = float(final_losses.mean()) / math.log(2) / bytes_per_token

    model.train()
    return out


# --- Hardware FLOPS Estimation ---

def get_hardware_peak_flops(system_config: Dict[str, Any], dtype: str) -> float:
    """Estimate hardware peak FLOPS based on detected GPU."""
    default_flops = 312e12  # A100 bfloat16 fallback

    try:
        pytorch_info = system_config.get("pytorch", {})
        gpu_devices = pytorch_info.get("gpu_devices", [])

        if not gpu_devices or "device" not in system_config.get("recommended_config", {}):
            return default_flops

        device_type = system_config["recommended_config"]["device"]

        # Apple Silicon — per-chip table keyed on (family, variant).
        # Values are FP16 peak TFLOPS for the reference core count; scaled
        # linearly by actual gpu_cores when available.
        if device_type == "mps" and gpu_devices:
            rec_cfg = system_config.get("recommended_config", {})
            family = (rec_cfg.get("apple_chip_family") or "").upper()
            variant = (rec_cfg.get("apple_chip_variant") or "").capitalize()
            cores = rec_cfg.get("apple_gpu_cores")
            gpu_name = gpu_devices[0].get("name", "").lower()

            # (fp16_tflops, reference_gpu_cores)
            apple_flops_table = {
                ("M1", ""):      (5.2,  8),
                ("M1", "Pro"):   (10.4, 16),
                ("M1", "Max"):   (20.8, 32),
                ("M1", "Ultra"): (42.0, 64),
                ("M2", ""):      (7.2,  10),
                ("M2", "Pro"):   (13.6, 19),
                ("M2", "Max"):   (27.2, 38),
                ("M2", "Ultra"): (54.4, 76),
                ("M3", ""):      (8.2,  10),
                ("M3", "Pro"):   (14.8, 18),
                ("M3", "Max"):   (28.4, 40),
                ("M3", "Ultra"): (56.8, 80),
                ("M4", ""):      (9.2,  10),
                ("M4", "Pro"):   (17.0, 20),
                ("M4", "Max"):   (35.8, 40),
                ("M5", ""):      (10.4, 10),
                ("M5", "Pro"):   (19.0, 20),
                ("M5", "Max"):   (40.0, 40),
            }

            entry = apple_flops_table.get((family, variant))

            # Name-based fallback if sysctl didn't yield family/variant
            if entry is None:
                for (fam, var), val in apple_flops_table.items():
                    key = f"{fam} {var}".strip().lower()
                    if key and key in gpu_name:
                        entry = val
                        break

            if entry is not None:
                fp16_tflops, ref_cores = entry
                if cores and ref_cores:
                    fp16_tflops *= cores / ref_cores
                fp16_flops = fp16_tflops * 1e12
                # MPS uses FP16 autocast; FP32 path is ~half throughput.
                return fp16_flops if dtype == "float16" else fp16_flops / 2

            return 10e12  # Unknown Apple Silicon chip fallback

        # CUDA GPUs
        elif device_type == "cuda" and gpu_devices:
            gpu_name = gpu_devices[0].get("name", "").lower()

            gpu_flops = {
                # RTX 50 series (Blackwell)
                "rtx 5090": 200e12 if dtype == "bfloat16" else 107e12,
                "rtx 5080": 120e12 if dtype == "bfloat16" else 60e12,
                "rtx 5070 ti": 100e12 if dtype == "bfloat16" else 50e12,
                "rtx 5070": 80e12 if dtype == "bfloat16" else 40e12,

                # RTX 40 series (Ada Lovelace)
                "rtx 4090": 165e12 if dtype == "bfloat16" else 83e12,
                "rtx 4080": 120e12 if dtype == "bfloat16" else 60e12,
                "rtx 4070 ti": 93e12 if dtype == "bfloat16" else 46e12,
                "rtx 4070": 90e12 if dtype == "bfloat16" else 45e12,
                "rtx 4060 ti": 44e12 if dtype == "bfloat16" else 22e12,
                "rtx 4060": 30e12 if dtype == "bfloat16" else 15e12,

                # RTX 30 series (Ampere)
                "rtx 3090 ti": 80e12 if dtype in ["bfloat16", "float16"] else 40e12,
                "rtx 3090": 71e12 if dtype in ["bfloat16", "float16"] else 35e12,
                "rtx 3080 ti": 68e12 if dtype in ["bfloat16", "float16"] else 34e12,
                "rtx 3080": 58e12 if dtype in ["bfloat16", "float16"] else 29e12,
                "rtx 3070 ti": 43e12 if dtype in ["bfloat16", "float16"] else 22e12,
                "rtx 3070": 40e12 if dtype in ["bfloat16", "float16"] else 20e12,
                "rtx 3060 ti": 32e12 if dtype in ["bfloat16", "float16"] else 16e12,
                "rtx 3060": 25e12 if dtype in ["bfloat16", "float16"] else 13e12,

                # Data center GPUs
                "h100": 756e12 if dtype == "bfloat16" else 378e12,
                "a100": 312e12 if dtype == "bfloat16" else 156e12,
                "a40": 150e12 if dtype == "bfloat16" else 75e12,
                "a30": 165e12 if dtype == "bfloat16" else 82e12,
                "v100": 125e12 if dtype == "float16" else 62e12,

                # AMD GPUs (ROCm) - RDNA 4
                "rx 9090 xt": 200e12 if dtype == "float16" else 100e12,
                "rx 9080 xt": 160e12 if dtype == "float16" else 80e12,
                "rx 9070 xt": 97e12 if dtype == "float16" else 48e12,
                "rx 9070": 72e12 if dtype == "float16" else 36e12,
                "rx 9060 xt": 45e12 if dtype == "float16" else 22e12,
                "rx 9060": 35e12 if dtype == "float16" else 18e12,

                # AMD RDNA 3
                "rx 7900 xtx": 122e12 if dtype == "float16" else 61e12,
                "rx 7900 xt": 103e12 if dtype == "float16" else 51e12,
                "rx 7800 xt": 75e12 if dtype == "float16" else 37e12,
                "rx 7700 xt": 60e12 if dtype == "float16" else 30e12,
                "rx 7600 xt": 40e12 if dtype == "float16" else 20e12,
                "rx 7600": 32e12 if dtype == "float16" else 16e12,

                # AMD RDNA 2
                "rx 6950 xt": 46e12 if dtype == "float16" else 23e12,
                "rx 6900 xt": 46e12 if dtype == "float16" else 23e12,
                "rx 6800 xt": 40e12 if dtype == "float16" else 20e12,
                "rx 6700 xt": 26e12 if dtype == "float16" else 13e12,
                "rx 6600 xt": 20e12 if dtype == "float16" else 10e12,

                # AMD data center
                "mi250": 180e12 if dtype == "bfloat16" else 90e12,
                "mi210": 180e12 if dtype == "bfloat16" else 90e12,
                "mi100": 185e12 if dtype == "bfloat16" else 92e12,
            }

            for gpu_key, flops in gpu_flops.items():
                if gpu_key in gpu_name:
                    return flops

            # Fallback based on compute capability
            compute_cap = gpu_devices[0].get("compute_capability", "")
            memory_gb = gpu_devices[0].get("memory_total", 0) / (1024 ** 3)

            if compute_cap.startswith("8."):
                if memory_gb > 20:
                    return 200e12 if dtype in ["bfloat16", "float16"] else 100e12
                elif memory_gb > 10:
                    return 100e12 if dtype in ["bfloat16", "float16"] else 50e12
                else:
                    return 60e12 if dtype in ["bfloat16", "float16"] else 30e12
            elif compute_cap.startswith("7."):
                return 60e12 if dtype == "float16" else 30e12
            else:
                return 40e12 if dtype == "float16" else 20e12

        return default_flops

    except Exception:
        return default_flops


# --- Learning Rate Schedule ---

def get_lr(it: int, config: Dict[str, Any]) -> float:
    """Warmup-Stable-Decay (WSD) learning rate schedule."""
    warmup_iters = config["warmup_iters"]
    max_iters = config["max_iters"]
    max_lr = config["learning_rate"]
    min_lr = config["min_lr"]
    decay_start = int(max_iters * (1.0 - config["decay_fraction"]))

    # Phase 1: Linear warmup
    if it < warmup_iters:
        return max_lr * (it + 1) / (warmup_iters + 1)

    # Phase 2: Stable (constant at max LR)
    if it < decay_start:
        return max_lr

    # Phase 3: Linear decay. Clamp progress to [0, 1] so the LR can never overshoot
    # max_lr or go below min_lr (a negative LR corrupts weights) — e.g. when resuming a
    # checkpoint whose iter_num already exceeds max_iters.
    progress = (it - decay_start) / max(1, max_iters - decay_start)
    progress = min(1.0, max(0.0, progress))
    return min_lr + (max_lr - min_lr) * (1.0 - progress)


# --- Training Visualization ---

def plot_training_metrics(metrics: Dict, out_dir: str):
    """Plot training metrics and save to file."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        print("Skipping training visualization.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('LatinLLM Training Metrics', fontsize=16, fontweight='bold')

    # 1. Loss curves
    ax = axes[0, 0]
    if metrics['train_losses']:
        iters, losses = zip(*metrics['train_losses'])
        ax.plot(iters, losses, label='Train Loss', alpha=0.8, color='#2196F3')
    if metrics['val_losses']:
        iters, losses = zip(*metrics['val_losses'])
        ax.plot(iters, losses, label='Val Loss', alpha=0.9, linewidth=2, color='#F44336')
        # Mark best val loss
        best_idx = losses.index(min(losses))
        ax.scatter([iters[best_idx]], [losses[best_idx]], color='#4CAF50', s=100, zorder=5, label=f'Best: {min(losses):.4f}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Learning rate schedule
    ax = axes[0, 1]
    if metrics['learning_rates']:
        iters, lrs = zip(*metrics['learning_rates'])
        ax.plot(iters, lrs, color='#FF9800', linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate (WSD Schedule)')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-4, -4))

    # 3. MFU
    ax = axes[1, 0]
    if metrics['mfu_values']:
        iters, mfus = zip(*metrics['mfu_values'])
        mfus_pct = [m * 100 for m in mfus if m > 0]
        iters_valid = [i for i, m in zip(iters, mfus) if m > 0]
        if mfus_pct:
            ax.plot(iters_valid, mfus_pct, color='#4CAF50', alpha=0.7)
            avg_mfu = sum(mfus_pct) / len(mfus_pct)
            ax.axhline(y=avg_mfu, color='#4CAF50', linestyle='--', alpha=0.5, label=f'Avg: {avg_mfu:.1f}%')
            ax.legend()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('MFU (%)')
    ax.set_title('Model FLOPs Utilization')
    ax.grid(True, alpha=0.3)

    # 4. Iteration time
    ax = axes[1, 1]
    if metrics['iter_times']:
        iters, times = zip(*metrics['iter_times'])
        times_ms = [t * 1000 for t in times]
        ax.plot(iters, times_ms, color='#9C27B0', alpha=0.5, linewidth=0.8)
        if len(times_ms) > 10:
            # Smoothed line
            window = min(50, len(times_ms) // 4)
            if window > 1:
                smoothed = [sum(times_ms[max(0, i - window):i + 1]) / min(i + 1, window) for i in range(len(times_ms))]
                ax.plot(iters, smoothed, color='#9C27B0', alpha=0.9, linewidth=2, label='Smoothed')
                ax.legend()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Iteration Time')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, 'training_metrics.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training visualization saved to {plot_path}")


# --- Main Training ---

def main():
    parser = argparse.ArgumentParser(description="Train LatinLLM model")
    parser.add_argument("--config", default=None, help="System config file")
    parser.add_argument("--init", choices=["scratch", "resume", "finetune"], required=True,
                        help="scratch: new model. resume: continue a run (iteration count, "
                             "LR schedule, RNG and optimizer state all restored). finetune: "
                             "load the weights but restart the schedule, accepting the "
                             "checkpoint's architecture.")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--block_size", type=int, help="Override context length")
    parser.add_argument("--eval_iters", type=int, help="Override number of eval batches")
    parser.add_argument("--eval_interval", type=int, help="Override iterations between evals")
    parser.add_argument("--sampling", choices=["weighted", "uniform"],
                        help="weighted: apply corpus tier multipliers via sampling. "
                             "uniform: ignore them (control condition).")
    parser.add_argument("--max_iters", type=int, default=75000, help="Maximum training iterations")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--no_compile", action="store_true",
                        help="Disable torch.compile (useful on Windows/Triton issues or debugging)")
    # Looped / recurrent-depth experiment overrides (Ouro, arXiv:2510.25741)
    parser.add_argument("--n_loops", type=int, help="Recurrence count; effective depth = n_layer * n_loops")
    parser.add_argument("--no_per_step_loss", action="store_true", help="Disable deep supervision across loop steps")
    parser.add_argument("--loop_loss_weighting", choices=["uniform", "linear", "final_only"],
                        help="How to weight per-step losses")
    # Model-size overrides (Tier 3): size by hand instead of the vocab-derived defaults
    parser.add_argument("--n_layer", type=int, help="Override number of unique transformer layers")
    parser.add_argument("--n_embd", type=int, help="Override embedding dimension")
    parser.add_argument("--n_head", type=int, help="Override number of attention heads")
    parser.add_argument("--n_kv_head", type=int, help="Override number of KV heads (GQA)")
    parser.add_argument("--dropout", type=float, help="Override dropout")
    paths.add_path_args(parser)
    args = parser.parse_args()

    print("LatinLLM Training Script")
    print("=" * 50)

    # Load system configuration
    system_config = load_system_config(args.config)
    config = setup_training_config(system_config, args)

    # Apply explicit CLI overrides (take precedence over auto-derived defaults).
    for key in ("n_loops", "n_layer", "n_embd", "n_head", "n_kv_head", "dropout",
                "loop_loss_weighting", "block_size", "eval_iters", "eval_interval",
                "sampling"):
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val
    if args.no_per_step_loss:
        config["per_step_loss"] = False
    if args.device:
        config["device"] = args.device
        if args.device == 'cpu':
            config["dtype"] = 'float32'
            config["compile"] = False

    # bytes/token lets us report bits/byte, which stays comparable if the tokenizer changes.
    stats = config.get("data_stats", {})
    if stats.get("train_bytes") and stats.get("train_tokens"):
        config["bytes_per_token"] = stats["train_bytes"] / stats["train_tokens"]

    Path(config["out_dir"]).mkdir(parents=True, exist_ok=True)

    if args.wandb:
        config["wandb_log"] = True

    if args.no_compile:
        config["compile"] = False

    # Hardware peak FLOPS for MFU calculation
    peak_flops = get_hardware_peak_flops(system_config, config["dtype"])
    config["peak_flops"] = peak_flops

    # Determine if we can use Muon (CUDA only)
    use_muon = config["device"] == "cuda"

    # Print configuration
    print(f"Device: {config['device']} ({config['dtype']})")
    print(f"Model: {config['n_layer']} layers, {config['n_head']} heads ({config['n_kv_head']} KV), {config['n_embd']} embd")
    if config["n_loops"] > 1:
        eff = config['n_layer'] * config['n_loops']
        print(f"Looped: n_loops={config['n_loops']} -> effective depth {eff} "
              f"(per_step_loss={config['per_step_loss']}, weighting={config['loop_loss_weighting']})")
    print(f"Architecture: RoPE + SwiGLU + GQA + RMSNorm + QK-norm")
    print(f"Optimizer: {'Muon + AdamW hybrid' if use_muon else 'AdamW'}")
    print(f"LR Schedule: WSD (warmup={config['warmup_iters']}, decay={config['decay_fraction']*100:.0f}%)")
    print(f"Hardware peak FLOPS: {peak_flops / 1e12:.1f} TFLOPS ({config['dtype']})")
    print(f"Training: {config['batch_size']} batch size, {config['block_size']} context length")
    print(f"Max iterations: {config['max_iters']}")
    print(f"Compilation: {'enabled' if config['compile'] else 'disabled'}")

    # DDP setup
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend=config["backend"])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'{config["device"]}:{ddp_local_rank}' if config["device"] == 'cuda' else config["device"]
        if config["device"] == 'cuda':
            torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        assert config["gradient_accumulation_steps"] % ddp_world_size == 0
        config["gradient_accumulation_steps"] //= ddp_world_size
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        device = config["device"]

    tokens_per_iter = config["gradient_accumulation_steps"] * ddp_world_size * config["batch_size"] * config["block_size"]
    print(f"Tokens per iteration: {tokens_per_iter:,}")

    if master_process:
        os.makedirs(config["out_dir"], exist_ok=True)

    torch.manual_seed(1337 + seed_offset)

    # Hardware optimizations
    if config["device"] == 'cuda':
        if config["enable_tf32"]:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # Mixed precision context
    device_type = config["device"] if config["device"] in ('cuda', 'mps') else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config["dtype"]]
    if device_type == 'cpu':
        ctx = nullcontext()
    else:
        ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Initialize model
    model_args = dict(
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_kv_head=config["n_kv_head"],
        n_embd=config["n_embd"],
        intermediate_size=config["intermediate_size"],
        block_size=config["block_size"],
        vocab_size=config["vocab_size"],
        dropout=config["dropout"],
        softcap=config["softcap"],
        rope_theta=config["rope_theta"],
        n_loops=config["n_loops"],
        loop_input_injection=config["loop_input_injection"],
        per_step_loss=config["per_step_loss"],
        loop_loss_weighting=config["loop_loss_weighting"],
    )

    iter_num = 0
    best_val_loss = 1e9
    patience_counter = 0

    metrics = {
        'train_losses': [],
        'val_losses': [],
        'learning_rates': [],
        'mfu_values': [],
        'iter_times': [],
    }

    # Provenance stamped into every checkpoint, so a saved model can always be traced back
    # to the exact data, tokenizer and code revision that produced it.
    provenance = build_provenance(config)

    checkpoint = None
    if config["init_from"] == 'scratch':
        print("Initializing new model ex nihilo")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif config["init_from"] in ('resume', 'finetune'):
        ckpt_path = Path(config["out_dir"]) / paths.CKPT_LATEST
        if not ckpt_path.exists():
            print(f"Error: --init {config['init_from']} requested but no checkpoint at {ckpt_path}")
            return 1
        print(f"{config['init_from'].capitalize()} from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        checkpoint_model_args = checkpoint['model_args']

        # Architecture comes from the checkpoint -- but the run config must AGREE with it,
        # not silently diverge. block_size is the dangerous one: the checkpoint restores the
        # model at its own context length while get_batch keeps building windows at the
        # configured length, so a mismatch used to blow up inside forward() mid-run.
        arch_keys = ['n_layer', 'n_head', 'n_kv_head', 'n_embd', 'intermediate_size',
                     'block_size', 'vocab_size', 'softcap', 'rope_theta',
                     'n_loops', 'loop_input_injection']
        conflicts = []
        for k in arch_keys:
            if k not in checkpoint_model_args:
                continue
            ckpt_val = checkpoint_model_args[k]
            if k in model_args and model_args[k] != ckpt_val:
                conflicts.append((k, model_args[k], ckpt_val))
            model_args[k] = ckpt_val

        if conflicts:
            print("\n  Checkpoint/config mismatch:")
            for k, requested, actual in conflicts:
                print(f"     {k}: config requests {requested}, checkpoint has {actual}")
            if config["init_from"] == 'resume':
                print("\n  Resuming would train a checkpoint-shaped model on config-shaped data.")
                print("  Fix the config to match, or use --init finetune to accept the "
                      "checkpoint's architecture deliberately.")
                return 1
            print("  --init finetune: adopting the checkpoint's architecture.\n")

        # The run must use the checkpoint's context length for data too.
        config["block_size"] = model_args["block_size"]

        # Optimizer family is a property of the hardware, but silently switching families
        # on resume throws away all optimizer state. Say so.
        ckpt_used_muon = checkpoint.get('use_muon', False)
        if ckpt_used_muon != use_muon:
            print(f"  Optimizer family changes on resume: checkpoint used "
                  f"{'Muon+AdamW' if ckpt_used_muon else 'AdamW'}, this device uses "
                  f"{'Muon+AdamW' if use_muon else 'AdamW'}. Optimizer state will NOT be "
                  f"restored; expect a transient loss bump.")

        # Data/tokenizer provenance: training on different data than the checkpoint saw is
        # legitimate for finetuning but almost never intended on resume.
        ckpt_prov = checkpoint.get('provenance', {})
        if ckpt_prov and config["init_from"] == 'resume':
            for key, label in (('train_bin_sha1', 'train.bin'), ('tokenizer_sha1', 'tokenizer')):
                old, new = ckpt_prov.get(key), provenance.get(key)
                if old and new and old != new:
                    print(f"  {label} changed since this checkpoint was written "
                          f"({old[:12]} -> {new[:12]}).")

        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)

        if config["init_from"] == 'resume':
            iter_num = checkpoint['iter_num']
            best_val_loss = float(checkpoint['best_val_loss'])
            patience_counter = checkpoint.get('early_stopping', {}).get('patience_counter', 0)
            # Restore the data stream so a resumed run does not replay the same windows.
            rng = checkpoint.get('rng_state')
            if rng:
                torch.set_rng_state(rng['torch'].cpu() if hasattr(rng['torch'], 'cpu') else rng['torch'])
                if rng.get('numpy') is not None:
                    np.random.set_state(rng['numpy'])
                if rng.get('cuda') is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(rng['cuda'])
            if 'metrics' in checkpoint:
                metrics = checkpoint['metrics']
        else:
            print("  --init finetune: restarting iteration count and LR schedule at 0.")
    else:
        raise ValueError(f"Unknown init mode {config['init_from']!r}")

    # Crop block size if necessary
    if config["block_size"] < model.config.block_size:
        model.crop_block_size(config["block_size"])
        model_args['block_size'] = config["block_size"]

    model.to(device)

    # Gradient scaler (only needed for float16)
    scaler = None
    if config["device"] == 'cuda' and config["dtype"] == 'float16':
        scaler = torch.amp.GradScaler('cuda', enabled=True)
    elif config["device"] == 'cuda':
        scaler = torch.amp.GradScaler('cuda', enabled=False)

    # Configure optimizer(s)
    raw_model = model.module if ddp else model  # need this before wrapping

    if use_muon:
        param_groups = raw_model.get_param_groups()
        muon_optimizer = Muon(param_groups['muon_params'], lr=config['muon_lr'], momentum=0.95,
                              nesterov=True, weight_decay=config['muon_weight_decay'])

        import inspect
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        adamw_groups = [
            {'params': param_groups['adamw_decay_params'], 'weight_decay': config['weight_decay']},
            {'params': param_groups['adamw_nodecay_params'], 'weight_decay': 0.0}
        ]
        adamw_optimizer = torch.optim.AdamW(
            adamw_groups, lr=config['learning_rate'],
            betas=(config['beta1'], config['beta2']), **extra_args
        )

        n_muon = sum(p.numel() for p in param_groups['muon_params'])
        n_adamw = sum(p.numel() for p in param_groups['adamw_decay_params']) + \
                  sum(p.numel() for p in param_groups['adamw_nodecay_params'])
        print(f"Muon params: {n_muon:,} | AdamW params: {n_adamw:,}")
        print(f"using fused AdamW: {use_fused}")

        if config["init_from"] == 'resume' and checkpoint and 'muon_optimizer' in checkpoint:
            muon_optimizer.load_state_dict(checkpoint['muon_optimizer'])
            adamw_optimizer.load_state_dict(checkpoint['adamw_optimizer'])
    else:
        optimizer = raw_model.configure_optimizers(
            config["weight_decay"], config["learning_rate"],
            (config["beta1"], config["beta2"]), device_type
        )
        if config["init_from"] == 'resume' and checkpoint and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])

    if checkpoint is not None:
        if config["init_from"] == 'resume' and scaler is not None and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        checkpoint = None  # Free memory

    # Compile model
    if config["compile"]:
        print("Compiling model (this may take a minute)...")
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank] if config["device"] == 'cuda' else None)

    # Initialize wandb
    if config["wandb_log"] and master_process:
        try:
            import wandb
            wandb.init(project=config["wandb_project"], name=config["wandb_run_name"], config=config)
        except ImportError:
            print("Weights & Biases not available, continuing without logging")
            config["wandb_log"] = False

    # Training loop
    print(f"\nStarting training for Latin corpus...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    try:
        X, Y = get_batch('train', config)
        # Fail fast if validation is missing. This used to fall back to scoring the TRAINING
        # data and reporting it as "val loss", which silently invalidates the whole run.
        _get_split_data('val', config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run prepare_corpus.py first to prepare the training data.")
        return 1

    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0

    def save_checkpoint(filename):
        """Write a full resumable checkpoint atomically.

        Includes RNG state, scaler state, metrics and provenance hashes so a resumed run
        continues the same data stream rather than re-randomizing, and so any checkpoint
        can be traced back to the exact data + tokenizer + code that produced it.
        Written to a temp file and renamed, so an interrupted save cannot leave a truncated
        checkpoint where a valid one used to be.
        """
        ckpt = {
            'model': raw_model.state_dict(),
            'model_args': model_args,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
            'config': config,
            'use_muon': use_muon,
            'metrics': metrics,
            'provenance': provenance,
            'rng_state': {
                'torch': torch.get_rng_state(),
                'numpy': np.random.get_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            'early_stopping': {'patience_counter': patience_counter},
        }
        if scaler is not None:
            ckpt['scaler'] = scaler.state_dict()
        if use_muon:
            ckpt['muon_optimizer'] = muon_optimizer.state_dict()
            ckpt['adamw_optimizer'] = adamw_optimizer.state_dict()
        else:
            ckpt['optimizer'] = optimizer.state_dict()

        dest = Path(config["out_dir"]) / filename
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        torch.save(ckpt, tmp)
        os.replace(tmp, dest)

    while True:
        # Set learning rate (WSD schedule)
        lr = get_lr(iter_num, config) if config["decay_lr"] else config["learning_rate"]

        if use_muon:
            # Scale both optimizers' LR
            lr_ratio = lr / config["learning_rate"]
            for param_group in adamw_optimizer.param_groups:
                param_group['lr'] = lr
            for param_group in muon_optimizer.param_groups:
                param_group['lr'] = config['muon_lr'] * lr_ratio
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Evaluation and checkpointing
        if iter_num % config["eval_interval"] == 0 and master_process:
            losses = estimate_loss(raw_model, config, ctx)
            # Select and report on the FINAL readout: it is the only one inference uses, so
            # it is what makes the number comparable to a conventional model. The averaged
            # loop objective is still logged, but never used to pick "best".
            val_loss = float(losses['val_final'])
            train_loss = float(losses['train_final'])
            msg = (f"Step {iter_num}: train {train_loss:.4f}, val {val_loss:.4f} (final readout)")
            if config["n_loops"] > 1:
                msg += f" | loop-avg objective: train {float(losses['train']):.4f}, val {float(losses['val']):.4f}"
            if 'val_bpb' in losses:
                msg += f" | val {losses['val_bpb']:.4f} bits/byte"
            print(msg)

            # Collect metrics
            metrics['train_losses'].append((iter_num, train_loss))
            metrics['val_losses'].append((iter_num, val_loss))

            if config["wandb_log"]:
                import wandb
                log = {
                    "iter": iter_num,
                    "train/loss_final": train_loss,
                    "val/loss_final": val_loss,
                    "train/loss_objective": float(losses['train']),
                    "val/loss_objective": float(losses['val']),
                    "lr": lr,
                    "mfu": running_mfu * 100,
                }
                if 'val_bpb' in losses:
                    log["val/bits_per_byte"] = losses['val_bpb']
                wandb.log(log)

            # Single source of truth for "did val loss improve?"
            improved = val_loss < (best_val_loss - config["min_delta"])
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            # Checkpointing (skip iter 0).
            #   ckpt.pt      -> rolling resume state (latest)
            #   ckpt_best.pt -> the best model seen, written ONLY on improvement so the
            #                   best generalizer is never clobbered by a later worse step.
            if iter_num > 0:
                if config["always_save_checkpoint"]:
                    save_checkpoint('ckpt.pt')
                    print(f"Saved rolling checkpoint to {config['out_dir']}/ckpt.pt")
                if is_best:
                    save_checkpoint('ckpt_best.pt')
                    print(f"New best val loss {best_val_loss:.4f} -> saved {config['out_dir']}/ckpt_best.pt")

            # Early stopping: only arm the patience counter once we're in the WSD decay
            # phase. A plateau during the long constant-LR stable phase is expected and
            # must not truncate the run before decay (which delivers the biggest drop).
            if config["early_stopping"] and iter_num > 0:
                decay_start = int(config["max_iters"] * (1.0 - config["decay_fraction"]))
                if iter_num >= decay_start:
                    if improved:
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        print(f"Patience: {patience_counter}/{config['patience']} (val: {val_loss:.4f}, best: {best_val_loss:.4f})")
                        if patience_counter >= config["patience"]:
                            print(f"Early stopping triggered after {patience_counter} evaluations without improvement")
                            print(f"   Best validation loss: {best_val_loss:.4f}")
                            break

        if iter_num == 0 and config["eval_only"]:
            break

        # Forward pass with gradient accumulation
        accumulated_loss = 0.0
        for micro_step in range(config["gradient_accumulation_steps"]):
            if ddp:
                model.require_backward_grad_sync = (micro_step == config["gradient_accumulation_steps"] - 1)

            with ctx:
                _, loss, _ = model(X, Y)
                loss = loss / config["gradient_accumulation_steps"]
                accumulated_loss += loss.item()

            X, Y = get_batch('train', config)

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        # Gradient clipping and optimizer step
        if use_muon:
            if config["grad_clip"] != 0.0:
                if scaler is not None:
                    scaler.unscale_(muon_optimizer)
                    scaler.unscale_(adamw_optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

            if scaler is not None:
                scaler.step(muon_optimizer)
                scaler.step(adamw_optimizer)
                scaler.update()
            else:
                muon_optimizer.step()
                adamw_optimizer.step()

            muon_optimizer.zero_grad(set_to_none=True)
            adamw_optimizer.zero_grad(set_to_none=True)
        else:
            if config["grad_clip"] != 0.0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        if iter_num % config["log_interval"] == 0 and master_process:
            lossf = accumulated_loss
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(config["batch_size"] * config["gradient_accumulation_steps"], dt, config["peak_flops"])
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"Iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")

            # Collect metrics
            metrics['learning_rates'].append((iter_num, lr))
            metrics['mfu_values'].append((iter_num, running_mfu))
            metrics['iter_times'].append((iter_num, dt))

        iter_num += 1
        local_iter_num += 1

        if iter_num > config["max_iters"]:
            break

    print("\nTraining completed!")

    # The loop exits between eval intervals, so the last iterations were never checkpointed.
    # Write them out explicitly instead of discarding them.
    if master_process:
        save_checkpoint(paths.CKPT_FINAL)
        print(f"Final weights (iter {iter_num}) saved to {config['out_dir']}/{paths.CKPT_FINAL}")

    # Generate training visualization
    if master_process:
        plot_training_metrics(metrics, config["out_dir"])

    if ddp:
        destroy_process_group()

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
