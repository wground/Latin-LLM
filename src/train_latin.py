"""
LatinLLM Training Script
Trains a modernized GPT model on Latin text corpus using system-optimized configurations.
Uses detect_system.py output for optimal hardware configuration.

Architecture: RoPE, SwiGLU, GQA, RMSNorm, QK-norm (nanoChat-inspired)
Optimizer: Muon + AdamW hybrid (CUDA) or AdamW (MPS/CPU)
LR Schedule: Warmup-Stable-Decay (WSD)

Usage:
    python3 train_latin.py [--config CONFIG_FILE] [--batch_size SIZE] [--max_iters ITERS]

Author: Willow Groundwater-Schuldt & Claude
"""

import os
import time
import math
import json
import pickle
import argparse
from contextlib import nullcontext
from typing import Dict, Any, Tuple

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

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

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
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

                p.add_(update, alpha=-lr)


# --- Configuration Loading ---

def load_system_config(config_path: str = "latin_training_config.json") -> Dict[str, Any]:
    """Load system configuration from detect_system.py output."""
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


def load_tokenizer_config(data_dir: str = "gpt_data_latin") -> Dict[str, Any]:
    """Load custom tokenizer configuration and metadata."""
    meta_path = os.path.join(data_dir, "meta.pkl")

    if not os.path.exists(meta_path):
        print(f"Tokenizer metadata not found at {meta_path}")
        print("You must run 'python3 prepare_latin.py' first to create custom tokenizer")
        exit(1)

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    print(f"Loaded custom Latin tokenizer metadata")
    print(f"   Vocabulary size: {meta['vocab_size']}")
    print(f"   Tokenizer type: {meta['tokenizer_config']['type']}")
    if "data_stats" in meta:
        print(f"   Training tokens: {meta['data_stats']['train_tokens']:,}")
        print(f"   Validation tokens: {meta['data_stats']['val_tokens']:,}")

    return {
        "vocab_size": meta["vocab_size"],
        "tokenizer_type": meta["tokenizer_config"]["type"],
        "data_stats": meta.get("data_stats", {})
    }


def setup_training_config(system_config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Setup training configuration based on system capabilities and user args."""
    rec_config = system_config["recommended_config"]

    tokenizer_config = load_tokenizer_config()
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
        "out_dir": "out-latin",
        "eval_interval": eval_interval,
        "log_interval": 10,
        "eval_iters": 150,
        "eval_only": False,
        "always_save_checkpoint": True,

        # Dataset Configuration
        "dataset": "latin",
        "gradient_accumulation_steps": gradient_accumulation_steps,

        # Model Configuration (modernized architecture)
        "n_layer": n_layer,
        "n_head": n_head,
        "n_kv_head": n_kv_head,
        "n_embd": n_embd,
        "intermediate_size": 0,  # Auto-compute SwiGLU hidden dim
        "dropout": 0.15,
        "softcap": 15.0,
        "rope_theta": 10000.0,

        # Optimizer Configuration
        "learning_rate": 3e-4 if vocab_size > 12000 else 4e-4,
        "muon_lr": 0.02,  # Muon base LR (only used on CUDA)
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
        "early_stopping": True,
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

    # Check if we should resume from checkpoint
    config["init_from"] = 'resume' if os.path.exists(os.path.join(config["out_dir"], 'ckpt.pt')) else 'scratch'

    return config


# --- Data Loading ---

def get_batch(split: str, config: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a batch of data from the Latin corpus."""
    data_dir = "gpt_data_latin"
    filename = os.path.join(data_dir, f"{split}.bin")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Data file {filename} not found. Run prepare_latin.py first.")

    data = np.memmap(filename, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - config["block_size"], (config["batch_size"],))

    x = torch.stack([torch.from_numpy((data[i:i + config["block_size"]]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + config["block_size"]]).astype(np.int64)) for i in ix])

    if config["device"] == 'cuda':
        x, y = x.pin_memory().to(config["device"], non_blocking=True), y.pin_memory().to(config["device"], non_blocking=True)
    else:
        x, y = x.to(config["device"]), y.to(config["device"])

    return x, y


@torch.no_grad()
def estimate_loss(model, config: Dict[str, Any], ctx) -> Dict[str, float]:
    """Estimate model loss on train and validation sets."""
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(config["eval_iters"])
        for k in range(config["eval_iters"]):
            try:
                X, Y = get_batch(split, config)
                with ctx:
                    _, loss = model(X, Y)
                losses[k] = loss.item()
            except FileNotFoundError:
                if split == 'val':
                    X, Y = get_batch('train', config)
                    with ctx:
                        _, loss = model(X, Y)
                    losses[k] = loss.item()
                else:
                    raise

        out[split] = losses.mean()

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

        # Apple Silicon
        if device_type == "mps" and gpu_devices:
            gpu_name = gpu_devices[0].get("name", "").lower()
            if "m4" in gpu_name:
                return 18e12 if dtype == "float16" else 9e12
            elif "m3" in gpu_name:
                return 15e12 if dtype == "float16" else 7.5e12
            elif "m2" in gpu_name:
                return 13e12 if dtype == "float16" else 6.5e12
            elif "m1" in gpu_name:
                return 11e12 if dtype == "float16" else 5.5e12
            else:
                return 10e12

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

    # Phase 3: Linear decay
    progress = (it - decay_start) / max(1, max_iters - decay_start)
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
    parser.add_argument("--config", default="latin_training_config.json", help="System config file")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--max_iters", type=int, default=75000, help="Maximum training iterations")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    args = parser.parse_args()

    print("LatinLLM Training Script")
    print("=" * 50)

    # Load system configuration
    system_config = load_system_config(args.config)
    config = setup_training_config(system_config, args)

    if args.wandb:
        config["wandb_log"] = True

    # Hardware peak FLOPS for MFU calculation
    peak_flops = get_hardware_peak_flops(system_config, config["dtype"])
    config["peak_flops"] = peak_flops

    # Determine if we can use Muon (CUDA only)
    use_muon = config["device"] == "cuda"

    # Print configuration
    print(f"Device: {config['device']} ({config['dtype']})")
    print(f"Model: {config['n_layer']} layers, {config['n_head']} heads ({config['n_kv_head']} KV), {config['n_embd']} embd")
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
    )

    iter_num = 0
    best_val_loss = 1e9
    patience_counter = 0

    if config["init_from"] == 'scratch':
        print("Initializing new model ex nihilo")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif config["init_from"] == 'resume':
        print(f"Resuming training from {config['out_dir']}")
        ckpt_path = os.path.join(config["out_dir"], 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        checkpoint_model_args = checkpoint['model_args']

        # Use checkpoint model args (handles architecture compatibility)
        for k in ['n_layer', 'n_head', 'n_kv_head', 'n_embd', 'intermediate_size',
                   'block_size', 'vocab_size', 'dropout', 'softcap', 'rope_theta']:
            if k in checkpoint_model_args:
                model_args[k] = checkpoint_model_args[k]

        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)

        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']

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
        muon_optimizer = Muon(param_groups['muon_params'], lr=config['muon_lr'], momentum=0.95, nesterov=True)

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

        if config["init_from"] == 'resume' and 'muon_optimizer' in checkpoint:
            muon_optimizer.load_state_dict(checkpoint['muon_optimizer'])
            adamw_optimizer.load_state_dict(checkpoint['adamw_optimizer'])
    else:
        optimizer = raw_model.configure_optimizers(
            config["weight_decay"], config["learning_rate"],
            (config["beta1"], config["beta2"]), device_type
        )
        if config["init_from"] == 'resume' and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])

    if config["init_from"] == 'resume':
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

    # Metrics collection for visualization
    metrics = {
        'train_losses': [],
        'val_losses': [],
        'learning_rates': [],
        'mfu_values': [],
        'iter_times': [],
    }

    # Training loop
    print(f"\nStarting training for Latin corpus...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    try:
        X, Y = get_batch('train', config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run prepare_latin.py first to prepare the training data.")
        return 1

    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0

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
            val_loss = losses.get('val', losses['train'])
            print(f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {val_loss:.4f}")

            # Collect metrics
            metrics['train_losses'].append((iter_num, losses['train'].item() if hasattr(losses['train'], 'item') else float(losses['train'])))
            metrics['val_losses'].append((iter_num, val_loss.item() if hasattr(val_loss, 'item') else float(val_loss)))

            if config["wandb_log"]:
                import wandb
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": val_loss,
                    "lr": lr,
                    "mfu": running_mfu * 100,
                })

            # Early stopping logic
            if config["early_stopping"] and iter_num > 0:
                if val_loss < best_val_loss - config["min_delta"]:
                    best_val_loss = val_loss
                    patience_counter = 0
                    print(f"New best validation loss: {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                    print(f"Patience: {patience_counter}/{config['patience']} (val: {val_loss:.4f}, best: {best_val_loss:.4f})")

                    if patience_counter >= config["patience"]:
                        print(f"Early stopping triggered after {patience_counter} evaluations without improvement")
                        print(f"   Best validation loss: {best_val_loss:.4f}")
                        break

            if val_loss < best_val_loss or config["always_save_checkpoint"]:
                if not config["early_stopping"]:
                    best_val_loss = val_loss
                if iter_num > 0:
                    ckpt = {
                        'model': raw_model.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                        'use_muon': use_muon,
                    }
                    if use_muon:
                        ckpt['muon_optimizer'] = muon_optimizer.state_dict()
                        ckpt['adamw_optimizer'] = adamw_optimizer.state_dict()
                    else:
                        ckpt['optimizer'] = optimizer.state_dict()
                    print(f"Saving checkpoint to {config['out_dir']}")
                    torch.save(ckpt, os.path.join(config["out_dir"], 'ckpt.pt'))

        if iter_num == 0 and config["eval_only"]:
            break

        # Forward pass with gradient accumulation
        accumulated_loss = 0.0
        for micro_step in range(config["gradient_accumulation_steps"]):
            if ddp:
                model.require_backward_grad_sync = (micro_step == config["gradient_accumulation_steps"] - 1)

            with ctx:
                _, loss = model(X, Y)
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

    # Generate training visualization
    if master_process:
        plot_training_metrics(metrics, config["out_dir"])

    if ddp:
        destroy_process_group()

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
