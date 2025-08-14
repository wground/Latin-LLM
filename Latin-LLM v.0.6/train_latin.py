"""
LatinLLM Training Script
Trains a GPT model on Latin text corpus using system-optimized configurations.
Uses detect_system.py output for optimal hardware configuration.

Usage:
    python3 train_latin.py [--config CONFIG_FILE] [--batch_size SIZE] [--max_iters ITERS]

Examples:
    # Use auto-detected system config
    python3 train_latin.py
    
    # Override batch size
    python3 train_latin.py --batch_size 8
    
    # Use custom config file
    python3 train_latin.py --config my_config.json

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

# Import local model
from model import GPTConfig, GPT

def load_system_config(config_path: str = "latin_training_config.json") -> Dict[str, Any]:
    """Load system configuration from detect_system.py output."""
    if not os.path.exists(config_path):
        print(f"⚠️  Config file {config_path} not found!")
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
    
    print(f"✅ Loaded system config from {config_path}")
    return config

def load_tokenizer_config(data_dir: str = "gpt_data_latin") -> Dict[str, Any]:
    """Load custom tokenizer configuration and metadata."""
    meta_path = os.path.join(data_dir, "meta.pkl")
    
    if not os.path.exists(meta_path):
        print(f"❌ Tokenizer metadata not found at {meta_path}")
        print("You must run 'python3 prepare_latin.py' first to create custom tokenizer")
        print("Cannot continue without custom Latin tokenizer metadata.")
        exit(1)
    
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    
    print(f"✅ Loaded custom Latin tokenizer metadata")
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
    
    # Load tokenizer config to determine vocab size and dataset size
    tokenizer_config = load_tokenizer_config()
    vocab_size = tokenizer_config["vocab_size"]
    data_stats = tokenizer_config.get("data_stats", {})
    train_tokens = data_stats.get("train_tokens", 0)
    
    # Adaptive model configuration based on vocab size and dataset size
    if vocab_size > 12000:  # Large vocab (16k+)
        n_layer = 8
        n_head = 8  
        n_embd = 512
        batch_size_multiplier = 0.8  # Reduce batch size for larger model
    elif vocab_size > 8000:  # Medium vocab (12k)
        n_layer = 7
        n_head = 7
        n_embd = 448
        batch_size_multiplier = 0.9
    else:  # Smaller vocab (8k)
        n_layer = 6
        n_head = 6
        n_embd = 384
        batch_size_multiplier = 1.0
    
    # Adaptive training parameters based on dataset size
    if train_tokens > 1_000_000:  # Large dataset
        eval_interval = 500
        gradient_accumulation_steps = 6
        warmup_iters = 1000
    elif train_tokens > 500_000:  # Medium dataset
        eval_interval = 350
        gradient_accumulation_steps = 5
        warmup_iters = 750
    else:  # Smaller dataset
        eval_interval = 250
        gradient_accumulation_steps = 4
        warmup_iters = 500
    
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
        
        # Adaptive Model Configuration
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "dropout": 0.2,  # Increased from 0.1 to combat overfitting
        "bias": False,
        "use_rmsnorm": True,   # Use RMSNorm for better efficiency (Though I need to play around with this vs layer normalization...)
        
        # Optimizer Configuration (adaptive learning rate)
        "learning_rate": 3e-4 if vocab_size > 12000 else 4e-4,  # Reduced for better stability
        "max_iters": args.max_iters,
        "weight_decay": 5e-2,  # Reduced from 2e-1 for better convergence
        "beta1": 0.9,
        "beta2": 0.95,
        "grad_clip": 1.0,  # Restored to 1.0 for sufficient gradient updates
        
        # Learning Rate Schedule
        "decay_lr": True,
        "warmup_iters": warmup_iters,
        "lr_decay_iters": args.max_iters,
        "min_lr": 5e-5,
        
        # Weights & Biases
        "wandb_log": False,
        "wandb_project": "latin-llm",
        "wandb_run_name": f"latin-gpt-v{vocab_size//1000}k",
        
        # Early Stopping Configuration
        "early_stopping": True,
        "patience": 15,  # Stop if val loss doesn't improve for 15 evaluations
        "min_delta": 0.005,  # More lenient minimum improvement threshold
        
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
    
    # Use vocab size from tokenizer
    config["vocab_size"] = vocab_size
    
    # Check if we should resume from checkpoint
    config["init_from"] = 'resume' if os.path.exists(os.path.join(config["out_dir"], 'ckpt.pt')) else 'scratch'
    
    return config

def get_batch(split: str, config: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a batch of data from the Latin corpus."""
    # Load the binary files created by prepare_latin.py (updated to use new data location)
    data_dir = "gpt_data_latin"
    filename = os.path.join(data_dir, f"{split}.bin")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Data file {filename} not found. Run prepare_latin.py first.")
    
    data = np.memmap(filename, dtype=np.uint16, mode='r')
    
    # Generate random indices
    ix = torch.randint(len(data) - config["block_size"], (config["batch_size"],))
    
    # Create input and target sequences
    x = torch.stack([torch.from_numpy((data[i:i+config["block_size"]]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+config["block_size"]]).astype(np.int64)) for i in ix])
    
    # Move to device
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
                    # If no validation set, use train set
                    X, Y = get_batch('train', config)
                    with ctx:
                        _, loss = model(X, Y)
                    losses[k] = loss.item()
                else:
                    raise
        
        out[split] = losses.mean()
    
    model.train()
    return out

def get_hardware_peak_flops(system_config: Dict[str, Any], dtype: str) -> float:
    """Estimate hardware peak FLOPS based on detected GPU."""
    # Default fallback to A100
    default_flops = 312e12  # A100 bfloat16
    
    try:
        pytorch_info = system_config.get("pytorch", {})
        gpu_devices = pytorch_info.get("gpu_devices", [])
        
        if not gpu_devices or "device" not in system_config.get("recommended_config", {}):
            return default_flops
            
        device_type = system_config["recommended_config"]["device"]
        
        # Handle MPS (Apple Silicon)
        if device_type == "mps" and gpu_devices:
            # Apple Silicon estimates based on chip type
            gpu_name = gpu_devices[0].get("name", "").lower()
            if "m3" in gpu_name:
                return 15e12 if dtype == "float16" else 7.5e12  # M3 estimates
            elif "m2" in gpu_name:
                return 13e12 if dtype == "float16" else 6.5e12  # M2 estimates  
            elif "m1" in gpu_name:
                return 11e12 if dtype == "float16" else 5.5e12  # M1 estimates
            else:
                return 10e12  # Conservative Apple Silicon estimate
        
        # Handle CUDA GPUs
        elif device_type == "cuda" and gpu_devices:
            gpu_name = gpu_devices[0].get("name", "").lower()
            
            # NVIDIA GPU peak FLOPS estimates (approximate, varies by exact model)
            gpu_flops = {
                # RTX 50 series (2025) - Blackwell architecture
                "rtx 5090": 200e12 if dtype == "bfloat16" else 107e12,  # Estimated mixed precision FLOPS
                "rtx 5080": 120e12 if dtype == "bfloat16" else 60e12,   # Estimated based on CUDA cores
                "rtx 5070 ti": 100e12 if dtype == "bfloat16" else 50e12, # Estimated
                "rtx 5070": 80e12 if dtype == "bfloat16" else 40e12,    # Estimated
                
                # RTX 40 series
                "rtx 4090": 165e12 if dtype == "bfloat16" else 83e12,
                "rtx 4080": 120e12 if dtype == "bfloat16" else 60e12,
                "rtx 4070": 90e12 if dtype == "bfloat16" else 45e12,
                
                # RTX 30 series (no bfloat16 support)
                "rtx 3090": 71e12 if dtype == "float16" else 35e12,
                "rtx 3080": 58e12 if dtype == "float16" else 29e12,
                "rtx 3070": 40e12 if dtype == "float16" else 20e12,
                
                # Tesla/Quadro data center GPUs
                "a100": 312e12 if dtype == "bfloat16" else 156e12,
                "h100": 756e12 if dtype == "bfloat16" else 378e12,
                "v100": 125e12 if dtype == "float16" else 62e12,
                "a40": 150e12 if dtype == "bfloat16" else 75e12,
                "a30": 165e12 if dtype == "bfloat16" else 82e12,
                
                # AMD GPUs (ROCm) - Updated with RDNA 4 (2025)
                "rx 9090 xt": 200e12 if dtype == "float16" else 100e12,  # Estimated high-end RDNA 4
                "rx 9080 xt": 160e12 if dtype == "float16" else 80e12,   # Estimated upper-tier RDNA 4
                "rx 9070 xt": 97e12 if dtype == "float16" else 48e12,    # 97.3 TFLOPS FP16, 48.7 TFLOPS FP32
                "rx 9070": 72e12 if dtype == "float16" else 36e12,       # 36.1 TFLOPS FP32, estimated FP16
                "rx 9060 xt": 45e12 if dtype == "float16" else 22e12,    # Estimated mid-range RDNA 4
                "rx 9060": 35e12 if dtype == "float16" else 18e12,       # Estimated entry RDNA 4
                
                # RDNA 3 (7000 series)
                "rx 7900 xtx": 122e12 if dtype == "float16" else 61e12,  # Actual specs
                "rx 7900 xt": 103e12 if dtype == "float16" else 51e12,   # Actual specs
                "rx 7800 xt": 75e12 if dtype == "float16" else 37e12,    # Estimated from compute units
                "rx 7700 xt": 60e12 if dtype == "float16" else 30e12,    # Estimated from compute units
                "rx 7600 xt": 40e12 if dtype == "float16" else 20e12,    # Estimated
                "rx 7600": 32e12 if dtype == "float16" else 16e12,       # Estimated
                
                # RDNA 2 (6000 series)
                "rx 6950 xt": 46e12 if dtype == "float16" else 23e12,
                "rx 6900 xt": 46e12 if dtype == "float16" else 23e12,
                "rx 6800 xt": 40e12 if dtype == "float16" else 20e12,
                "rx 6700 xt": 26e12 if dtype == "float16" else 13e12,
                "rx 6600 xt": 20e12 if dtype == "float16" else 10e12,
                
                # Data center AMD cards
                "mi250": 180e12 if dtype == "bfloat16" else 90e12,
                "mi210": 180e12 if dtype == "bfloat16" else 90e12,
                "mi100": 185e12 if dtype == "bfloat16" else 92e12,
            }
            
            # Try to match GPU name
            for gpu_key, flops in gpu_flops.items():
                if gpu_key in gpu_name:
                    return flops
            
            # Fallback: estimate based on memory and compute capability
            compute_cap = gpu_devices[0].get("compute_capability", "")
            memory_gb = gpu_devices[0].get("memory_total", 0) / (1024**3)
            
            if compute_cap.startswith("8."):  # Ampere or newer
                if memory_gb > 20:
                    return 200e12 if dtype in ["bfloat16", "float16"] else 100e12
                elif memory_gb > 10:
                    return 100e12 if dtype in ["bfloat16", "float16"] else 50e12
                else:
                    return 60e12 if dtype in ["bfloat16", "float16"] else 30e12
            elif compute_cap.startswith("7."):  # Turing
                return 60e12 if dtype == "float16" else 30e12
            else:
                return 40e12 if dtype == "float16" else 20e12
        
        return default_flops
        
    except Exception:
        # If anything fails, fall back to A100
        return default_flops

def get_lr(it: int, config: Dict[str, Any]) -> float:
    """Learning rate scheduler with cosine decay."""
    # Linear warmup
    if it < config["warmup_iters"]:
        return config["learning_rate"] * (it + 1) / (config["warmup_iters"] + 1)
    
    # Cosine decay
    if it > config["lr_decay_iters"]:
        return config["min_lr"]
    
    decay_ratio = (it - config["warmup_iters"]) / (config["lr_decay_iters"] - config["warmup_iters"])
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config["min_lr"] + coeff * (config["learning_rate"] - config["min_lr"])

def main():
    parser = argparse.ArgumentParser(description="Train LatinLLM model")
    parser.add_argument("--config", default="latin_training_config.json", help="System config file")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--max_iters", type=int, default=75000, help="Maximum training iterations") # Mess with max_iters when testing training
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    args = parser.parse_args()
    
    print("🏛️  LatinLLM Training Script")
    print("=" * 50)
    
    # Load system configuration
    system_config = load_system_config(args.config)
    config = setup_training_config(system_config, args)
    
    # Override wandb setting if specified
    if args.wandb:
        config["wandb_log"] = True
    
    # Calculate hardware-specific peak FLOPS for accurate MFU
    peak_flops = get_hardware_peak_flops(system_config, config["dtype"])
    config["peak_flops"] = peak_flops
    
    # Print configuration summary
    print(f"Device: {config['device']} ({config['dtype']})")
    print(f"Model: {config['n_layer']} layers, {config['n_head']} heads, {config['n_embd']} embedding dim")
    print(f"Normalization: {'RMSNorm' if config['use_rmsnorm'] else 'LayerNorm'}")
    print(f"Hardware peak FLOPS: {peak_flops/1e12:.1f} TFLOPS ({config['dtype']})")
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
    
    # Seed for reproducibility
    torch.manual_seed(1337 + seed_offset)
    
    # Hardware optimizations
    if config["device"] == 'cuda':
        if config["enable_tf32"]:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    # Set up mixed precision context
    if config["device"] == 'mps':
        device_type = 'cpu'  # MPS uses CPU autocast
    elif config["device"] == 'cuda':
        device_type = 'cuda'
    else:
        device_type = 'cpu'
    
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config["dtype"]]
    ctx = nullcontext() if device_type == 'cpu' and config["device"] == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Initialize model
    model_args = dict(
        n_layer=config["n_layer"], 
        n_head=config["n_head"], 
        n_embd=config["n_embd"],
        block_size=config["block_size"],
        bias=config["bias"], 
        vocab_size=config["vocab_size"], 
        dropout=config["dropout"],
        use_rmsnorm=config["use_rmsnorm"]
    )
    
    iter_num = 0
    best_val_loss = 1e9
    patience_counter = 0  # For early stopping
    
    if config["init_from"] == 'scratch':
        print("‼️ Initializing new model ex nihilo")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif config["init_from"] == 'resume':
        print(f"🔄 Resuming training from {config['out_dir']}")
        ckpt_path = os.path.join(config["out_dir"], 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        
        # Use checkpoint model args
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        
        # Handle new parameters with defaults for backward compatibility
        model_args['use_rmsnorm'] = checkpoint_model_args.get('use_rmsnorm', False)
        
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        
        # Load model state
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
    
    # Initialize gradient scaler
    scaler = None
    if config["device"] == 'cuda' and config["dtype"] == 'float16':
        scaler = torch.amp.GradScaler('cuda', enabled=True)
    elif config["device"] == 'cuda':
        scaler = torch.amp.GradScaler('cuda', enabled=False)
    # MPS doesn't use gradient scaling
    
    # Configure optimizer
    optimizer = model.configure_optimizers(
        config["weight_decay"], 
        config["learning_rate"], 
        (config["beta1"], config["beta2"]), 
        device_type
    )
    
    if config["init_from"] == 'resume' and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    checkpoint = None  # Free memory
    
    # Compile model if supported and enabled
    if config["compile"]:
        print("⚡ Compiling model (this may take a minute)...")
        model = torch.compile(model)
    
    # Wrap with DDP if needed
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank] if config["device"] == 'cuda' else None)
    
    # Initialize wandb
    if config["wandb_log"] and master_process:
        try:
            import wandb
            wandb.init(project=config["wandb_project"], name=config["wandb_run_name"], config=config)
        except ImportError:
            print("⚠️  Weights & Biases not available, continuing without logging")
            config["wandb_log"] = False
    
    # Training loop
    print(f"\n🏛️  Starting training for Latin corpus...")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
    # Check data files exist
    try:
        X, Y = get_batch('train', config)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Please run prepare_latin.py first to prepare the training data.")
        return 1
    
    t0 = time.time()
    local_iter_num = 0
    raw_model = model.module if ddp else model
    running_mfu = -1.0
    
    while True:
        # Set learning rate
        lr = get_lr(iter_num, config) if config["decay_lr"] else config["learning_rate"]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluation and checkpointing
        if iter_num % config["eval_interval"] == 0 and master_process:
            losses = estimate_loss(raw_model, config, ctx)
            print(f"📈 Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses.get('val', losses['train']):.4f}")
            
            if config["wandb_log"]:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses.get('val', losses['train']),
                    "lr": lr,
                    "mfu": running_mfu*100,
                })
            
            val_loss = losses.get('val', losses['train'])
            
            # Early stopping logic
            if config["early_stopping"] and iter_num > 0:
                if val_loss < best_val_loss - config["min_delta"]:
                    best_val_loss = val_loss
                    patience_counter = 0
                    print(f"🎯 New best validation loss: {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                    print(f"⏳ Patience: {patience_counter}/{config['patience']} (val loss: {val_loss:.4f}, best: {best_val_loss:.4f})")
                    
                    if patience_counter >= config["patience"]:
                        print(f"🛑 Early stopping triggered after {patience_counter} evaluations without improvement")
                        print(f"   Best validation loss: {best_val_loss:.4f}")
                        break
            
            if val_loss < best_val_loss or config["always_save_checkpoint"]:
                if not config["early_stopping"]:  # Only update best_val_loss if not using early stopping
                    best_val_loss = val_loss
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"💾 Saving checkpoint to {config['out_dir']}")
                    torch.save(checkpoint, os.path.join(config["out_dir"], 'ckpt.pt'))
        
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
            
            # Fetch next batch
            X, Y = get_batch('train', config)
            
            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        
        # Optimizer step
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
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"⚡ Iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        
        iter_num += 1
        local_iter_num += 1
        
        # Check termination
        if iter_num > config["max_iters"]:
            break
    
    print("\n✅ Training completed!")
    
    if ddp:
        destroy_process_group()
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)