# LatinLLM

![Pantheon](assets/pantheon.jpg)

Modern LLMs are trained on diverse datasets which often prioritize modern material. With the increased presence of AI in education, governance, etc. it is important that we maintain access to the past and the Western tradition, even when using modern technology. This project is meant to be a proof-of-concept, training a GPT model on Latin language text exclusively.
Latin-LLM is a GPT language model trained on a large corpus of classical, medieval, and some neo-Latin texts. Built on a modernized transformer architecture inspired by [nanoChat](https://github.com/karpathy/nanochat).

## Model Architecture

LatinLLM uses a modern transformer stack with techniques from LLaMA, nanoChat, and recent LLM research:

- **Rotary Position Embeddings (RoPE)** instead of learned positional embeddings — better generalization, no wasted parameters
- **Parameterless RMSNorm** — simpler and more efficient than LayerNorm
- **SwiGLU MLP** — gated activation (SiLU gate * linear up, then down projection) with 8/3x hidden dim for parameter parity
- **Grouped Query Attention (GQA)** — fewer KV heads than query heads, reducing memory with minimal quality loss
- **QK Normalization** — normalizes queries and keys before attention for training stability
- **Logit Soft-Capping** — tanh-based capping of output logits for numerical stability
- **Flash Attention** via PyTorch's `scaled_dot_product_attention`
- **Weight Tying** between token embeddings and output head
- **No bias terms** anywhere in the network
- **Looped / recurrent depth (optional)** — the block stack can be iterated `n_loops` times in latent space, giving an effective depth of `n_layer × n_loops` with **no extra parameters** (see below)

### Model Configuration

The model adapts its size based on vocabulary:

| Vocab Size | Layers | Heads | KV Heads | Embedding Dim | SwiGLU Hidden | Parameters |
|------------|--------|-------|----------|---------------|---------------|------------|
| 8K         | 6      | 6     | 3 (GQA)  | 384           | 1024          | ~20M       |
| 12K        | 7      | 7     | 7 (MHA)  | 448           | 1216          | ~27M       |
| 16K+       | 8      | 8     | 4 (GQA)  | 512           | 1408          | ~32M       |

Layer count, embedding dim, and dropout can be overridden directly (`--n_layer`, `--n_embd`, `--dropout`) instead of using the vocab-derived defaults.

### Looped / Recurrent Depth (Ouro-style)

Inspired by [Ouro: Looped Language Models](https://arxiv.org/abs/2510.25741), the model can reuse its `n_layer` blocks for `n_loops` iterations, reaching an **effective depth of `n_layer × n_loops` without adding parameters** — a strong lever for a data-constrained corpus, where adding raw width/depth risks overfitting. Each iteration re-injects the token embedding (Universal-Transformer style), and **deep supervision** computes the LM loss after every loop step (averaged uniformly, or linearly up-weighting later steps).

```bash
# effective depth 8 × 3 = 24, deep supervision on
python3 train_latin.py --n_loops 3
```

With `n_loops=1` (the default) the model is identical to the standard transformer above.

## Training

### Optimizer: Muon + AdamW Hybrid

On CUDA GPUs, LatinLLM uses a hybrid optimizer for ~2x compute efficiency:

- **[Muon](https://github.com/KellerJordan/Muon)** for all 2D weight matrices (attention projections, MLP weights) — orthogonalizes gradient updates via Newton-Schulz iterations, with decoupled weight decay ([Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982))
- **AdamW** for embeddings and 1D parameters (norms)
- Falls back to AdamW-only on MPS (Apple Silicon) and CPU

### Learning Rate: Warmup-Stable-Decay (WSD)

Replaces the older cosine schedule. Three phases:
1. **Warmup** — linear ramp to peak LR
2. **Stable** — constant at peak LR through the middle of training (more useful learning than cosine)
3. **Decay** — linear decay to min LR over the final `decay_fraction` of training (default 30%)

### Other Training Features
- **Custom BPE tokenizer** trained on the Latin corpus (16K vocab)
- **Hardware auto-detection** — optimal dtype, batch size, and compilation settings per device
- **Best-checkpoint tracking** — `ckpt_best.pt` is saved only when validation loss improves, while `ckpt.pt` holds the latest resumable state
- **Early stopping** (off by default) — when enabled, patience is only armed during the decay phase so a stable-phase plateau can't end the run before decay delivers its drop
- **Mixed precision** (bfloat16/float16) with gradient scaling
- **DDP** support for multi-GPU training
- **Training visualization** — generates loss/LR/MFU plots at end of training

## Training Data

~118M unique tokens from 20,000+ Latin texts spanning:
- **Classical**: Cicero, Caesar, Virgil, Horace, Ovid, Livy, Tacitus, Catullus, Pliny, Terence, Varro
- **Biblical/Patristic**: Vulgata Clementina, Patrologia Latina
- **Medieval**: Charters, chronicles, correspondence
- **Renaissance**: Erasmus' Colloquia, humanist texts
- **Fables & misc**: Aesop (Latin), educational texts

**Split & weighting** (`prepare_latin_weighted.py`): a 90/10 split is made at the **file level from the unique corpus**, so the validation set (~12M tokens) is never duplicated and val loss stays an honest generalization signal. The training set then applies per-text **tier multipliers** (2×/5×/15× by quality/importance, from `corpus_multiplier_manifest.json`), expanding to ~671M weighted training tokens.

## Usage

All scripts run from the `src/` directory.

```bash
cd src/

# 1. Detect hardware and generate config
python3 detect_system.py

# 2. Prepare training data (tokenize corpus)
#    prepare_latin.py        -> trains a fresh BPE tokenizer + encodes the corpus
#    prepare_latin_weighted.py -> reuses the tokenizer, honest file-level val split,
#                                 tier-weighted train set, bounded memory (recommended)
python3 prepare_latin_weighted.py

# 3. Train the model
python3 train_latin.py

# 4. Generate Latin text
python3 sample_latin.py --start="arma uirumque cano"

# 5. Interactive writing assistant
python3 scriptor.py
```

### Command-Line Options

```bash
# Training
python3 train_latin.py --batch_size 16 --max_iters 50000 --wandb

# Looped depth + model-size / regularization overrides
python3 train_latin.py --n_loops 3 --n_layer 6 --n_embd 512 --dropout 0.05

# Sampling
python3 sample_latin.py --start="in principio" --num_samples 5 --temperature 0.7 --top_k 50
```

| Training flag | Purpose |
|---------------|---------|
| `--batch_size`, `--max_iters` | Override batch size / total iterations |
| `--n_loops N` | Recurrence count; effective depth = `n_layer × n_loops` |
| `--no_per_step_loss` | Disable deep supervision across loop steps |
| `--loop_loss_weighting {uniform,linear}` | Weighting of per-step losses |
| `--n_layer`, `--n_embd`, `--dropout` | Override the vocab-derived model defaults |
| `--wandb` | Enable Weights & Biases logging |
| `--no_compile` | Disable `torch.compile` (Windows/Triton issues or debugging) |

## Files

| File | Purpose |
|------|---------|
| `model.py` | GPT model (RoPE, SwiGLU, GQA, RMSNorm, QK-norm, looped depth) |
| `train_latin.py` | Training loop, Muon optimizer, WSD schedule, visualization |
| `prepare_latin.py` | Corpus merging, BPE tokenizer training, binary encoding |
| `prepare_latin_weighted.py` | Honest file-level val split + tier-weighted train set (reuses existing tokenizer) |
| `detect_system.py` | Hardware detection and optimal config generation |
| `sample_latin.py` | Batch text generation from trained model |
| `scriptor.py` | Interactive Latin writing assistant with context memory |

## Requirements

- Python 3.10+
- PyTorch 2.0+ (2.4+ recommended for best MPS/Flash Attention support)
- NumPy
- [tokenizers](https://github.com/huggingface/tokenizers) (HuggingFace, for BPE)
- matplotlib (optional, for training visualization)
- CUDA toolkit (for GPU training / Muon optimizer)

## License

MIT
