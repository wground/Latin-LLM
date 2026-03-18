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

### Model Configuration

The model adapts its size based on vocabulary:

| Vocab Size | Layers | Heads | KV Heads | Embedding Dim | SwiGLU Hidden | Parameters |
|------------|--------|-------|----------|---------------|---------------|------------|
| 8K         | 6      | 6     | 3 (GQA)  | 384           | 1024          | ~20M       |
| 12K        | 7      | 7     | 7 (MHA)  | 448           | 1216          | ~27M       |
| 16K+       | 8      | 8     | 4 (GQA)  | 512           | 1408          | ~32M       |

## Training

### Optimizer: Muon + AdamW Hybrid

On CUDA GPUs, LatinLLM uses a hybrid optimizer for ~2x compute efficiency:

- **[Muon](https://github.com/KellerJordan/Muon)** for all 2D weight matrices (attention projections, MLP weights) — orthogonalizes gradient updates via Newton-Schulz iterations
- **AdamW** for embeddings and 1D parameters (norms)
- Falls back to AdamW-only on MPS (Apple Silicon) and CPU

### Learning Rate: Warmup-Stable-Decay (WSD)

Replaces the older cosine schedule. Three phases:
1. **Warmup** — linear ramp to peak LR
2. **Stable** — constant at peak LR for 80% of training (more useful learning than cosine)
3. **Decay** — linear decay to min LR in the final 20%

### Other Training Features
- **Custom BPE tokenizer** trained on the Latin corpus (16K vocab)
- **Hardware auto-detection** — optimal dtype, batch size, and compilation settings per device
- **Early stopping** with configurable patience
- **Mixed precision** (bfloat16/float16) with gradient scaling
- **DDP** support for multi-GPU training
- **Training visualization** — generates loss/LR/MFU plots at end of training

## Training Data

~117M tokens from 20,000+ Latin texts spanning:
- **Classical**: Cicero, Caesar, Virgil, Horace, Ovid, Livy, Tacitus, Catullus, Pliny, Terence, Varro
- **Biblical/Patristic**: Vulgata Clementina, Patrologia Latina
- **Medieval**: Charters, chronicles, correspondence
- **Renaissance**: Erasmus' Colloquia, humanist texts
- **Fables & misc**: Aesop (Latin), educational texts

Train/val split: 90/10.

## Usage

All scripts run from the `src/` directory.

```bash
cd src/

# 1. Detect hardware and generate config
python3 detect_system.py

# 2. Prepare training data (tokenize corpus)
python3 prepare_latin.py

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

# Sampling
python3 sample_latin.py --start="in principio" --num_samples 5 --temperature 0.7 --top_k 50
```

## Files

| File | Purpose |
|------|---------|
| `model.py` | GPT model (RoPE, SwiGLU, GQA, RMSNorm, QK-norm) |
| `train_latin.py` | Training loop, Muon optimizer, WSD schedule, visualization |
| `prepare_latin.py` | Corpus merging, BPE tokenizer training, binary encoding |
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
