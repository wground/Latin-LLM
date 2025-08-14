# LatinLLM

![Pantheon](assets/pantheon.jpg)

A GPT-based language model trained on classical Latin texts, designed for generating and understanding Latin prose and poetry.

## Model Architecture

LatinLLM is built on a transformer-based architecture with the following key components:

### Core Architecture
- **Base Model**: GPT architecture derived from nanoGPT
- **Normalization**: Configurable LayerNorm or RMSNorm for improved efficiency
- **Attention**: Causal self-attention with Flash Attention support (PyTorch >=2.0)
- **Feed-Forward**: Standard MLP with GELU activation
- **Weight Tying**: Token embeddings tied with output layer weights

### Model Configurations
The model adapts its architecture based on vocabulary size:

| Vocab Size | Layers | Heads | Embedding Dim | Parameters |
|------------|--------|-------|---------------|------------|
| 8K         | 6      | 6     | 384          | ~15M       |
| 12K        | 7      | 7     | 448          | ~25M       |
| 16K+       | 8      | 8     | 512          | ~35M       |

### Training Features
- **Custom Tokenizer**: BPE tokenizer trained specifically on Latin corpus
- **Adaptive Hyperparameters**: Learning rate, batch size, and training schedule adapt to dataset size
- **System Optimization**: Hardware-specific configurations for optimal performance
- **Early Stopping**: Configurable patience and improvement thresholds

## Training Data

The model is trained on a comprehensive corpus of classical Latin texts including:
- **Poetry**: Virgil's Aeneid, Horace's Odes, Catullus, Ovid's Metamorphoses
- **Prose**: Livy's Ab Urbe Condita, Tacitus' Annales, Pliny's Letters

## Usage

### Setup
```bash
# Install dependencies
pip install torch numpy

# Detect optimal system configuration
python detect_system.py

# Prepare training data
python prepare_latin.py

# Train model
python train_latin.py
```

### Training Configuration
The model automatically configures based on detected hardware:
- GPU/CPU selection and optimization
- Mixed precision training (bfloat16/float16)
- Batch size and gradient accumulation
- Memory-efficient settings

### Generation
```bash
# Generate Latin text
python sample_latin.py --prompt "In principio"
```

## Technical Details

### Normalization Layers
- **LayerNorm**: Standard normalization with optional bias
- **RMSNorm**: Root Mean Square normalization for improved efficiency
- Configurable via `use_rmsnorm` parameter

### Optimization
- AdamW optimizer with configurable weight decay
- Cosine learning rate schedule with warmup
- Gradient clipping for training stability
- Hardware-specific optimizations (TF32, fused operations)

### Performance
- Flash Attention for efficient GPU utilization
- Model compilation support (PyTorch 2.0+)
- Distributed training capability
- Model size estimation and FLOPS calculation

## Files

- `train_latin.py` - Main training script with adaptive configuration
- `model.py` - GPT model implementation with RMSNorm support
- `prepare_latin.py` - Data preprocessing and tokenizer training
- `detect_system.py` - Hardware detection and optimization
- `sample_latin.py` - Text generation utilities

## Requirements

- Python 3.8+
- PyTorch 2.0+ (recommended for Flash Attention)
- NumPy
- CUDA toolkit (for GPU training)