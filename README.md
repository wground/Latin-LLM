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

~158M unique tokens from 22,652 Latin texts. Sources and licences are recorded per
document in `src/fetch_manifest.jsonl`:

| Source | Licence | Contribution |
|---|---|---|
| Original collection (Wikisource scans, Patrologia Latina, …) | mixed | 20,147 docs |
| [The Latin Library](https://www.thelatinlibrary.com/) via [cltk/lat_text_latin_library](https://github.com/cltk/lat_text_latin_library) | Public Domain Mark 1.0 | 2,101 docs, 93.9M chars |
| [Perseus canonical-latinLit](https://github.com/PerseusDL/canonical-latinLit) | CC BY-SA 4.0 | 404 docs, 46.5M chars |

The added sources exist to fix a specific bottleneck: classical Latin was only ~3.4M
tokens, so *any* attempt to weight it heavily re-read the same text hundreds of times.
It is now 12.5% of the corpus by bytes (~18M tokens), which makes a classical-heavy
mixture trainable rather than memorizable.

Spanning:
- **Classical**: Cicero, Caesar, Virgil, Horace, Ovid, Livy, Tacitus, Catullus, Pliny, Terence, Varro
- **Biblical/Patristic**: Vulgata Clementina, Patrologia Latina
- **Medieval**: Charters, chronicles, correspondence
- **Renaissance**: Erasmus' Colloquia, humanist texts
- **Fables & misc**: Aesop (Latin), educational texts

**Split & weighting** (`prepare_corpus.py`): the split is made **by work**, not by file.
Nearly half the corpus is individual scanned pages (`Pagina_<Work>.djvu_<n>.txt`) and most
of the rest is one chapter or book per file, so a random file-level split scattered pages of
the *same book* across train and val. Documents are now grouped into works (and into
exact-duplicate clusters) and whole groups are assigned to one side, filled to a 10% byte
target. Documents are separated by `<|endoftext|>`, and the per-text **tier multipliers**
(2×/5×/15×, from `corpus_multiplier_manifest.json`) are applied as *sampling weights* at
training time rather than by physically duplicating text into the binary.

Every document's id, work group, split assignment, hashes and cleaning flags are recorded in
`corpus_ledger.jsonl`, along with classification labels (`classify.py`):

| Signal | Coverage | How |
|---|---|---|
| `source_type` | 100% | scan page vs text |
| `form` (prose/verse) | 92% | line-length statistics; abstains on scan pages, whose printed line breaks mimic verse |
| `ocr_quality`, `non_latin_per_1k` | 100% | character and function-word statistics |
| `genre` | 60% | title keyword rules (exegesis, letters, sermons, history, law, liturgy, …) |
| `era` | 44% | curated author and work tables, plus Patrologia Latina volume numbers (PL is chronological) |

Anything not confidently derivable is labelled `unknown` rather than guessed, and multi-era
collections (Gallia Christiana, Denzinger, papal registers) are labelled `compilation`
rather than forced into a single era — a wrong label would silently corrupt any per-era
comparison. Editors and series (`ed. Migne`, `PL 086`) are explicitly excluded from being
read as authors.

**The corpus is predominantly patristic and medieval, not classical.** Rabanus Maurus
(19 MB) and Jerome (17 MB) each outweigh Cicero (1.8 MB) by an order of magnitude. By
labelled bytes: ~15% late antique, ~13% early medieval, ~5% high medieval, ~3% classical.
Claims about the model's "Latin" should be read in that light.

Breakdowns are available at evaluation time:

```bash
python3 src/evaluate.py --split val --by era genre form
```

### Corpus mixture

Because weighting happens in the sampler, the mixture is a config choice, not a property of
`train.bin` — the binary never changes size and never has to be re-encoded. Changing the
mixture takes under a second:

```bash
python3 src/prepare_corpus.py --reweight --weight-profile canonical
```

| Profile | Intent |
|---|---|
| `uniform` | Every token equally likely. The control condition. |
| `manifest` | The hand-made tier file only (default; historical behaviour). |
| `canonical` | Emphasise historically important works; long tail kept at low weight as background linguistic support. |
| `balanced` | Flatten the corpus's heavy medieval skew without privileging a canon. |

`weight = tier × era × genre × canon × quality`, each factor overridable:

```bash
python3 src/prepare_corpus.py --reweight --weight-profile canonical \
    --era-weight classical=8 neo_latin=3 --genre-weight poetry=4 liturgy=0.2 \
    --canon-boost 6 --max-weight 30 --min-quality 0.85 --budget-tokens 2.5e9
```

**Read the `epochs` column before training.** The factors compound, so heavy emphasis plus
a long run means seeing the same text dozens or hundreds of times — which memorizes it
rather than teaching its register. `--max-weight` caps this and the tool warns past ~40
epochs.

The shipped mixture is `canonical` with `--max-weight 10`: classical material is 47.7% of
sampled tokens (from 12.5% of the corpus), poetry gets 3.4× emphasis, and exegesis,
sermons and canon law drop to 2–4 epochs as background linguistic support. That lands
classical at ~40 epochs for a **1.5B-token budget** (~60k iterations at 24,576 tokens/iter).
A longer run needs either lower emphasis or more classical text — corpus size sets a hard
floor, since even *zero* emphasis puts classical at ~17 epochs over a 2.458B-token run.

Two other knobs, both off by default because they are destructive:

```bash
--orthography {none,conservative,classical,modern}   # macrons, u/v, i/j
--max-fragment-score 0.7                             # drop stubs, incipits, index pages
```

Orthography standardization is irreversible in the encoded data and changes tokenization,
so checkpoints trained at different levels are not comparable. `conservative` strips
macrons only; `classical` folds v→u and j→i.

## Usage

Scripts resolve all paths relative to `src/`, so they behave identically no matter which
directory you invoke them from.

```bash
# 1. Detect hardware and generate config
python3 src/detect_system.py

# 2. Build the dataset (work-level split, EOS separators, ledger)
python3 src/prepare_corpus.py
python3 src/prepare_corpus.py --dry-run     # report the split without writing

# 3. Train the model (--init is required, see below)
python3 src/train_latin.py --init scratch

# 4. Score a checkpoint honestly
python3 src/evaluate.py --split val

# 5. Generate Latin text
python3 src/sample_latin.py --start="arma uirumque cano"

# 6. Interactive writing assistant
python3 src/scriptor.py
```

### Reporting losses

For a looped model the training objective is a weighted average over the `n_loops`
readouts, but inference only ever uses the **final** readout. Those are different numbers,
so `exp(objective)` is not perplexity. Training and `evaluate.py` both report
final-readout cross-entropy (and bits/byte, which stays comparable across tokenizers);
checkpoint selection uses the final readout too. Evaluation windows are fixed by
`eval_seed`, drawn from a dedicated RNG so that evaluating does not perturb the training
data stream.

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
| `--init {scratch,resume,finetune}` | **Required.** How to initialize (see below) |
| `--batch_size`, `--block_size`, `--max_iters` | Override batch size / context / iterations |
| `--n_loops N` | Recurrence count; effective depth = `n_layer × n_loops` |
| `--no_per_step_loss` | Disable deep supervision across loop steps |
| `--loop_loss_weighting {uniform,linear,final_only}` | Weighting of per-step losses (default `linear`) |
| `--sampling {weighted,uniform}` | Apply corpus tier multipliers, or ignore them (control) |
| `--n_layer`, `--n_embd`, `--n_head`, `--n_kv_head`, `--dropout` | Override model defaults |
| `--eval_iters`, `--eval_interval` | Evaluation batch count / frequency |
| `--data-dir`, `--out-dir`, `--tokenizer-dir`, `--device` | Path and device overrides |
| `--wandb` | Enable Weights & Biases logging |
| `--no_compile` | Disable `torch.compile` (Windows/Triton issues or debugging) |

### Initialization modes

`--init` is required and explicit. It previously defaulted to "resume" whenever
`out-dir/ckpt.pt` happened to exist, which meant any run — including a smoke test — would
silently continue the real training run and then overwrite its checkpoint.

- `scratch` — new model.
- `resume` — continue a run: iteration count, LR schedule, optimizer, RNG and metrics all
  restored. Refuses to start if the config's architecture disagrees with the checkpoint's,
  rather than training a checkpoint-shaped model on config-shaped data.
- `finetune` — load the weights, adopt the checkpoint's architecture, restart the schedule.

Use `--out-dir` to point experimental runs at a throwaway directory.

## Files

| File | Purpose |
|------|---------|
| `model.py` | GPT model (RoPE, SwiGLU, GQA, RMSNorm, QK-norm, looped depth, KV cache) |
| `train_latin.py` | Training loop, Muon optimizer, WSD schedule, visualization |
| `prepare_corpus.py` | Work-level split, EOS separators, ledger, sampling weights |
| `evaluate.py` | Final-readout CE, bits/byte, bootstrap CIs, contamination probe |
| `paths.py` | Single source of truth for every artifact location |
| `artifacts.py` | Shared loading of system config, tokenizer and checkpoints |
| `detect_system.py` | Hardware detection and optimal config generation |
| `sample_latin.py` | Batch text generation from trained model |
| `scriptor.py` | Interactive Latin writing assistant with context memory |
| `tests/test_pipeline.py` | CPU tests for split, EOS, eval determinism, KV cache |
| `prepare_latin.py`, `prepare_latin_weighted.py` | Superseded by `prepare_corpus.py`; kept for reference |

## Current results

The 31.8M-parameter looped checkpoint (`n_loops=3`, 512 context, 16k vocab, iter 98.5k),
re-measured with `evaluate.py` on fixed windows:

| Metric | Value |
|---|---|
| Final-readout CE (work-held-out val) | **3.68** (95% CI 3.66–3.71) |
| Final-readout perplexity | 39.7 |
| Bits per byte | 1.18 |
| Longest verbatim training match in generated text | 32 tokens (median 14) |

Two things worth knowing about the older reported figure of 3.896:

1. **It was not perplexity.** It averaged the three loop readouts, while inference uses only
   the final one. The honest final-readout CE on that same validation set is ≈3.914.
2. **It was a lucky draw.** Replicating the original random-window evaluation across 10
   seeds gives mean 3.926, sd 0.015 — and 3.896 sits below all ten. Evaluation is now
   deterministic, so "best" tracks the model rather than the sample.

The old validation split leaked at the work level (55.7% of val files shared a work with
training). Measuring its effect directly — leaking vs held-out subsets of the *same*
validation set — showed CE 3.9026 vs 3.9145, i.e. essentially none. At this scale the model
is not memorizing individual works. The work-level split is still the correct structure and
matters more as models grow, but it did not inflate the published number.

## Requirements

- Python 3.10+
- PyTorch 2.0+ (2.4+ recommended for best MPS/Flash Attention support)
- NumPy
- [tokenizers](https://github.com/huggingface/tokenizers) (HuggingFace, for BPE)
- matplotlib (optional, for training visualization)
- CUDA toolkit (for GPU training / Muon optimizer)

## License

MIT
