"""
Prepare Latin training data with per-text weighting, an honest split, and
bounded memory.

Key differences from prepare_latin.py:
  * Reuses the existing tokenizer in tokenizer_latin/ (no BPE retraining -- that
    crashed on the 3.3 GB duplicated corpus and adds nothing, since duplication
    does not change subword statistics).
  * Splits by FILE from the UNIQUE corpus (originals only). Validation never
    contains duplicated text, so val loss stays a real generalization signal.
  * Applies the tier multipliers (from corpus_multiplier_manifest.json) to the
    TRAIN set only, encoding one file at a time and streaming to train.bin, so
    peak memory stays small regardless of corpus size.

The physical .__dup__ copies are NOT read here; weighting is reconstructed from
the manifest. They can be deleted to reclaim disk.
"""
import os
import re
import glob
import json
import pickle
import random
import numpy as np
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA = os.path.join(HERE, "Training Data")
TOKENIZER_DIR = os.path.join(HERE, "tokenizer_latin")
MANIFEST = os.path.join(HERE, "corpus_multiplier_manifest.json")
OUTPUT_DIR = Path(os.path.join(HERE, "gpt_data_latin"))
MARKER_RE = re.compile(r"\.__dup\d+__\.txt$")
VAL_FRACTION = 0.10
SEED = 1337


def clean_ocr_artifacts(text: str) -> str:
    text = re.sub(r"_{3,}", "", text)
    text = re.sub(r"-{5,}", "", text)
    text = re.sub(r"[{}\[\]]{2,}", "", text)
    text = re.sub(r" _ ", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = re.sub(r"^[\s\.\-_\*=]+$", "", text, flags=re.MULTILINE)
    return text


def load_multipliers():
    """relpath -> tier multiplier, from the manifest. Default 1 if absent."""
    if not os.path.exists(MANIFEST):
        print(f"  (no manifest at {MANIFEST}; every file weighted 1x)")
        return {}
    with open(MANIFEST, encoding="utf-8") as fh:
        data = json.load(fh)
    return {e["original"]: int(e["tier"]) for e in data.get("files", [])}


def main():
    print("=" * 64)
    print("    Latin data prep — weighted train, honest val split")
    print("=" * 64)

    tok = ByteLevelBPETokenizer(
        os.path.join(TOKENIZER_DIR, "vocab.json"),
        os.path.join(TOKENIZER_DIR, "merges.txt"),
    )
    vocab_size = tok.get_vocab_size()
    print(f"Reusing tokenizer ({vocab_size} tokens) from {TOKENIZER_DIR}")

    all_txt = glob.glob(os.path.join(TRAINING_DATA, "**", "*.txt"), recursive=True)
    originals = sorted(f for f in all_txt if not MARKER_RE.search(os.path.basename(f)))
    if not originals:
        raise SystemExit(f"No original .txt files in {TRAINING_DATA}")
    mult = load_multipliers()
    print(f"Found {len(originals):,} unique source files")

    # File-level split from UNIQUE corpus only.
    rng = random.Random(SEED)
    shuffled = originals[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * VAL_FRACTION))
    val_files = set(shuffled[:n_val])
    train_files = shuffled[n_val:]
    print(f"Split: {len(train_files):,} train files / {len(val_files):,} val files")

    OUTPUT_DIR.mkdir(exist_ok=True)

    def encode_file(path):
        with open(path, "r", encoding="utf-8") as f:
            text = clean_ocr_artifacts(f.read().strip())
        if not text:
            return None
        ids = np.array(tok.encode(text + "\n\n").ids, dtype=np.uint16)
        return ids

    # --- val.bin : unique text, weight 1, streamed ---
    val_tokens = 0
    print("Encoding validation set (unique, 1x)...")
    with open(OUTPUT_DIR / "val.bin", "wb") as out:
        for path in sorted(val_files):
            ids = encode_file(path)
            if ids is None:
                continue
            out.write(ids.tobytes())
            val_tokens += len(ids)

    # --- train.bin : weighted by tier, streamed ---
    train_tokens = 0
    weighted_files = 0
    tier_counts = {}
    print("Encoding train set (weighted by tier)...")
    with open(OUTPUT_DIR / "train.bin", "wb") as out:
        for path in train_files:
            ids = encode_file(path)
            if ids is None:
                continue
            rel = os.path.relpath(path, TRAINING_DATA)
            m = mult.get(rel, 1)
            tier_counts[m] = tier_counts.get(m, 0) + 1
            payload = ids.tobytes()
            for _ in range(m):
                out.write(payload)
            train_tokens += len(ids) * m
            weighted_files += 1

    meta = {
        "vocab_size": vocab_size,
        "tokenizer_config": {
            "vocab_file": os.path.join(TOKENIZER_DIR, "vocab.json"),
            "merges_file": os.path.join(TOKENIZER_DIR, "merges.txt"),
            "type": "ByteLevelBPE",
        },
        "data_stats": {
            "unique_files": len(originals),
            "train_files": len(train_files),
            "val_files": len(val_files),
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
            "split": "file-level, val held out from unique corpus",
            "weighting": "tier multipliers applied to train only",
            "tier_file_counts": tier_counts,
        },
    }
    with open(OUTPUT_DIR / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    def m(n):
        return f"{n/1e6:,.1f}M"

    print("-" * 64)
    print(f"Tier file counts (train): {dict(sorted(tier_counts.items()))}")
    print(f"Train tokens (weighted)  : {m(train_tokens)}")
    print(f"Val tokens (unique, 1x)  : {m(val_tokens)}")
    unique_train = sum(1 for _ in train_files)
    print(f"Wrote {OUTPUT_DIR}/train.bin, val.bin, meta.pkl")
    print("-" * 64)


if __name__ == "__main__":
    main()
