"""
Single source of truth for every filesystem location in the project.

Everything is anchored on ``Path(__file__)``, never on the current working directory.
Before this module existed the scripts resolved "gpt_data_latin", "tokenizer_latin" and
"out-latin" as *relative* paths, so running the same script from the repo root and from
src/ silently picked up different data, different tokenizers and different checkpoints.

The tokenizer location also used to be baked into meta.pkl as an absolute path, which made
the pickle unusable on any other machine. ``tokenizer_files()`` below resolves tokenizer
paths relative to the meta.pkl that names them, and still accepts the legacy absolute form
so old artifacts keep loading.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# --- Anchors ---

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# --- Default artifact locations (all inside src/) ---

DATA_DIR = SRC_DIR / "gpt_data_latin"
TOKENIZER_DIR = SRC_DIR / "tokenizer_latin"
OUT_DIR = SRC_DIR / "out-latin"
CORPUS_DIR = SRC_DIR / "Training Data"

SYSTEM_CONFIG = SRC_DIR / "latin_training_config.json"
MULTIPLIER_MANIFEST = SRC_DIR / "corpus_multiplier_manifest.json"

# --- Well-known filenames ---

META_NAME = "meta.pkl"
LEDGER_NAME = "corpus_ledger.jsonl"
TRAIN_BIN = "train.bin"
VAL_BIN = "val.bin"

CKPT_LATEST = "ckpt.pt"
CKPT_BEST = "ckpt_best.pt"
CKPT_FINAL = "ckpt_final.pt"

VOCAB_NAME = "vocab.json"
MERGES_NAME = "merges.txt"


# --- Tokenizer path handling ---

def make_tokenizer_config(tokenizer_dir: Path, meta_dir: Path) -> Dict[str, str]:
    """Build the ``tokenizer_config`` dict to store in meta.pkl.

    Paths are stored *relative to meta.pkl* so the artifact stays portable. Falls back to
    an absolute path only when the tokenizer lives on a different drive/root than the data
    directory, where a relative path cannot be expressed.
    """
    tokenizer_dir = Path(tokenizer_dir).resolve()
    meta_dir = Path(meta_dir).resolve()
    try:
        rel = Path(os.path.relpath(tokenizer_dir, meta_dir))
    except ValueError:
        rel = tokenizer_dir
    return {
        "vocab_file": (rel / VOCAB_NAME).as_posix(),
        "merges_file": (rel / MERGES_NAME).as_posix(),
        "type": "ByteLevelBPE",
    }


def tokenizer_files(meta: Dict[str, Any], meta_path: Path) -> Tuple[Path, Path]:
    """Resolve (vocab_file, merges_file) from a loaded meta.pkl.

    Handles three cases, in order:
      1. paths relative to meta.pkl (the format written by ``make_tokenizer_config``),
      2. legacy absolute paths that still exist on this machine,
      3. legacy paths whose *directory* has moved -- fall back to the default
         ``TOKENIZER_DIR`` so old metadata keeps working after a checkout elsewhere.
    """
    meta_dir = Path(meta_path).resolve().parent
    cfg = meta.get("tokenizer_config", {})

    resolved = []
    for key, default_name in (("vocab_file", VOCAB_NAME), ("merges_file", MERGES_NAME)):
        raw = cfg.get(key)
        candidates = []
        if raw:
            raw_path = Path(raw)
            if raw_path.is_absolute():
                candidates.append(raw_path)
            else:
                candidates.append(meta_dir / raw_path)
            # A legacy absolute path from another machine still tells us the filename.
            candidates.append(TOKENIZER_DIR / raw_path.name)
        candidates.append(TOKENIZER_DIR / default_name)

        for cand in candidates:
            if cand.exists():
                resolved.append(cand.resolve())
                break
        else:
            raise FileNotFoundError(
                f"Could not resolve tokenizer {key!r} from {meta_path}. Tried: "
                + ", ".join(str(c) for c in candidates)
            )
    return resolved[0], resolved[1]


# --- Hashing (used to stamp checkpoints with data/tokenizer provenance) ---

def file_sha1(path: Path, max_bytes: Optional[int] = None) -> str:
    """SHA-1 of a file. ``max_bytes`` hashes only a prefix -- enough to fingerprint a
    multi-GB .bin without reading all of it."""
    h = hashlib.sha1()
    remaining = max_bytes
    with open(path, "rb") as fh:
        while True:
            chunk_size = 1 << 20 if remaining is None else min(1 << 20, remaining)
            if chunk_size <= 0:
                break
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return h.hexdigest()


def dir_sha1(paths) -> str:
    """Stable combined hash over several files (order-independent)."""
    digests = sorted(file_sha1(Path(p)) for p in paths)
    return hashlib.sha1("".join(digests).encode()).hexdigest()


# --- Shared CLI wiring ---

def add_path_args(parser, *, include_checkpoint: bool = False) -> None:
    """Attach the standard path/device overrides to an argparse parser.

    Every script that touches data or checkpoints should expose these so no run is ever
    forced to guess -- and, critically, so tests can point at a throwaway --out-dir instead
    of the single unbacked-up real checkpoint directory.
    """
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help=f"Directory holding train.bin/val.bin/meta.pkl (default: {DATA_DIR})")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help=f"Directory for checkpoints and plots (default: {OUT_DIR})")
    parser.add_argument("--tokenizer-dir", type=Path, default=TOKENIZER_DIR,
                        help=f"Directory holding vocab.json/merges.txt (default: {TOKENIZER_DIR})")
    parser.add_argument("--device", type=str, default=None,
                        help="Override the detected device (cuda|mps|cpu)")
    if include_checkpoint:
        parser.add_argument("--checkpoint", type=Path, default=None,
                            help=f"Checkpoint file to load (default: <out-dir>/{CKPT_BEST})")
