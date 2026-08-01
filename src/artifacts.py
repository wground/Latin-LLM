"""
Loading of the artifacts every entry point needs: the system config, the tokenizer, and
trained checkpoints.

``load_system_config`` and ``load_latin_tokenizer`` were previously duplicated verbatim in
sample_latin.py and scriptor.py, both resolving paths from the current working directory.
They live here once so a path fix applies everywhere.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from tokenizers import ByteLevelBPETokenizer

import paths

# Fallback used when detect_system.py has never been run.
_CPU_FALLBACK = {
    "recommended_config": {
        "device": "cpu",
        "dtype": "float32",
        "compile": False,
        "backend": "cpu",
        "enable_tf32": False,
    }
}


def load_system_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the hardware config emitted by detect_system.py."""
    config_path = Path(config_path) if config_path is not None else paths.SYSTEM_CONFIG
    if not config_path.exists():
        print(f"⚠️  Config file {config_path} not found!")
        print("Run 'python3 detect_system.py' first to generate system config.")
        print("Using default CPU configuration...")
        return _CPU_FALLBACK

    with open(config_path, "r") as f:
        config = json.load(f)
    print(f"✅ Loaded system config from {config_path}")
    return config


def load_meta(data_dir: Optional[Path] = None) -> Tuple[Dict[str, Any], Path]:
    """Load meta.pkl and return it alongside its own path (needed to resolve relative
    tokenizer paths)."""
    data_dir = Path(data_dir) if data_dir is not None else paths.DATA_DIR
    meta_path = data_dir / paths.META_NAME
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No {paths.META_NAME} at {meta_path}. Run prepare_corpus.py first."
        )
    with open(meta_path, "rb") as f:
        return pickle.load(f), meta_path


def load_latin_tokenizer(
    data_dir: Optional[Path] = None,
) -> Tuple[Callable[[str], list], Callable[[list], str], Dict[str, Any]]:
    """Return ``(encode, decode, meta)`` for the corpus tokenizer.

    Tokenizer files are resolved relative to meta.pkl, so the artifact tree is portable
    across machines and checkouts.
    """
    meta, meta_path = load_meta(data_dir)
    vocab_file, merges_file = paths.tokenizer_files(meta, meta_path)

    print("✅ Loading custom Latin tokenizer")
    print(f"   Vocabulary size: {meta['vocab_size']}")
    print(f"   Files: {vocab_file}")

    tokenizer = ByteLevelBPETokenizer(str(vocab_file), str(merges_file))

    def encode(text: str) -> list:
        return tokenizer.encode(text).ids

    def decode(ids: list) -> str:
        return tokenizer.decode(ids)

    return encode, decode, meta


def special_token_ids(meta: Dict[str, Any]) -> Dict[str, int]:
    """Special token ids recorded at prepare time.

    Falls back to the ByteLevelBPE convention (eos=0, pad=1) used by this project's
    tokenizer when the metadata predates the field.
    """
    recorded = meta.get("special_tokens")
    if isinstance(recorded, dict) and "eos" in recorded:
        return recorded
    return {"eos": 0, "pad": 1}


def resolve_checkpoint(out_dir: Path, checkpoint: Optional[Path] = None,
                       prefer_best: bool = True) -> Path:
    """Pick which checkpoint file to load.

    Defaults to ckpt_best.pt -- the previous default was the rolling ckpt.pt, which is
    whatever the last eval interval happened to write rather than the best model.
    """
    if checkpoint is not None:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
        return ckpt_path

    out_dir = Path(out_dir)
    order = [paths.CKPT_BEST, paths.CKPT_LATEST] if prefer_best else [paths.CKPT_LATEST, paths.CKPT_BEST]
    for name in order:
        cand = out_dir / name
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"No checkpoint found in {out_dir} (looked for {', '.join(order)}). "
        "Train a model first with train_latin.py."
    )


def load_model(ckpt_path: Path, device: str, model_cls, config_cls):
    """Load a GPT from a checkpoint, stripping any torch.compile prefix."""
    print(f"Loading model from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    gptconf = config_cls(**checkpoint["model_args"])
    model = model_cls(gptconf)

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    best = checkpoint.get("best_val_loss")
    if best is not None:
        print(f"Model loaded. Best recorded val loss: {float(best):.4f} "
              f"(iter {checkpoint.get('iter_num', '?')})")
    return model, checkpoint
