"""
Score a trained checkpoint on a held-out split.

This exists because the number the training loop used to print was not what it appeared to
be. For a looped model, ``forward`` averaged the cross-entropy of every loop readout, but
inference only ever uses the final one -- so the reported loss was not comparable to a
conventional model's, and exp(loss) was not its perplexity.

What this reports instead:
  * final-readout CE and perplexity  -- the number that actually describes the model
  * per-readout CE                   -- how much each loop step contributes
  * bits per byte                    -- comparable across different tokenizers
  * bootstrap confidence intervals   -- so two runs can be told apart honestly
  * optional per-stratum breakdown   -- scan pages vs clean text, by work group

Usage:
    python3 evaluate.py --data-dir gpt_data_latin_v2 --split val
    python3 evaluate.py --data-dir gpt_data_latin --split val --label "old leaky split"
    python3 evaluate.py --compare gpt_data_latin gpt_data_latin_v2
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

import paths
from artifacts import load_system_config, resolve_checkpoint
from model import GPTConfig, GPT


def load_split(data_dir: Path, split: str) -> np.memmap:
    f = Path(data_dir) / f"{split}.bin"
    if not f.exists():
        raise FileNotFoundError(f"No {split}.bin in {data_dir}")
    return np.memmap(f, dtype=np.uint16, mode="r")


def bytes_per_token(data_dir: Path, split: str) -> Optional[float]:
    """Bytes per token for this split, needed to convert CE into bits/byte."""
    meta_path = Path(data_dir) / paths.META_NAME
    if not meta_path.exists():
        return None
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    stats = meta.get("data_stats", {})
    b, t = stats.get(f"{split}_bytes"), stats.get(f"{split}_tokens")
    if b and t:
        return b / t
    # Older metadata recorded chars rather than bytes; close enough for Latin (mostly ASCII).
    b, t = stats.get(f"{split}_chars"), stats.get(f"{split}_tokens")
    return (b / t) if (b and t) else None


@torch.no_grad()
def score(model, data: np.memmap, block_size: int, batch_size: int, n_batches: int,
          device: str, ctx, seed: int = 1337) -> Dict[str, object]:
    """Score fixed windows drawn from a dedicated RNG.

    Windows are deterministic given (seed, block_size, batch_size, n_batches), so repeated
    runs and different checkpoints are compared on exactly the same text.
    """
    high = len(data) - block_size - 1
    if high <= 0:
        raise ValueError("Split is shorter than block_size")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, high, size=n_batches * batch_size, dtype=np.int64)

    final_losses: List[float] = []
    obj_losses: List[float] = []
    per_step: List[List[float]] = []

    for k in range(n_batches):
        chunk = starts[k * batch_size:(k + 1) * batch_size]
        idx = chunk[:, None] + np.arange(block_size + 1, dtype=np.int64)[None, :]
        seq = torch.from_numpy(data[idx].astype(np.int64))
        x = seq[:, :-1].contiguous().to(device)
        y = seq[:, 1:].contiguous().to(device)
        with ctx:
            _, loss, aux = model(x, y)
        obj_losses.append(float(loss))
        final_losses.append(float(aux.get("final_loss", loss)))
        if aux.get("step_losses"):
            per_step.append([float(s) for s in aux["step_losses"]])

    final = np.array(final_losses)
    # Bootstrap over batches: a plain mean hides how noisy a 150-batch estimate is.
    boot = np.array([np.mean(rng.choice(final, size=len(final), replace=True))
                     for _ in range(2000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    out = {
        "final_ce": float(final.mean()),
        "final_ce_ci95": [float(lo), float(hi)],
        "final_ppl": float(math.exp(final.mean())),
        "objective_ce": float(np.mean(obj_losses)),
        "n_batches": n_batches,
        "n_tokens_scored": int(n_batches * batch_size * block_size),
    }
    if per_step:
        arr = np.array(per_step)
        out["per_readout_ce"] = [float(v) for v in arr.mean(axis=0)]
    return out


def evaluate_one(args, data_dir: Path, label: str) -> Dict[str, object]:
    system_config = load_system_config()
    rec = system_config["recommended_config"]
    device = args.device or rec["device"]
    dtype = "float32" if device == "cpu" else rec["dtype"]

    ckpt_path = resolve_checkpoint(args.out_dir, args.checkpoint)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    sd = checkpoint["model"]
    for k in list(sd.keys()):
        if k.startswith("_orig_mod."):
            sd[k[len("_orig_mod."):]] = sd.pop(k)
    model.load_state_dict(sd)
    model.eval().to(device)

    device_type = device if device in ("cuda", "mps") else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
               "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(
        device_type=device_type, dtype=ptdtype)

    data = load_split(data_dir, args.split)
    block_size = args.block_size or gptconf.block_size
    result = score(model, data, block_size, args.batch_size, args.n_batches,
                   device, ctx, seed=args.seed)

    bpt = bytes_per_token(data_dir, args.split)
    if bpt:
        result["bits_per_byte"] = result["final_ce"] / math.log(2) / bpt
        result["bytes_per_token"] = bpt

    if args.by:
        ledger = load_ledger(data_dir, args.split)
        result["strata"] = {
            f: score_by_stratum(model, data, ledger, f, block_size, args.batch_size,
                                max(4, args.n_batches // 8), device, ctx, seed=args.seed)
            for f in args.by
        }

    if args.contamination:
        print("  running contamination probe (this scans the training stream)...")
        result["contamination"] = contamination_probe(
            model, data_dir, device, ctx, n_samples=args.contamination)

    result.update({
        "label": label,
        "data_dir": str(data_dir),
        "split": args.split,
        "checkpoint": str(ckpt_path),
        "checkpoint_sha1": paths.file_sha1(ckpt_path, max_bytes=1 << 20),
        "iter_num": checkpoint.get("iter_num"),
        "reported_best_val_loss": float(checkpoint["best_val_loss"])
        if checkpoint.get("best_val_loss") is not None else None,
        "block_size": block_size,
        "n_loops": gptconf.n_loops,
        "loop_loss_weighting": gptconf.loop_loss_weighting,
    })
    return result


def load_ledger(data_dir: Path, split: str) -> List[Dict]:
    f = Path(data_dir) / paths.LEDGER_NAME
    if not f.exists():
        raise FileNotFoundError(
            f"No {paths.LEDGER_NAME} in {data_dir}; rebuild with prepare_corpus.py")
    rows = []
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("split") == split and "token_offset" in r:
                rows.append(r)
    return rows


@torch.no_grad()
def score_by_stratum(model, data: np.memmap, ledger: List[Dict], field: str,
                     block_size: int, batch_size: int, max_batches: int,
                     device: str, ctx, seed: int = 1337) -> Dict[str, Dict]:
    """Final-readout CE broken down by a ledger field (era, genre, form, source_type).

    Only documents at least one full window long are used, so every scored window lies
    entirely inside a single stratum -- otherwise a window straddling a document boundary
    would be credited to whichever stratum it started in.
    """
    groups: Dict[str, List[Dict]] = {}
    for r in ledger:
        if r.get("tokens", 0) > block_size + 1:
            groups.setdefault(str(r.get(field, "unknown")), []).append(r)

    rng = np.random.default_rng(seed)
    out: Dict[str, Dict] = {}
    for name, docs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        starts = []
        for _ in range(max_batches * batch_size):
            d = docs[rng.integers(0, len(docs))]
            span = d["tokens"] - block_size - 1
            starts.append(d["token_offset"] + int(rng.integers(0, max(1, span))))
        starts = np.array(starts, dtype=np.int64)
        starts = starts[starts + block_size + 1 <= len(data)]
        if len(starts) < batch_size:
            continue

        losses = []
        n = len(starts) // batch_size
        for k in range(n):
            chunk = starts[k * batch_size:(k + 1) * batch_size]
            idx = chunk[:, None] + np.arange(block_size + 1, dtype=np.int64)[None, :]
            seq = torch.from_numpy(data[idx].astype(np.int64))
            x = seq[:, :-1].contiguous().to(device)
            y = seq[:, 1:].contiguous().to(device)
            with ctx:
                _, loss, aux = model(x, y)
            losses.append(float(aux.get("final_loss", loss)))
        arr = np.array(losses)
        out[name] = {
            "final_ce": float(arr.mean()),
            "sem": float(arr.std() / math.sqrt(len(arr))),
            "n_docs": len(docs),
            "n_batches": len(arr),
        }
    return out


@torch.no_grad()
def contamination_probe(model, data_dir: Path, device: str, ctx, n_samples: int = 5,
                        n_tokens: int = 200, top_k: int = 50, seed: int = 1337
                        ) -> Dict[str, object]:
    """Longest exact token n-gram shared between generated text and the training set.

    A small model that has genuinely learned Latin should produce novel token sequences; a
    model that has memorized its corpus will reproduce long verbatim spans. Matching is
    exact and anchored on the RAREST token of each candidate span, so the search stays cheap
    even against a 100M-token training stream.
    """
    train = np.asarray(np.memmap(Path(data_dir) / paths.TRAIN_BIN, dtype=np.uint16, mode="r"))
    counts = np.bincount(train, minlength=int(train.max()) + 1)

    meta_path = Path(data_dir) / paths.META_NAME
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    eos = meta.get("special_tokens", {}).get("eos", 0)

    torch.manual_seed(seed)
    prompt = torch.tensor([[eos]], dtype=torch.long, device=device)

    def longest_match(seq: np.ndarray) -> int:
        """Longest suffix-anchored exact match of any window of `seq` in `train`."""
        best = 0
        for start in range(len(seq)):
            n = best + 1
            if start + n > len(seq):
                break
            # Anchor on the rarest token in the candidate window.
            window = seq[start:start + n]
            anchor_i = int(np.argmin(counts[window]))
            anchor_tok = int(window[anchor_i])
            cand = np.flatnonzero(train == anchor_tok) - anchor_i
            cand = cand[(cand >= 0) & (cand + len(seq) - start <= len(train))]
            for c in cand:
                k = 0
                while (start + k < len(seq)
                       and train[c + k] == seq[start + k]):
                    k += 1
                best = max(best, k)
        return best

    results = []
    for i in range(n_samples):
        out = model.generate(prompt.clone(), n_tokens, temperature=0.8, top_k=top_k,
                             eos_token_id=eos)
        seq = out[0, 1:].cpu().numpy().astype(np.uint16)
        if len(seq) == 0:
            continue
        results.append(longest_match(seq))

    return {
        "longest_exact_train_ngram_tokens": results,
        "max": max(results) if results else 0,
        "median": float(np.median(results)) if results else 0.0,
    }


def print_result(r: Dict[str, object]):
    print(f"\n=== {r['label']} ===")
    print(f"  data            : {r['data_dir']} [{r['split']}]")
    print(f"  checkpoint      : {Path(r['checkpoint']).name} (iter {r['iter_num']})")
    lo, hi = r["final_ce_ci95"]
    print(f"  final-readout CE: {r['final_ce']:.4f}  (95% CI {lo:.4f}–{hi:.4f})")
    print(f"  final-readout PPL: {r['final_ppl']:.2f}")
    if r.get("per_readout_ce"):
        steps = ", ".join(f"{v:.4f}" for v in r["per_readout_ce"])
        print(f"  per-readout CE  : [{steps}]")
        print(f"  loop-avg objective: {r['objective_ce']:.4f}   <-- what training used to report")
    if "bits_per_byte" in r:
        print(f"  bits/byte       : {r['bits_per_byte']:.4f}")
    print(f"  tokens scored   : {r['n_tokens_scored']:,}")
    if r.get("contamination"):
        c = r["contamination"]
        print(f"  longest verbatim training match: {c['max']} tokens "
              f"(median {c['median']:.0f} across samples)")
    for field, strata in (r.get("strata") or {}).items():
        print(f"\n  --- final-readout CE by {field} ---")
        for name, s in sorted(strata.items(), key=lambda kv: kv[1]["final_ce"]):
            print(f"      {s['final_ce']:6.3f} ±{s['sem']:.3f}   {name:24s} "
                  f"({s['n_docs']:,} docs)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="val")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_batches", type=int, default=200)
    p.add_argument("--block_size", type=int, default=None,
                   help="Defaults to the checkpoint's context length")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--label", default=None)
    p.add_argument("--compare", nargs=2, metavar=("DIR_A", "DIR_B"),
                   help="Score the same checkpoint on two data dirs and report the gap")
    p.add_argument("--by", nargs="+", default=None,
                   choices=["era", "genre", "form", "source_type", "work_group"],
                   help="Break the score down by these ledger fields")
    p.add_argument("--contamination", type=int, default=0, metavar="N",
                   help="Generate N samples and report the longest exact token n-gram they "
                        "share with the training set (0 = skip)")
    p.add_argument("--json-out", type=Path, default=None)
    paths.add_path_args(p, include_checkpoint=True)
    args = p.parse_args()

    results = []
    if args.compare:
        for d in args.compare:
            results.append(evaluate_one(args, Path(d), label=str(d)))
    else:
        d = Path(args.data_dir)
        results.append(evaluate_one(args, d, label=args.label or str(d)))

    for r in results:
        print_result(r)

    if len(results) == 2:
        a, b = results
        gap = b["final_ce"] - a["final_ce"]
        print(f"\n=== Comparison ===")
        print(f"  {a['label']}: {a['final_ce']:.4f}")
        print(f"  {b['label']}: {b['final_ce']:.4f}")
        print(f"  gap: {gap:+.4f} nats  ({b['final_ppl']/a['final_ppl']:.2f}x perplexity)")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
