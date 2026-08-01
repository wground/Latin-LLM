"""
Build the training dataset with a work-level split, document separators, and a ledger.

Supersedes prepare_latin_weighted.py. Three things changed, each of which invalidated a
result the old pipeline produced:

1. THE SPLIT IS BY WORK, NOT BY FILE.
   Nearly half the corpus is single scanned pages named
   ``Pagina_<Work>.djvu_<n>.txt``, and most of the rest is one chapter/book/question per
   file. A random file-level split therefore scattered pages of the *same book* across
   train and val: 55.7% of validation files belonged to a work that was also in training.
   Validation was measuring interpolation to familiar text, not generalization. Documents
   are now grouped into works (and into exact-duplicate clusters) and whole groups are
   assigned to one side.

2. DOCUMENTS ARE SEPARATED BY <|endoftext|>.
   The previous train.bin contained exactly zero EOS tokens, so the model had no way to
   learn where a document ends -- which is why generation never stops.

3. WEIGHTING IS SAMPLING, NOT DUPLICATION.
   Tier multipliers used to be applied by writing a document's bytes 2/5/15 times into
   train.bin. Now each document is written once and its weight is recorded in the ledger;
   the trainer samples accordingly. This makes the corpus mixture a run-time choice
   (and an ablatable one) rather than something baked into a 1.3 GB binary.

Outputs, all under --out-dir (default src/gpt_data_latin/):
    train.bin, val.bin      uint16 token streams
    train_index.npy         (n_docs, 2) int64 [start_offset, length] into train.bin
    train_weights.npy       (n_docs,) float32 sampling weight per document
    corpus_ledger.jsonl     one row per document: ids, work, split, hashes, stats
    meta.pkl                vocab, tokenizer config (relative paths), data stats
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import pickle
import random
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tokenizers import ByteLevelBPETokenizer

import classify
import paths
import weights as weightlib

DUP_MARKER_RE = re.compile(r"\.__dup\d+__\.txt$")
# Wikisource scan pages: "Pagina_<Work>.djvu_<page>.txt"
PAGINA_RE = re.compile(r"^Pagina_(?P<work>.+?)\.djvu_\d+$")
# Volume numbers inside a series name, e.g. "Patrologia Latina 139" -> "Patrologia Latina".
SERIES_VOLUME_RE = re.compile(r"^(?P<series>.+?)[ ,]+(?:t\.?|tomus|vol\.?|band)?\s*\d+$", re.I)
KNOWN_SERIES = ("patrologia latina", "gallia christiana", "acta sanctorum",
                "monumenta germaniae historica")


# --- Cleaning -------------------------------------------------------------------------

ZERO_WIDTH_RE = re.compile(r"[​‌‍⁠﻿]")
REPLACEMENT_RE = re.compile(r"�")
# Standalone editorial/apparatus letters: " .a ", " .b. ". Deliberately anchored on
# whitespace so genuine Latin abbreviations (a.u.c., s.r.e.) are left alone.
EDITORIAL_LETTER_RE = re.compile(r"(?<=\s)\.[a-z]\.?(?=\s|$)")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> Dict[str, object]:
    """Clean one document. Returns the text plus flags describing what was found.

    Deliberately conservative: u/v, i/j, macrons and medieval spellings are NOT normalized.
    Those distinctions are signal about period and edition, and flattening them would
    destroy the thing the model is supposed to learn.
    """
    flags = {
        "had_replacement_char": bool(REPLACEMENT_RE.search(text)),
        "had_zero_width": bool(ZERO_WIDTH_RE.search(text)),
        "had_editorial_codes": bool(EDITORIAL_LETTER_RE.search(text)),
    }

    text = unicodedata.normalize("NFC", text)
    text = CONTROL_RE.sub("", text)
    text = ZERO_WIDTH_RE.sub("", text)
    text = REPLACEMENT_RE.sub("", text)
    text = EDITORIAL_LETTER_RE.sub("", text)

    # OCR / layout residue (carried over from the previous pipeline)
    text = re.sub(r"_{3,}", "", text)
    text = re.sub(r"-{5,}", "", text)
    text = re.sub(r"[{}\[\]]{2,}", "", text)
    text = re.sub(r" _ ", " ", text)
    text = re.sub(r"^[\s\.\-_\*=]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]{3,}", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    flags["text"] = text.strip()
    return flags


# --- Work identity --------------------------------------------------------------------

def work_id_for(filename: str) -> str:
    """Map a filename to the work it belongs to.

    The corpus follows a consistent convention:
        ``Pagina_<Work>.djvu_<n>.txt``      one scanned page of <Work>
        ``<Work>_<part>[_<subpart>].txt``   one subdivision of <Work>
        ``<Work>.txt``                      a whole short work
    Author parentheses are kept, so ``Carmina (Horatius)`` and
    ``Carmina (Venantius Fortunatus)`` stay distinct works.
    """
    stem = filename[:-4] if filename.endswith(".txt") else filename

    m = PAGINA_RE.match(stem)
    if m:
        work = m.group("work")
    else:
        # Everything before the first underscore is the work; the rest is subdivision.
        work = stem.split("_", 1)[0]

    work = work.strip().lower()

    # Collapse volume numbers for known multi-volume series, so that volume 1 of a series
    # in train and volume 2 in val does not count as an honest held-out work.
    vm = SERIES_VOLUME_RE.match(work)
    if vm:
        series = vm.group("series").strip()
        if any(series.startswith(k) for k in KNOWN_SERIES):
            work = series

    return work or stem.lower()


# --- Main -----------------------------------------------------------------------------

def load_multipliers(manifest_path: Path) -> Dict[str, int]:
    if not manifest_path.exists():
        print(f"  (no manifest at {manifest_path}; every document weighted 1x)")
        return {}
    with open(manifest_path, encoding="utf-8") as fh:
        data = json.load(fh)
    return {e["original"]: int(e["tier"]) for e in data.get("files", [])}


def report_classification(docs: List[Dict]) -> None:
    """Print label coverage by BYTES, and flag documents that look non-Latin or badly OCR'd.

    Coverage is reported honestly: "unknown" is a category, not a gap to be papered over.
    Per-stratum evaluation is only meaningful on the labelled portion.
    """
    total = sum(d["bytes"] for d in docs) or 1

    def dist(field):
        agg: Dict[str, int] = defaultdict(int)
        for d in docs:
            agg[str(d.get(field, "unknown"))] += d["bytes"]
        return sorted(agg.items(), key=lambda kv: -kv[1])

    for field in ("era", "genre", "form", "source_type"):
        rows = dist(field)
        known = sum(b for k, b in rows if k != "unknown")
        print(f"\n  {field} coverage: {100*known/total:.1f}% of bytes labelled")
        for k, b in rows[:8]:
            print(f"      {100*b/total:5.1f}%  {k}")

    suspect = [d for d in docs if d.get("non_latin_per_1k", 0) > 15]
    poor = [d for d in docs if d.get("ocr_quality", 1.0) < 0.90]
    print(f"\n  quality flags: {len(suspect):,} docs look non-Latin "
          f"({100*sum(d['bytes'] for d in suspect)/total:.1f}% of bytes), "
          f"{len(poor):,} docs score <0.90 on OCR quality "
          f"({100*sum(d['bytes'] for d in poor)/total:.1f}% of bytes)")


def apply_weights(docs: List[Dict], args, out_dir: Path) -> Dict:
    """Compute per-document sampling weights and write train_weights.npy.

    Separated from encoding on purpose: the mixture can be changed without touching
    train.bin, which is what makes `--reweight` a seconds-long operation.
    """
    profile = weightlib.resolve_profile(
        args.weight_profile,
        era_overrides=weightlib.parse_kv(args.era_weight),
        genre_overrides=weightlib.parse_kv(args.genre_weight),
        canon_boost=args.canon_boost,
        use_quality=(None if args.quality_weighting is None else args.quality_weighting),
        use_tier=(None if args.use_tier is None else args.use_tier),
        max_weight=args.max_weight,
    )

    train_docs = [d for d in docs if d["split"] == "train"]
    w = [weightlib.weight_for(d, profile, min_quality=args.min_quality) for d in train_docs]
    for d, wi in zip(train_docs, w):
        d["weight"] = wi

    np.save(out_dir / "train_weights.npy", np.array(w, dtype=np.float32))
    (out_dir / "mixture.json").write_text(json.dumps(profile, indent=2))

    print(f"\nMixture profile: {profile['name']}")
    print(weightlib.summarize(train_docs, w, [d.get("tokens", 0) for d in train_docs],
                              budget_tokens=args.budget_tokens))
    return profile


def reweight(args) -> int:
    """Recompute sampling weights from an existing ledger, without re-encoding anything."""
    out_dir = Path(args.out_dir)
    ledger_path = out_dir / paths.LEDGER_NAME
    index_path = out_dir / "train_index.npy"
    if not (ledger_path.exists() and index_path.exists()):
        print(f"Need an existing build in {out_dir} (run prepare_corpus.py first).",
              file=sys.stderr)
        return 1

    docs = [json.loads(line) for line in open(ledger_path, encoding="utf-8")]
    train_docs = [d for d in docs if d["split"] == "train"]
    index = np.load(index_path)
    if len(index) != len(train_docs):
        print(f"Ledger/index mismatch ({len(train_docs)} vs {len(index)}); rebuild.",
              file=sys.stderr)
        return 1
    for d, (_, n) in zip(train_docs, index):
        d["tokens"] = int(n)

    apply_weights(docs, args, out_dir)

    # Keep the ledger's weight column in step with what the sampler will use.
    with open(ledger_path, "w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"\nUpdated {out_dir}/train_weights.npy and {paths.LEDGER_NAME}. "
          f"train.bin untouched.")
    return 0


def build(args) -> int:
    corpus_dir = Path(args.corpus_dir)
    out_dir = Path(args.out_dir)
    tokenizer_dir = Path(args.tokenizer_dir)

    print("=" * 72)
    print("    Latin corpus build — work-level split, EOS-separated, ledgered")
    print("=" * 72)

    tok = ByteLevelBPETokenizer(
        str(tokenizer_dir / paths.VOCAB_NAME),
        str(tokenizer_dir / paths.MERGES_NAME),
    )
    vocab_size = tok.get_vocab_size()
    vocab = tok.get_vocab()
    eos_id = vocab.get("<|endoftext|>", 0)
    pad_id = vocab.get("<|pad|>", 1)
    print(f"Tokenizer: {vocab_size} tokens from {tokenizer_dir}")
    print(f"Special tokens: eos={eos_id}, pad={pad_id}")

    all_txt = glob.glob(os.path.join(str(corpus_dir), "**", "*.txt"), recursive=True)
    originals = sorted(f for f in all_txt if not DUP_MARKER_RE.search(os.path.basename(f)))
    n_physical_dups = len(all_txt) - len(originals)
    if not originals:
        print(f"No original .txt files under {corpus_dir}", file=sys.stderr)
        return 1
    print(f"Found {len(originals):,} unique source files "
          f"({n_physical_dups:,} physical .__dup__ copies ignored)")

    multipliers = load_multipliers(Path(args.manifest))

    # --- Pass 1: read, clean, hash, assign work ids ---
    print("Reading and cleaning...")
    docs: List[Dict] = []
    by_hash: Dict[str, List[int]] = defaultdict(list)
    for path in originals:
        basename = os.path.basename(path)
        try:
            raw = open(path, "r", encoding="utf-8", errors="replace").read()
        except OSError as e:
            print(f"  skipping {basename}: {e}")
            continue
        cleaned = clean_text(raw)
        text = cleaned.pop("text")
        if len(text) < args.min_chars:
            continue

        rel = os.path.relpath(path, corpus_dir)
        content_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()
        is_scan = basename.startswith("Pagina_")
        title = basename[:-4] if basename.endswith(".txt") else basename
        doc = {
            "doc_id": len(docs),
            "path": rel,
            "work_id": work_id_for(basename),
            "is_scan_page": is_scan,
            "chars": len(text),
            "bytes": len(text.encode("utf-8")),
            "content_sha1": content_hash,
            "tier": multipliers.get(rel, 1),
            **cleaned,
            **classify.classify(text, title, is_scan),
        }
        docs.append(doc)
        by_hash[content_hash].append(doc["doc_id"])
        docs[-1]["_text"] = text

    print(f"  {len(docs):,} documents kept ({len(originals) - len(docs):,} dropped as too short)")
    report_classification(docs)

    # --- Grouping: union work families with exact-duplicate clusters ---
    # A group is the unit that must not straddle the split. Exact duplicates are merged
    # into the same group even when their filenames suggest different works, so identical
    # text can never appear on both sides.
    group_of: Dict[int, str] = {d["doc_id"]: d["work_id"] for d in docs}
    merged = 0
    for h, ids in by_hash.items():
        if len(ids) > 1:
            target = group_of[ids[0]]
            for i in ids[1:]:
                if group_of[i] != target:
                    old = group_of[i]
                    for d in docs:
                        if group_of[d["doc_id"]] == old:
                            group_of[d["doc_id"]] = target
                    merged += 1
    if merged:
        print(f"  merged {merged} work group(s) via exact-duplicate clusters")

    groups: Dict[str, List[int]] = defaultdict(list)
    for d in docs:
        groups[group_of[d["doc_id"]]].append(d["doc_id"])
    print(f"  {len(groups):,} distinct work groups")

    # --- Split by group, targeting a byte fraction ---
    # Group sizes are wildly uneven (a 900-page scanned series vs a one-page hymn), so we
    # fill validation by BYTES rather than by group count.
    doc_by_id = {d["doc_id"]: d for d in docs}
    group_bytes = {g: sum(doc_by_id[i]["bytes"] for i in ids) for g, ids in groups.items()}
    total_bytes = sum(group_bytes.values())
    target_val_bytes = total_bytes * args.val_fraction

    rng = random.Random(args.seed)
    order = sorted(groups.keys())
    rng.shuffle(order)

    val_groups, val_bytes = set(), 0
    for g in order:
        if val_bytes >= target_val_bytes:
            break
        # Never let a single group swamp validation.
        if group_bytes[g] > target_val_bytes * args.max_group_share:
            continue
        val_groups.add(g)
        val_bytes += group_bytes[g]

    for d in docs:
        d["split"] = "val" if group_of[d["doc_id"]] in val_groups else "train"
        d["work_group"] = group_of[d["doc_id"]]

    n_val_docs = sum(1 for d in docs if d["split"] == "val")
    print(f"Split: {len(docs) - n_val_docs:,} train docs / {n_val_docs:,} val docs "
          f"({len(val_groups):,} of {len(groups):,} work groups held out)")
    print(f"       {(total_bytes - val_bytes)/1e6:.1f} MB train / {val_bytes/1e6:.1f} MB val "
          f"({100*val_bytes/total_bytes:.1f}%)")

    if args.dry_run:
        print("\n--dry-run: no files written.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Encode and write ---
    # Each document is written ONCE, terminated by EOS. Weighting happens at sample time.
    def write_split(split: str, filename: str):
        index = []
        offset = 0
        total_tokens = 0
        with open(out_dir / filename, "wb") as out:
            for d in docs:
                if d["split"] != split:
                    continue
                ids = tok.encode(d["_text"]).ids
                ids.append(eos_id)
                arr = np.array(ids, dtype=np.uint16)
                out.write(arr.tobytes())
                index.append((offset, len(ids)))
                d["token_offset"] = offset
                d["tokens"] = len(ids)
                offset += len(ids)
                total_tokens += len(ids)
        return np.array(index, dtype=np.int64), total_tokens

    print("Encoding train split...")
    train_index, train_tokens = write_split("train", paths.TRAIN_BIN)
    print("Encoding val split...")
    _, val_tokens = write_split("val", paths.VAL_BIN)

    np.save(out_dir / "train_index.npy", train_index)
    profile = apply_weights(docs, args, out_dir)

    # --- Ledger ---
    with open(out_dir / paths.LEDGER_NAME, "w", encoding="utf-8") as fh:
        for d in docs:
            row = {k: v for k, v in d.items() if not k.startswith("_")}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # --- Meta ---
    train_bytes = sum(d["bytes"] for d in docs if d["split"] == "train")
    val_bytes_final = sum(d["bytes"] for d in docs if d["split"] == "val")
    tier_counts: Dict[int, int] = defaultdict(int)
    for d in docs:
        if d["split"] == "train":
            tier_counts[d["tier"]] += 1

    meta = {
        "vocab_size": vocab_size,
        "tokenizer_config": paths.make_tokenizer_config(tokenizer_dir, out_dir),
        "special_tokens": {"eos": eos_id, "pad": pad_id},
        "eos_separated": True,
        "split_policy": "work-level (pages/books/chapters grouped by work; exact-duplicate "
                        "clusters merged); validation filled by byte target",
        "seed": args.seed,
        "data_stats": {
            "documents": len(docs),
            "work_groups": len(groups),
            "val_work_groups": len(val_groups),
            "train_docs": len(docs) - n_val_docs,
            "val_docs": n_val_docs,
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
            "train_bytes": train_bytes,
            "val_bytes": val_bytes_final,
            "tier_file_counts": dict(sorted(tier_counts.items())),
            "weighting": "sampling weights in train_weights.npy (train.bin is unweighted)",
        },
    }
    with open(out_dir / paths.META_NAME, "wb") as f:
        pickle.dump(meta, f)

    print("-" * 72)
    print(f"Train tokens : {train_tokens/1e6:,.1f}M  ({len(train_index):,} documents)")
    print(f"Val tokens   : {val_tokens/1e6:,.1f}M")
    print(f"Bytes/token  : {train_bytes/max(1,train_tokens):.3f}")
    print(f"Tier counts  : {dict(sorted(tier_counts.items()))}")
    print(f"Wrote {out_dir}/{{train.bin,val.bin,train_index.npy,train_weights.npy,"
          f"{paths.LEDGER_NAME},{paths.META_NAME}}}")
    print("-" * 72)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus-dir", type=Path, default=paths.CORPUS_DIR)
    p.add_argument("--out-dir", type=Path, default=paths.DATA_DIR)
    p.add_argument("--tokenizer-dir", type=Path, default=paths.TOKENIZER_DIR)
    p.add_argument("--manifest", type=Path, default=paths.MULTIPLIER_MANIFEST)
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--max-group-share", type=float, default=0.25,
                   help="Reject a work group from val if it alone exceeds this share of the "
                        "val byte target (keeps one huge series from dominating validation)")
    p.add_argument("--min-chars", type=int, default=32,
                   help="Drop documents shorter than this after cleaning")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--dry-run", action="store_true",
                   help="Report the split without writing any files")

    mix = p.add_argument_group(
        "corpus mixture",
        "Weights are applied by the sampler, so they never change train.bin's size and "
        "never require re-encoding. Use --reweight to change the mixture in seconds.")
    mix.add_argument("--weight-profile", default="manifest",
                     choices=sorted(weightlib.PROFILES),
                     help="uniform: all equal. manifest: the hand-made tier file (default, "
                          "historical behaviour). canonical: emphasise historically "
                          "important works, long tail as background. balanced: flatten the "
                          "corpus's medieval skew.")
    mix.add_argument("--era-weight", nargs="+", metavar="ERA=W",
                     help="e.g. --era-weight classical=8 late_antique=2")
    mix.add_argument("--genre-weight", nargs="+", metavar="GENRE=W",
                     help="e.g. --genre-weight poetry=4 liturgy=0.3")
    mix.add_argument("--canon-boost", type=float, default=None,
                     help="Multiplier for works in the canonical list (weights.py)")
    mix.add_argument("--quality-weighting", action="store_true", default=None,
                     help="Scale weight by OCR quality")
    mix.add_argument("--no-quality-weighting", dest="quality_weighting",
                     action="store_false")
    mix.add_argument("--use-tier", action="store_true", default=None,
                     help="Include the hand-made tier manifest as a factor")
    mix.add_argument("--no-use-tier", dest="use_tier", action="store_false")
    mix.add_argument("--max-weight", type=float, default=None,
                     help="Ceiling on any single document's weight. Prevents era x genre x "
                          "canon from compounding into hundreds-fold repetition.")
    mix.add_argument("--min-quality", type=float, default=0.0,
                     help="Exclude documents scoring below this on OCR quality (0-1)")
    mix.add_argument("--budget-tokens", type=float, default=2.458e9,
                     help="Planned training budget (iters x tokens/iter), used to report "
                          "how many epochs each stratum would get. Default 2.458e9 matches "
                          "the previous 100k-iteration run.")
    mix.add_argument("--reweight", action="store_true",
                     help="Recompute weights from an existing build; does not touch "
                          "train.bin. Takes seconds instead of minutes.")
    p.add_argument("--prune-duplicates", action="store_true",
                   help="DELETE the physical .__dupN__.txt copies and exit. They are no "
                        "longer read by anything (weighting is applied at sample time via "
                        "train_weights.npy), so this only reclaims disk. Irreversible.")
    args = p.parse_args()

    if args.prune_duplicates:
        return prune_duplicates(Path(args.corpus_dir))
    if args.reweight:
        return reweight(args)
    return build(args)


def prune_duplicates(corpus_dir: Path) -> int:
    """Delete the physical .__dupN__.txt copies left over from the old weighting scheme."""
    dups = [f for f in glob.glob(os.path.join(str(corpus_dir), "**", "*.txt"), recursive=True)
            if DUP_MARKER_RE.search(os.path.basename(f))]
    if not dups:
        print("No .__dup__ copies found.")
        return 0
    total = sum(os.path.getsize(f) for f in dups)
    print(f"Found {len(dups):,} duplicate copies occupying {total/1e9:.2f} GB.")
    reply = input("Delete them? This cannot be undone. [y/N] ").strip().lower()
    if reply != "y":
        print("Aborted; nothing deleted.")
        return 1
    for f in dups:
        os.remove(f)
    print(f"Deleted {len(dups):,} files, reclaimed {total/1e9:.2f} GB.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
