"""
Corpus mixture: how much of the model's training budget each document gets.

Weights are applied by the *sampler*, not by duplicating text on disk, so changing the
mixture is free — `train.bin` never changes size and never has to be re-encoded. What
changes is the effective weighted exposure: the number of tokens the model actually sees.

    weight(doc) = tier x era x genre x canon x quality

Each factor is independently overridable from the command line, so a mixture is a
reproducible, inspectable config rather than something baked into a binary.

The intent behind the default `canonical` profile: give the historically important works
enough repetition that the model learns their register properly, while keeping the long
tail present at low weight as background linguistic support rather than dropping it.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

import classify

# --- Canonical works -------------------------------------------------------------------
# Substring matched against the lowercased work_id. These are the works whose style is
# worth learning precisely, as opposed to material that is mainly useful as language
# exposure. Extend freely; the boost is a single tunable (--canon-boost).
CANONICAL_WORKS: List[str] = [
    # Classical prose
    "de bello gallico", "de bello civili", "ab urbe condita", "naturalis historia",
    "de officiis", "de oratore", "de finibus", "de natura deorum", "tusculanae",
    "epistulae", "de re publica", "de legibus", "de amicitia", "de senectute",
    "in catilinam", "pro milone", "philippicae", "brutus", "orator",
    "annales", "historiae", "germania", "agricola", "de institutione oratoria",
    "bellum catilinae", "bellum iugurthinum", "de architectura", "de lingua latina",
    "de re rustica", "strategemata", "factorum ac dictorum",
    # Classical verse
    "aeneis", "aeneid", "georgica", "bucolica", "eclogae", "metamorphoses",
    "ars amatoria", "fasti", "tristia", "heroides", "carmina", "odae", "epodi",
    "satirae", "saturae", "epistulae ex ponto", "de rerum natura", "pharsalia",
    "thebais", "silvae", "epigrammata", "elegiae", "punica", "argonautica",
    # Drama
    "amphitruo", "aulularia", "menaechmi", "miles gloriosus", "captivi", "mostellaria",
    "rudens", "pseudolus", "andria", "eunuchus", "adelphoe", "phormio", "hecyra",
    "heautontimorumenos", "medea", "phaedra", "thyestes", "oedipus",
    # Late antique / patristic landmarks
    "confessiones", "de civitate dei", "de trinitate", "de doctrina christiana",
    "vulgata", "biblia sacra", "de consolatione philosophiae", "etymologiarum",
    "regula", "moralia", "historia ecclesiastica", "adversus haereses",
    "divinae institutiones", "apologeticum",
    # Medieval landmarks
    "summa theologiae", "summa contra gentiles", "sic et non", "historia calamitatum",
    "cur deus homo", "proslogion", "monologion", "de divisione naturae",
    "historia regum britanniae", "gesta francorum", "chronica majora",
    "sententiae", "didascalicon", "policraticus", "summa logicae",
    "carmina burana", "dies irae", "stabat mater",
    # Renaissance / early modern
    "encomium moriae", "colloquia familiaria", "adagia", "utopia",
    "de revolutionibus orbium", "sidereus nuncius", "principia", "novum organum",
    "ethica", "meditationes de prima philosophia", "systema naturae",
]

_CANON_RE = re.compile("|".join(re.escape(w) for w in CANONICAL_WORKS))

# --- Default profiles ------------------------------------------------------------------
# A profile is a set of multiplicative factors. Values are deliberately modest: the
# factors compound, so 5x era times 4x genre times 6x canon is already 120x.

PROFILES: Dict[str, Dict] = {
    # Every document equally likely per token. The control condition.
    "uniform": {
        "use_tier": False, "era": {}, "genre": {}, "canon": 1.0, "quality": False,
    },
    # Reproduces the historical behaviour: the hand-made tier manifest only.
    "manifest": {
        "use_tier": True, "era": {}, "genre": {}, "canon": 1.0, "quality": False,
    },
    # Emphasise historically important material; keep the long tail as background.
    # NOTE: use_tier is off here. The hand-made tier manifest already encodes an
    # "importance" judgement, so multiplying it by era x genre x canon double-counts the
    # same intent and compounds into hundreds-fold repetition.
    "canonical": {
        "use_tier": False,
        "era": {
            classify.CLASSICAL: 6.0,
            classify.LATE_ANTIQUE: 2.0,
            classify.EARLY_MEDIEVAL: 1.0,
            classify.HIGH_MEDIEVAL: 1.5,
            classify.LATE_MEDIEVAL: 1.5,
            classify.NEO_LATIN: 2.0,
            classify.COMPILATION: 0.4,
            classify.UNKNOWN: 0.6,
        },
        "genre": {
            "poetry": 3.0,
            "rhetoric_dialogue": 2.5,
            "history_hagiography": 1.5,
            "philosophy_scholastic": 1.5,
            "science_technical": 1.2,
            "scripture": 1.0,
            "exegesis": 0.5,
            "sermons": 0.6,
            "letters": 0.8,
            "law_canon": 0.5,
            "liturgy": 0.4,
            "monastic_devotional": 0.8,
            classify.UNKNOWN: 0.8,
        },
        "canon": 5.0,
        "quality": True,
        # Hard ceiling on any single document's weight. Without it the factors compound
        # (era x genre x canon) into repetition rates that memorize rather than teach.
        # 10 keeps classical material dominant (~48% of sampled tokens) while holding
        # repetition to a level a ~1.5B-token run can absorb.
        "max_weight": 10.0,
    },
    # Flatten the corpus's heavy medieval skew without privileging any canon.
    "balanced": {
        "use_tier": False,
        "era": {
            classify.CLASSICAL: 8.0,
            classify.LATE_ANTIQUE: 2.0,
            classify.EARLY_MEDIEVAL: 0.7,
            classify.HIGH_MEDIEVAL: 2.0,
            classify.LATE_MEDIEVAL: 4.0,
            classify.NEO_LATIN: 5.0,
            classify.COMPILATION: 0.5,
            classify.UNKNOWN: 1.0,
        },
        "genre": {"exegesis": 0.5, "letters": 0.7, "liturgy": 0.5, "law_canon": 0.6},
        "canon": 1.5,
        "quality": True,
        "max_weight": 25.0,
    },
}


def is_canonical(work_id: str) -> bool:
    return bool(_CANON_RE.search(work_id.lower()))


def parse_kv(pairs: Optional[List[str]]) -> Dict[str, float]:
    """Parse ``--era-weight classical=8 poetry=3`` style arguments."""
    out: Dict[str, float] = {}
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"expected key=value, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = float(v)
    return out


def resolve_profile(name: str, era_overrides=None, genre_overrides=None,
                    canon_boost: Optional[float] = None,
                    use_quality: Optional[bool] = None,
                    use_tier: Optional[bool] = None,
                    max_weight: Optional[float] = None) -> Dict:
    """A named profile with per-key CLI overrides layered on top."""
    if name not in PROFILES:
        raise ValueError(f"unknown profile {name!r}; choose from {sorted(PROFILES)}")
    p = json.loads(json.dumps(PROFILES[name]))  # deep copy
    p["era"].update(era_overrides or {})
    p["genre"].update(genre_overrides or {})
    if canon_boost is not None:
        p["canon"] = canon_boost
    if use_quality is not None:
        p["quality"] = use_quality
    if use_tier is not None:
        p["use_tier"] = use_tier
    if max_weight is not None:
        p["max_weight"] = max_weight
    p["name"] = name
    return p


def weight_for(doc: Dict, profile: Dict, min_quality: float = 0.0) -> float:
    """Sampling weight for one document. 0 means excluded from training."""
    q = doc.get("ocr_quality", 1.0)
    if q < min_quality:
        return 0.0
    # Documents that read as another language are language noise, not Latin exposure.
    if doc.get("non_latin_per_1k", 0.0) > 40:
        return 0.0

    w = float(doc.get("tier", 1)) if profile.get("use_tier") else 1.0
    w *= profile["era"].get(str(doc.get("era", classify.UNKNOWN)), 1.0)
    w *= profile["genre"].get(str(doc.get("genre", classify.UNKNOWN)), 1.0)
    if profile.get("canon", 1.0) != 1.0 and is_canonical(str(doc.get("work_id", ""))):
        w *= profile["canon"]
    if profile.get("quality"):
        # Scale smoothly with OCR quality rather than applying a hard cliff.
        w *= max(0.1, min(1.0, (q - 0.7) / 0.3)) if q < 1.0 else 1.0

    cap = profile.get("max_weight")
    if cap:
        w = min(w, float(cap))
    return round(w, 4)


def summarize(docs: List[Dict], weights: List[float], token_counts: List[int],
              budget_tokens: float = 2.458e9) -> str:
    """Report what a mixture actually produces.

    ``budget_tokens`` is how many tokens the training run will present
    (iterations x tokens_per_iteration); the previous 100k-iteration run presented 2.458B.
    Epochs are reported against that budget, because how often the model revisits a text
    depends on the mixture AND the run length, not on the mixture alone.
    """
    mass = [w * t for w, t in zip(weights, token_counts)]
    total = sum(mass) or 1.0
    raw = sum(token_counts) or 1

    lines = [
        f"  physical train tokens : {raw/1e6:,.1f}M   (train.bin size depends only on this)",
        f"  weighted stream       : {total/1e6:,.1f}M   ({total/raw:.2f}x expansion vs raw)",
        f"  documents excluded    : {sum(1 for w in weights if w == 0):,}",
        f"  epochs below assume a {budget_tokens/1e9:.3f}B-token training budget",
    ]
    worst, worst_name = 0.0, ""
    for field in ("era", "genre"):
        sampled: Dict[str, float] = {}
        rawtok: Dict[str, float] = {}
        for d, m, t in zip(docs, mass, token_counts):
            k = str(d.get(field, classify.UNKNOWN))
            sampled[k] = sampled.get(k, 0.0) + m
            rawtok[k] = rawtok.get(k, 0.0) + t

        lines.append(f"\n  by {field}:{'':<17}sampled     raw  emphasis   epochs")
        for k, v in sorted(sampled.items(), key=lambda kv: -kv[1]):
            share = v / total
            raw_share = rawtok[k] / raw if rawtok.get(k) else 0.0
            emphasis = share / raw_share if raw_share else 0.0
            # Epochs over this stratum: tokens of it presented / its unique size.
            epochs = (budget_tokens * share / rawtok[k]) if rawtok.get(k) else 0.0
            if epochs > worst:
                worst, worst_name = epochs, f"{field}={k}"
            flag = "  <-- high" if epochs > 40 else ""
            lines.append(f"      {k:<22} {100*share:6.1f}%  {100*raw_share:5.1f}%  "
                         f"{emphasis:7.1f}x  {epochs:7.0f}{flag}")

    if worst > 40:
        lines.append(
            f"\n  WARNING: {worst_name} would be seen ~{worst:.0f} times at this budget.\n"
            f"  Repetition past roughly 40 epochs tends to memorize text rather than teach\n"
            f"  its register (arXiv:2605.12715, arXiv:2606.06888). Either lower\n"
            f"  --canon-boost / --max-weight, shorten the run, or accept it knowingly.")
    return "\n".join(lines)
