"""
Fast CPU tests for the invariants that the audit found broken.

Each test corresponds to a specific defect:
  * paths resolve from the module, not the cwd
  * meta.pkl stores portable (relative) tokenizer paths
  * documents are EOS-separated
  * no work group straddles the train/val split
  * evaluation is deterministic and does not perturb the training stream
  * the final-readout loss is reported separately from the loop-averaged objective
  * KV-cache generation is token-identical to full recomputation
  * generation stops at EOS

Run:  /usr/bin/python3 -m pytest src/tests/ -q
      /usr/bin/python3 src/tests/test_pipeline.py     (no pytest needed)
"""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

import paths  # noqa: E402
import prepare_corpus  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402


# --- Fixtures -------------------------------------------------------------------------

def _tiny_corpus(root: Path, n_works: int = 12, pages_per_work: int = 4) -> Path:
    """A miniature corpus that mimics the real naming conventions."""
    corpus = root / "Training Data"
    corpus.mkdir(parents=True)
    for w in range(n_works):
        for p in range(pages_per_work):
            # Alternate the two real conventions: scan pages and subdivided works.
            if w % 2 == 0:
                name = f"Pagina_Opus Numerus {w}.djvu_{p}.txt"
            else:
                name = f"Opus Alterum {w}_Liber {p}.txt"
            body = " ".join(f"uerbum{w}{p}{i} latinum est" for i in range(120))
            (corpus / name).write_text(body, encoding="utf-8")
    return corpus


def _build(root: Path):
    corpus = _tiny_corpus(root)
    out = root / "data"

    class A:
        corpus_dir = corpus
        out_dir = out
        tokenizer_dir = paths.TOKENIZER_DIR
        manifest = root / "missing_manifest.json"
        val_fraction = 0.2
        max_group_share = 1.0
        min_chars = 8
        seed = 1337
        dry_run = False
        # mixture options
        orthography = "none"
        max_fragment_score = 1.01
        weight_profile = "manifest"
        era_weight = None
        genre_weight = None
        canon_boost = None
        quality_weighting = None
        use_tier = None
        max_weight = None
        min_quality = 0.0
        budget_tokens = 2.458e9

    rc = prepare_corpus.build(A())
    assert rc == 0
    return out


# --- Tests ----------------------------------------------------------------------------

def test_work_id_groups_pages_and_subdivisions():
    """Pages of one book and chapters of one work must map to a single work id."""
    assert prepare_corpus.work_id_for("Pagina_De Bello Gallico.djvu_1.txt") == \
           prepare_corpus.work_id_for("Pagina_De Bello Gallico.djvu_207.txt")
    assert prepare_corpus.work_id_for("Summa Theologiae_Prima pars_Quaestio IX.txt") == \
           prepare_corpus.work_id_for("Summa Theologiae_Tertia pars_Quaestio CXIV.txt")
    # Volumes of a known series collapse together.
    assert prepare_corpus.work_id_for("Patrologia Latina_84.txt") == \
           prepare_corpus.work_id_for("Pagina_Patrologia Latina 139.djvu_89.txt")
    # The same work named with and without its author is ONE work, not two. Different
    # sources name works differently, and treating them as distinct would put one copy in
    # train and the other in val.
    assert prepare_corpus.work_id_for("Aeneis_I.txt") == \
           prepare_corpus.work_id_for("Aeneis (Vergilius)_I.txt")


def test_author_disambiguates_only_genuinely_ambiguous_titles():
    """Data-driven: keep the author in the group key exactly when the corpus holds that
    title under more than one author."""
    def doc(name):
        return {"doc_id": name,
                "work_id": prepare_corpus.work_id_for(name),
                "author_hint": prepare_corpus.author_hint(name)}

    docs = [doc("Carmina (Horatius)_Liber I.txt"),
            doc("Carmina (Venantius Fortunatus)_II.txt"),
            doc("Aeneis (Vergilius)_I.txt"),
            doc("Aeneis_II.txt")]
    prepare_corpus.assign_work_groups(docs)
    groups = {d["doc_id"]: d["work_group"] for d in docs}

    # "Carmina" exists under two authors -> stays split.
    assert groups["Carmina (Horatius)_Liber I.txt"] != \
           groups["Carmina (Venantius Fortunatus)_II.txt"]
    # "Aeneis" has one author -> the two namings collapse into one group.
    assert groups["Aeneis (Vergilius)_I.txt"] == groups["Aeneis_II.txt"]


def test_cleaner_preserves_latin_orthography():
    """u/v, i/j and macrons carry period information and must survive cleaning."""
    src = "uenit iulius cæsar ā rōmā .b et sīc​ fīnis"
    out = prepare_corpus.clean_text(src)["text"]
    assert "uenit" in out and "iulius" in out          # no u/v or i/j flattening
    assert "ā" in out and "rōmā" in out                # macrons kept
    assert "​" not in out                          # zero-width stripped
    assert " .b " not in out and not out.endswith(" .b")  # editorial marker stripped
    # Genuine abbreviations must NOT be damaged.
    assert prepare_corpus.clean_text("anno a.u.c. dcc")["text"] == "anno a.u.c. dcc"


def test_documents_are_eos_separated():
    with tempfile.TemporaryDirectory() as td:
        out = _build(Path(td))
        meta = pickle.load(open(out / paths.META_NAME, "rb"))
        eos = meta["special_tokens"]["eos"]
        assert meta["eos_separated"] is True

        for split in ("train", "val"):
            data = np.memmap(out / f"{split}.bin", dtype=np.uint16, mode="r")
            n_eos = int((np.asarray(data) == eos).sum())
            n_docs = meta["data_stats"][f"{split}_docs"]
            assert n_eos == n_docs, f"{split}: {n_eos} EOS for {n_docs} docs"


def test_no_work_group_straddles_the_split():
    """The defect that made val loss meaningless: pages of one book on both sides."""
    import json
    with tempfile.TemporaryDirectory() as td:
        out = _build(Path(td))
        train_groups, val_groups = set(), set()
        with open(out / paths.LEDGER_NAME, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                (val_groups if row["split"] == "val" else train_groups).add(row["work_group"])
        assert train_groups & val_groups == set(), \
            f"work groups on both sides: {train_groups & val_groups}"
        assert val_groups, "validation split is empty"


def test_meta_tokenizer_paths_are_relative():
    """Absolute paths in meta.pkl made the artifact unusable on any other machine."""
    with tempfile.TemporaryDirectory() as td:
        out = _build(Path(td))
        meta = pickle.load(open(out / paths.META_NAME, "rb"))
        for key in ("vocab_file", "merges_file"):
            val = meta["tokenizer_config"][key]
            assert not os.path.isabs(val), f"{key} is absolute: {val}"
        # ...and they still resolve.
        v, m = paths.tokenizer_files(meta, out / paths.META_NAME)
        assert v.exists() and m.exists()


def test_editors_are_not_mistaken_for_authors():
    """'ed. migne' is 24MB of the corpus and is an editor, not someone writing Latin."""
    import classify
    for editor in ("De civitate Dei (ed. Migne)", "Sermones (Migne)",
                   "Breviarium (PL 086)", "Codex Amiatinus (1854)"):
        assert classify.author_from_title(editor) is None, editor
    # Real authors still come through.
    assert classify.author_from_title("Carmina (Horatius)_Liber I.txt") == "horatius"


def test_era_evidence_precedence():
    """Author beats work title beats PL volume number."""
    import classify
    # Work title, author absent from filename.
    assert classify.era_for(None, "De civitate Dei (ed. Migne)") == classify.LATE_ANTIQUE
    assert classify.era_for(None, "Summa Theologiae_Prima pars") == classify.HIGH_MEDIEVAL
    assert classify.era_for(None, "Ab Urbe Condita_liber I") == classify.CLASSICAL
    # Named author wins over anything in the title.
    assert classify.era_for("erasmus", "De civitate Dei") == classify.NEO_LATIN
    # Patrologia Latina volume number dates the volume.
    assert classify.era_from_pl_volume("pagina_patrologia latina 20.djvu_5") == \
           classify.LATE_ANTIQUE
    assert classify.era_from_pl_volume("pagina_patrologia latina 210.djvu_5") == \
           classify.HIGH_MEDIEVAL
    # Multi-era collections are labelled as such, never forced into one era.
    assert classify.era_for(None, "Gallia Christiana, 1720, T2") == classify.COMPILATION
    # Genuinely unknown stays unknown rather than being guessed.
    assert classify.era_for(None, "Quod paucis") == classify.UNKNOWN


def test_verse_detection_and_scan_page_abstention():
    """Scan pages keep printed line breaks, which mimic verse -- so abstain there."""
    import classify
    verse = "\n".join(["arma uirumque cano troiae qui primus ab oris"] * 20)
    prose = "\n".join([" ".join(["uerbum latinum est et scriptum"] * 14)] * 12)
    assert classify.verse_signal(verse, is_scan_page=False) == "verse"
    assert classify.verse_signal(prose, is_scan_page=False) == "prose"
    assert classify.verse_signal(verse, is_scan_page=True) == classify.UNKNOWN


def test_non_latin_and_ocr_signals():
    import classify
    latin = "gallia est omnis diuisa in partes tres quarum unam incolunt belgae"
    english = "This page is part of the collection and should not be here at all"
    assert classify.non_latin_ratio(english) > classify.non_latin_ratio(latin)
    assert classify.non_latin_ratio(latin) == 0.0
    assert classify.ocr_quality(latin) > classify.ocr_quality("g@ll1a €st 0mn|s d1u1sa")


def test_orthography_levels():
    """Standardization is destructive, so each level must do exactly what it claims."""
    import classify
    t = "Uēnī uīdī uīcī. Iulius veni vidi vici."

    assert classify.standardize_orthography(t, "none") == t

    # conservative: macrons go, u/v and i/j distinctions stay. Both spellings from the
    # input must survive side by side.
    cons = classify.standardize_orthography(t, "conservative")
    assert "ē" not in cons and "ī" not in cons
    assert "uidi" in cons and "vidi" in cons, \
        f"conservative must NOT fold u/v, got {cons!r}"

    # classical: everything folds toward u/i.
    clas = classify.standardize_orthography(t, "classical")
    assert "v" not in clas and "j" not in clas
    assert "ueni uidi uici" in clas

    # modern: folds the other way, including capitals.
    mod = classify.standardize_orthography(t, "modern")
    assert "Veni vidi vici" in mod and "Julius" in mod

    with pytest_raises(ValueError):
        classify.standardize_orthography(t, "nonsense")


class pytest_raises:
    """Minimal context manager so the suite runs with or without pytest installed."""

    def __init__(self, exc):
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, t, v, tb):
        if t is None:
            raise AssertionError(f"expected {self.exc.__name__}")
        return issubclass(t, self.exc)


def test_fragment_score_separates_prose_from_stubs():
    import classify
    prose = " ".join(["gallia est omnis diuisa in partes tres."] * 40)
    assert classify.fragment_score(prose) == 0.0
    assert classify.fragment_score("incipit liber primus.") == 1.0
    assert classify.fragment_score("") == 1.0
    # Lacunose excerpts score above clean prose.
    lacunose = " ".join(["uerbum ... *** aliud"] * 60)
    assert classify.fragment_score(lacunose) > classify.fragment_score(prose)


def test_mixture_profiles_shift_emphasis_without_touching_disk():
    """The point of sampler-side weighting: change the mixture, not the binary."""
    import weights as weightlib
    docs = [
        {"era": "classical", "genre": "poetry", "work_id": "aeneis", "tier": 5,
         "ocr_quality": 1.0, "non_latin_per_1k": 0.0},
        {"era": "early_medieval", "genre": "liturgy", "work_id": "missale x", "tier": 5,
         "ocr_quality": 1.0, "non_latin_per_1k": 0.0},
    ]
    uniform = weightlib.resolve_profile("uniform")
    canonical = weightlib.resolve_profile("canonical")

    wu = [weightlib.weight_for(d, uniform) for d in docs]
    wc = [weightlib.weight_for(d, canonical) for d in docs]
    assert wu[0] == wu[1], "uniform must not discriminate"
    # Canonical must favour the canonical classical poem over background liturgy.
    assert wc[0] / wc[1] > 20, (wc, "canonical profile should strongly favour the canon")


def test_weight_cap_prevents_compounding():
    """era x genre x canon compounds; without a ceiling it reaches hundreds-fold."""
    import weights as weightlib
    doc = {"era": "classical", "genre": "poetry", "work_id": "aeneis", "tier": 15,
           "ocr_quality": 1.0, "non_latin_per_1k": 0.0}
    capped = weightlib.resolve_profile("canonical")
    assert weightlib.weight_for(doc, capped) <= capped["max_weight"]
    uncapped = weightlib.resolve_profile("canonical", max_weight=1e9, use_tier=True)
    assert weightlib.weight_for(doc, uncapped) > 100, "test premise: factors do compound"


def test_non_latin_and_low_quality_documents_are_excluded():
    import weights as weightlib
    p = weightlib.resolve_profile("manifest")
    english = {"era": "unknown", "genre": "unknown", "work_id": "x", "tier": 2,
               "ocr_quality": 1.0, "non_latin_per_1k": 120.0}
    garbled = {"era": "unknown", "genre": "unknown", "work_id": "x", "tier": 2,
               "ocr_quality": 0.4, "non_latin_per_1k": 0.0}
    assert weightlib.weight_for(english, p) == 0.0
    assert weightlib.weight_for(garbled, p, min_quality=0.8) == 0.0


def test_summarize_warns_about_over_repetition():
    """A small stratum weighted heavily and trained long WILL be memorized; say so."""
    import weights as weightlib
    docs = [{"era": "classical", "genre": "poetry"}, {"era": "unknown", "genre": "unknown"}]
    # 1M tokens of classical weighted 40x against 100M of background.
    out = weightlib.summarize(docs, [40.0, 1.0], [1_000_000, 100_000_000],
                              budget_tokens=2.458e9)
    assert "WARNING" in out and "classical" in out
    quiet = weightlib.summarize(docs, [1.0, 1.0], [1_000_000, 100_000_000],
                                budget_tokens=1e8)
    assert "WARNING" not in quiet


def _tiny_model(n_loops=2):
    cfg = GPTConfig(block_size=32, vocab_size=128, n_layer=2, n_head=4, n_kv_head=2,
                    n_embd=32, dropout=0.0, n_loops=n_loops)
    m = GPT(cfg)
    m.eval()
    return m


def test_forward_reports_final_readout_separately():
    """exp(reported loss) was not perplexity because readouts were averaged.

    At random init every readout has essentially the same loss (~log vocab), so the two
    numbers coincide by accident. Train briefly so the readouts actually separate, then
    check that the objective and the final-readout metric are genuinely distinct.
    """
    torch.manual_seed(0)
    m = _tiny_model(n_loops=3)
    m.train()
    x = torch.randint(0, 128, (4, 16))
    y = torch.randint(0, 128, (4, 16))

    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    for _ in range(30):
        _, loss, _ = m(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    m.eval()
    _, loss, aux = m(x, y)
    steps = torch.stack(aux["step_losses"])
    assert "final_loss" in aux and len(aux["step_losses"]) == 3

    # The final readout is the last step, and the objective is the weighted blend of all
    # three -- so the two are different quantities by construction.
    assert torch.isclose(steps[-1], aux["final_loss"])
    w = torch.arange(1, 4, dtype=steps.dtype)
    assert torch.isclose(loss, (steps * w).sum() / w.sum(), atol=1e-6)

    # Readouts must actually carry different information; if they were identical, deep
    # supervision would be doing nothing and the distinction would be moot.
    assert steps.std() > 0, "loop readouts are bitwise identical"
    assert float(loss) != float(aux["final_loss"]), \
        "objective and final-readout metric collapsed to the same number"


def test_linear_weighting_favours_later_readouts():
    m = _tiny_model(n_loops=3)
    x = torch.randint(0, 128, (2, 16))
    y = torch.randint(0, 128, (2, 16))
    m.config.loop_loss_weighting = "uniform"
    _, uniform_loss, aux = m(x, y)
    m.config.loop_loss_weighting = "linear"
    _, linear_loss, _ = m(x, y)
    steps = torch.stack(aux["step_losses"])
    w = torch.arange(1, 4, dtype=steps.dtype)
    assert torch.isclose(linear_loss, (steps * w).sum() / w.sum(), atol=1e-5)
    assert torch.isclose(uniform_loss, steps.mean(), atol=1e-5)


def test_kv_cache_matches_full_recomputation():
    for n_loops in (1, 3):
        m = _tiny_model(n_loops=n_loops)
        idx = torch.randint(0, 128, (2, 8))
        a = m.generate(idx.clone(), 16, top_k=1, use_cache=False)
        b = m.generate(idx.clone(), 16, top_k=1, use_cache=True)
        assert torch.equal(a, b), f"cache mismatch at n_loops={n_loops}"


def test_generation_stops_at_eos():
    m = _tiny_model(n_loops=1)
    idx = torch.randint(0, 128, (1, 8))
    greedy = m.generate(idx.clone(), 12, top_k=1)
    first = int(greedy[0, 8])
    stopped = m.generate(idx.clone(), 12, top_k=1, eos_token_id=first)
    assert stopped.size(1) == 9, f"expected stop after 1 token, got {stopped.size(1) - 8}"


def test_eval_is_deterministic_and_does_not_perturb_training_stream():
    """Two defects at once: eval windows were random every time, AND drawing them from the
    global torch RNG shifted the training batches that followed."""
    import train_latin

    with tempfile.TemporaryDirectory() as td:
        out = _build(Path(td))
        train_latin._DATA_CACHE.clear()
        train_latin._EVAL_WINDOWS.clear()
        train_latin._SAMPLER_CACHE.clear()

        config = {
            "data_dir": str(out), "device": "cpu", "block_size": 16,
            "batch_size": 2, "eval_iters": 3, "eval_seed": 1337, "n_loops": 1,
            "sampling": "weighted",
        }
        # The fixture is tokenized with the real 16k tokenizer, so the model must span it.
        meta = pickle.load(open(out / paths.META_NAME, "rb"))
        cfg = GPTConfig(block_size=16, vocab_size=meta["vocab_size"], n_layer=1, n_head=2,
                        n_kv_head=1, n_embd=16, dropout=0.0, n_loops=1)
        m = GPT(cfg)
        m.eval()
        from contextlib import nullcontext

        a = train_latin.estimate_loss(m, config, nullcontext())
        b = train_latin.estimate_loss(m, config, nullcontext())
        for k in a:
            assert abs(float(a[k]) - float(b[k])) < 1e-9, f"eval not deterministic on {k}"
        assert "val_final" in a and "train_final" in a

        # Training batches must be unaffected by whether an eval happened in between.
        torch.manual_seed(0)
        x1, _ = train_latin.get_batch("train", config)
        torch.manual_seed(0)
        train_latin.estimate_loss(m, config, nullcontext())
        x2, _ = train_latin.get_batch("train", config)
        assert torch.equal(x1, x2), "evaluating changed the training data stream"


def test_missing_validation_raises_instead_of_scoring_train():
    """Silently reporting training loss as validation loss invalidates a whole run."""
    import train_latin
    with tempfile.TemporaryDirectory() as td:
        out = _build(Path(td))
        os.remove(out / "val.bin")
        train_latin._DATA_CACHE.clear()
        config = {"data_dir": str(out), "device": "cpu", "block_size": 16,
                  "batch_size": 2, "eval_iters": 1, "eval_seed": 1337, "n_loops": 1,
                  "sampling": "weighted"}
        try:
            train_latin._get_split_data("val", config)
        except FileNotFoundError:
            return
        raise AssertionError("missing val.bin did not raise")


def test_weighted_sampling_respects_tiers():
    """Tier weighting used to be physical duplication; it now happens at sample time."""
    import train_latin
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "data"
        out.mkdir(parents=True)
        # Two documents of equal length, weights 1 and 9.
        np.save(out / "train_index.npy", np.array([[0, 1000], [1000, 1000]], dtype=np.int64))
        np.save(out / "train_weights.npy", np.array([1.0, 9.0], dtype=np.float32))
        train_latin._SAMPLER_CACHE.clear()
        s = train_latin._get_doc_sampler({"data_dir": str(out), "sampling": "weighted"})
        # cdf[0] is the probability mass of the first document: 1/(1+9).
        assert abs(s["cdf"][0] - 0.1) < 1e-6, s["cdf"]

        train_latin._SAMPLER_CACHE.clear()
        s = train_latin._get_doc_sampler({"data_dir": str(out), "sampling": "uniform"})
        assert abs(s["cdf"][0] - 0.5) < 1e-6, s["cdf"]


def test_scripts_resolve_paths_independent_of_cwd():
    """The original defect: the same command behaved differently from src/ vs repo root."""
    code = ("import sys; sys.path.insert(0, %r); import paths; "
            "print(paths.DATA_DIR, paths.TOKENIZER_DIR, paths.OUT_DIR)" % str(SRC))
    outs = []
    for cwd in (SRC, SRC.parent, Path(tempfile.gettempdir())):
        r = subprocess.run([sys.executable, "-c", code], cwd=str(cwd),
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout.strip())
    assert len(set(outs)) == 1, f"paths depend on cwd: {outs}"


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
