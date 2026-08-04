"""
Document classification for the corpus ledger.

Two kinds of signal, kept strictly separate so that nothing is ever presented as more
certain than it is:

  AUTOMATIC (full coverage, derived from the text itself)
      prose vs verse, OCR quality score, non-Latin contamination, source type.

  CURATED (partial coverage, from a hand-maintained author/work table)
      author, era, genre. Anything not confidently derivable is labelled "unknown"
      rather than guessed -- a wrong era label is worse than a missing one, because
      per-stratum evaluation would silently compare the wrong things.

A note on what the table revealed: this corpus is overwhelmingly patristic and medieval,
not classical. Rabanus Maurus (19 MB) and Jerome (17 MB) each outweigh Cicero (1.8 MB) by
an order of magnitude. Any claim about the model's "Latin" should be read in that light.
"""
from __future__ import annotations

import re
import statistics
import unicodedata
from typing import Dict, List, Optional, Tuple

# --- Eras -----------------------------------------------------------------------------

CLASSICAL = "classical"            # to ~200 AD
LATE_ANTIQUE = "late_antique"      # ~200-600
EARLY_MEDIEVAL = "early_medieval"  # ~600-1000
HIGH_MEDIEVAL = "high_medieval"    # ~1000-1300
LATE_MEDIEVAL = "late_medieval"    # ~1300-1500
NEO_LATIN = "neo_latin"            # 1500+
# Multi-era collections (Gallia Christiana, Denzinger, papal registers). Forcing one era
# on these would be a lie, and would quietly corrupt any per-era comparison.
COMPILATION = "compilation"
UNKNOWN = "unknown"

# Parenthetical tokens that name an editor, series or volume rather than an author.
# "ed. migne" is 24 MB of the corpus and is emphatically not a person writing Latin.
NON_AUTHOR_PAREN = re.compile(
    r"^(ed\.?\s*migne|migne|pl\s*\d+|patrologia.*|\d{3,4}|[ivxlcdm]+|"
    r"kauer-lindsay.*|ed\..*|.*\bed\b\.?|vol\.?\s*\d+|t\.?\s*\d+)$", re.I)

# Author (Latin name form) -> era. Curated; extend freely.
AUTHOR_ERA: Dict[str, str] = {
    # --- Classical ---
    "marcus tullius cicero": CLASSICAL, "cicero": CLASSICAL,
    "gaius iulius caesar": CLASSICAL, "caesar": CLASSICAL,
    "publius vergilius maro": CLASSICAL, "vergilius": CLASSICAL, "virgilius": CLASSICAL,
    "quintus horatius flaccus": CLASSICAL, "horatius": CLASSICAL,
    "publius ovidius naso": CLASSICAL, "ovidius": CLASSICAL,
    "titus livius": CLASSICAL, "livius": CLASSICAL,
    "cornelius tacitus": CLASSICAL, "tacitus": CLASSICAL,
    "gaius valerius catullus": CLASSICAL, "catullus": CLASSICAL,
    "plinius": CLASSICAL, "plinius maior": CLASSICAL, "plinius minor": CLASSICAL,
    "titus maccius plautus": CLASSICAL, "plautus": CLASSICAL,
    "publius terentius afer": CLASSICAL, "terentius": CLASSICAL,
    "quintilianus": CLASSICAL, "seneca": CLASSICAL, "lucretius": CLASSICAL,
    "sallustius": CLASSICAL, "iuvenalis": CLASSICAL, "martialis": CLASSICAL,
    "propertius": CLASSICAL, "tibullus": CLASSICAL, "persius": CLASSICAL,
    "lucanus": CLASSICAL, "statius": CLASSICAL, "suetonius": CLASSICAL,
    "apuleius": CLASSICAL, "aulus gellius": CLASSICAL, "varro": CLASSICAL,
    "columella": CLASSICAL, "vitruvius": CLASSICAL, "celsus": CLASSICAL,
    "phaedrus": CLASSICAL, "curtius rufus": CLASSICAL, "valerius maximus": CLASSICAL,
    "firmicus maternus": CLASSICAL,

    # --- Late antique / patristic ---
    "hieronymus": LATE_ANTIQUE, "augustinus": LATE_ANTIQUE, "ambrosius": LATE_ANTIQUE,
    "ambrosiaster": LATE_ANTIQUE, "tertullianus": LATE_ANTIQUE,
    "cyprianus": LATE_ANTIQUE, "lactantius": LATE_ANTIQUE, "rufinus": LATE_ANTIQUE,
    "leo i": LATE_ANTIQUE, "leo magnus": LATE_ANTIQUE,
    "petrus chrysologus": LATE_ANTIQUE, "maximus": LATE_ANTIQUE,
    "eusebius vercellensis": LATE_ANTIQUE, "hilarius": LATE_ANTIQUE,
    "prudentius": LATE_ANTIQUE, "boethius": LATE_ANTIQUE, "cassiodorus": LATE_ANTIQUE,
    "sidonius apollinaris": LATE_ANTIQUE, "ennodius": LATE_ANTIQUE,
    "orosius": LATE_ANTIQUE, "sulpicius severus": LATE_ANTIQUE,
    "ulfilas": LATE_ANTIQUE, "vincentius lerinensis": LATE_ANTIQUE,
    "iohannes cassianus": LATE_ANTIQUE, "salvianus": LATE_ANTIQUE,
    "arnobius": LATE_ANTIQUE, "venantius fortunatus": LATE_ANTIQUE,
    "gregorius magnus": LATE_ANTIQUE, "isidorus": LATE_ANTIQUE,
    "isidorus hispalensis": LATE_ANTIQUE,

    # --- Early medieval (Carolingian core of this corpus) ---
    "rabanus maurus": EARLY_MEDIEVAL, "hrabanus maurus": EARLY_MEDIEVAL,
    "beda": EARLY_MEDIEVAL, "beda venerabilis": EARLY_MEDIEVAL,
    "alcuinus": EARLY_MEDIEVAL, "hincmarus rhemensis": EARLY_MEDIEVAL,
    "haymo halberstatensis": EARLY_MEDIEVAL, "paschasius radbertus": EARLY_MEDIEVAL,
    "remigius antissiodorensis": EARLY_MEDIEVAL, "atto vercellensis": EARLY_MEDIEVAL,
    "joannes scotus erigena": EARLY_MEDIEVAL, "iohannes scotus erigena": EARLY_MEDIEVAL,
    "odo cluniacensis": EARLY_MEDIEVAL, "flodoardus remensis": EARLY_MEDIEVAL,
    "angelomus luxovensis": EARLY_MEDIEVAL, "ado viennensis": EARLY_MEDIEVAL,
    "ratherius veronensis": EARLY_MEDIEVAL, "freculphus lexovensis": EARLY_MEDIEVAL,
    "nicolaus i": EARLY_MEDIEVAL, "walahfrid strabo": EARLY_MEDIEVAL,
    "einhardus": EARLY_MEDIEVAL, "paulus diaconus": EARLY_MEDIEVAL,
    "smaragdus": EARLY_MEDIEVAL, "florus lugdunensis": EARLY_MEDIEVAL,
    "agobardus": EARLY_MEDIEVAL, "hrothsuita gandersheimensis": EARLY_MEDIEVAL,
    "gregorius turonensis": EARLY_MEDIEVAL, "aldhelmus": EARLY_MEDIEVAL,
    "theodulfus": EARLY_MEDIEVAL, "amalarius": EARLY_MEDIEVAL,
    "sedulius scottus": EARLY_MEDIEVAL, "lupus servatus": EARLY_MEDIEVAL,
    "ionas aurelianensis": EARLY_MEDIEVAL, "prudentius trecensis": EARLY_MEDIEVAL,

    # --- High medieval ---
    "anselmus laudunensis": HIGH_MEDIEVAL, "anselmus cantuariensis": HIGH_MEDIEVAL,
    "urbanus ii": HIGH_MEDIEVAL, "paschalis ii": HIGH_MEDIEVAL,
    "innocentius ii": HIGH_MEDIEVAL, "eugenius iii": HIGH_MEDIEVAL,
    "bernardus claraevallensis": HIGH_MEDIEVAL, "petrus abaelardus": HIGH_MEDIEVAL,
    "petrus lombardus": HIGH_MEDIEVAL, "hugo de sancto victore": HIGH_MEDIEVAL,
    "thomas aquinas": HIGH_MEDIEVAL, "bonaventura": HIGH_MEDIEVAL,
    "albertus magnus": HIGH_MEDIEVAL, "ivo carnotensis": HIGH_MEDIEVAL,
    "petrus damiani": HIGH_MEDIEVAL, "rupertus tuitiensis": HIGH_MEDIEVAL,
    "honorius augustodunensis": HIGH_MEDIEVAL, "gratianus": HIGH_MEDIEVAL,
    "petrus comestor": HIGH_MEDIEVAL, "guillelmus durandus": HIGH_MEDIEVAL,
    "iohannes saresberiensis": HIGH_MEDIEVAL,

    # --- Late medieval ---
    "guillelmus de ockham": LATE_MEDIEVAL, "iohannes buridanus": LATE_MEDIEVAL,
    "duns scotus": LATE_MEDIEVAL, "boccaccio": LATE_MEDIEVAL,
    "petrarca": LATE_MEDIEVAL, "franciscus petrarcha": LATE_MEDIEVAL,
    "thomas a kempis": LATE_MEDIEVAL, "iohannes gerson": LATE_MEDIEVAL,

    # --- Renaissance / neo-Latin ---
    "erasmus": NEO_LATIN, "desiderius erasmus": NEO_LATIN,
    "pius ii": NEO_LATIN, "aeneas silvius piccolomini": NEO_LATIN,
    "thomas morus": NEO_LATIN, "nicolaus copernicus": NEO_LATIN,
    "galileo galilei": NEO_LATIN, "isaac newtonus": NEO_LATIN,
    "franciscus baconus": NEO_LATIN, "spinoza": NEO_LATIN,
    "leonhardus euler": NEO_LATIN, "carolus linnaeus": NEO_LATIN,
    "iohannes kepler": NEO_LATIN, "rene descartes": NEO_LATIN,
    "melanchthon": NEO_LATIN, "lutherus": NEO_LATIN, "calvinus": NEO_LATIN,
    "sebastianus castellio": NEO_LATIN, "ioannes ludovicus vives": NEO_LATIN,
}

# Well-known works whose author is absent from the filename, or masked by "(ed. migne)".
# Matched as a substring of the lowercased work title. Ordered longest-first at match time
# so that "de civitate dei" does not shadow a more specific entry.
WORK_ERA: Dict[str, str] = {
    # Augustine — a large share of the "ed. migne" material
    "de civitate dei": LATE_ANTIQUE,
    "enarrationes in psalmos": LATE_ANTIQUE,
    "in evangelium ioannis tractatus": LATE_ANTIQUE,
    "in ioannis evangelium tractatus": LATE_ANTIQUE,
    "contra iulianum": LATE_ANTIQUE,
    "opus imperfectum contra": LATE_ANTIQUE,
    "confessiones": LATE_ANTIQUE,
    "de doctrina christiana": LATE_ANTIQUE,
    "de trinitate": LATE_ANTIQUE,
    "retractationes": LATE_ANTIQUE,
    "dissertationes de historia pelagiana": NEO_LATIN,
    # Other late antique
    "moralia": LATE_ANTIQUE,                      # Gregory the Great, Moralia in Iob
    "in vergilii aeneidem commentarii": LATE_ANTIQUE,   # Servius
    "etymologiarum libri": LATE_ANTIQUE,          # Isidore
    "institutiones divinarum": LATE_ANTIQUE,
    "de consolatione philosophiae": LATE_ANTIQUE,
    "historia ecclesiastica": LATE_ANTIQUE,
    "vulgata": LATE_ANTIQUE,
    "biblia sacra": LATE_ANTIQUE,
    "codex amiatinus": LATE_ANTIQUE,
    # Classical
    "ab urbe condita": CLASSICAL,
    "naturalis historia": CLASSICAL,
    "de institutione oratoria": CLASSICAL,
    "de bello gallico": CLASSICAL,
    "de bello civili": CLASSICAL,
    "de re coquinaria": CLASSICAL,
    "de lingua latina": CLASSICAL,
    "de architectura": CLASSICAL,
    "strategemata": CLASSICAL,
    "mathesis": CLASSICAL,
    "metamorphoses": CLASSICAL,
    "de rerum natura": CLASSICAL,
    "de viris illustribus urbis romae": CLASSICAL,
    # High / late medieval
    "summa theologiae": HIGH_MEDIEVAL,
    "summa contra gentiles": HIGH_MEDIEVAL,
    "historia scholastica": HIGH_MEDIEVAL,        # Peter Comestor
    "logica ingredientibus": HIGH_MEDIEVAL,       # Abelard
    "historia regum britanniae": HIGH_MEDIEVAL,
    "libri v decretalium": HIGH_MEDIEVAL,
    "collectio decretalium": HIGH_MEDIEVAL,
    "de divisione naturae": EARLY_MEDIEVAL,
    "sermones in canticum canticorum": HIGH_MEDIEVAL,   # Bernard
    "summa logicae": LATE_MEDIEVAL,
    "summulae de dialectica": LATE_MEDIEVAL,
    "quaestiones subtilissimae in metaphysicen": LATE_MEDIEVAL,
    "de genealogiis deorum gentilium": LATE_MEDIEVAL,   # Boccaccio
    "malleus maleficarum": LATE_MEDIEVAL,
    "de imitatione christi": LATE_MEDIEVAL,
    # Neo-Latin
    "erasmi colloquia": NEO_LATIN,
    "de revolutionibus orbium": NEO_LATIN,
    "discorsi e dimostrazioni": NEO_LATIN,
    "le opere di galileo": NEO_LATIN,
    "ad exercitia linguae latinae dialogi": NEO_LATIN,
    "de interpretatione recta": NEO_LATIN,
    # Multi-era collections
    "gallia christiana": COMPILATION,
    "patrologia latina": COMPILATION,
    "enchiridion symbolorum": COMPILATION,
    "regesta sive epistolae": COMPILATION,
    "acta sanctorum": COMPILATION,
    "monumenta germaniae": COMPILATION,
    "concilium": COMPILATION,
    "missale mixtum": COMPILATION,
    "liber sacramentorum": COMPILATION,
    "hymni ecclesiae": COMPILATION,
    "eclogae latinae": COMPILATION,
}

# Patrologia Latina is ordered chronologically (vol. 1 Tertullian ... vol. 217 Innocent
# III), so the volume number in a scan-page filename dates its contents.
PL_VOLUME_RE = re.compile(r"patrologia latina[ ,]*(\d{1,3})|(?:^|\W)pl[ _]*(\d{1,3})\b", re.I)


def era_from_pl_volume(title: str) -> Optional[str]:
    m = PL_VOLUME_RE.search(title)
    if not m:
        return None
    vol = int(m.group(1) or m.group(2))
    if vol < 1 or vol > 217:
        return None
    if vol <= 79:
        return LATE_ANTIQUE      # through ~600
    if vol <= 129:
        return EARLY_MEDIEVAL    # ~600-900
    if vol <= 190:
        return EARLY_MEDIEVAL    # ~900-1100, mostly still early medieval
    return HIGH_MEDIEVAL         # ~1100-1216


# Work-title keyword -> genre. Checked in order; first match wins.
GENRE_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(commentari|expositio|enarratio|glossa|catena|"
                r"quaestiones? in|tractatus super|breviarium in)", re.I), "exegesis"),
    (re.compile(r"\b(epistol|epistul|litterae|regesta)", re.I), "letters"),
    (re.compile(r"\b(sermo|homili|conciones)", re.I), "sermons"),
    (re.compile(r"\b(historia|chronic|annales|gesta|res gestae|de bello|"
                r"vita|vitae|passio|acta|martyr|legenda)", re.I), "history_hagiography"),
    (re.compile(r"\b(summa|logic|dialectic|metaphysic|quaestiones|sententiae|"
                r"de anima|physic|ethic|topica|analytic)", re.I), "philosophy_scholastic"),
    (re.compile(r"\b(carmen|carmina|hymn|poema|eclog|elegi|versus|"
                r"aeneid|metamorph|satur)", re.I), "poetry"),
    (re.compile(r"\b(decret|canon|concili|constitutio|privilegi|lex|leges|"
                r"institutiones|digest|bulla)", re.I), "law_canon"),
    (re.compile(r"\b(missal|sacramentari|breviari|liturg|officium|ordo romanus|"
                r"antiphon|benedictio|oratio)", re.I), "liturgy"),
    (re.compile(r"\b(biblia|vulgata|evangeli|genesis|exodus|psalm|apocalyps|"
                r"testamentum|pentateuch)", re.I), "scripture"),
    (re.compile(r"\b(de re coquinaria|naturalis historia|de architectura|mathesis|"
                r"astronom|medicin|herbari|de universo|etymolog|computo|"
                r"revolutionibus|de lingua)", re.I), "science_technical"),
    (re.compile(r"\b(colloqui|dialogi|de institutione oratoria|rhetoric|"
                r"declamatio|orationes)", re.I), "rhetoric_dialogue"),
    (re.compile(r"\b(regula|de moribus|confessio|meditatio|speculum|"
                r"de imitatione)", re.I), "monastic_devotional"),
]

# Whole-word markers of non-Latin text. Deliberately words that are not also Latin.
NON_LATIN_WORDS = re.compile(
    r"\b(the|and|of|that|with|from|this|which|there|would|about|"
    r"und|der|die|das|nicht|eine|auch|aber|sich|dieser|"
    r"les|des|une|dans|pour|avec|cette|nous|vous|"
    r"della|degli|questo|essere)\b", re.I)

SUSPICIOUS_CHARS = re.compile(r"[^a-zA-ZāēīōūăĕĭŏŭæœÆŒ\s\.,;:!?'\"()\[\]\-–—0-9/*&%†‡§]")


def author_from_title(title: str) -> Optional[str]:
    """Extract a parenthetical author, rejecting editors/series/volume numbers."""
    for match in re.findall(r"\(([^)]+)\)", title):
        cand = match.strip().lower()
        if not cand or NON_AUTHOR_PAREN.match(cand):
            continue
        return cand
    return None


def era_for(author: Optional[str], title: str) -> str:
    """Era for a document, most-reliable evidence first.

    Order matters: a named author beats a work title, and a work title beats the
    Patrologia Latina volume number (which dates the volume's contents only approximately).
    """
    if author and author in AUTHOR_ERA:
        return AUTHOR_ERA[author]

    t = title.lower()

    # Titles that name their author without parentheses.
    for name, era in AUTHOR_ERA.items():
        if len(name) > 8 and name in t:
            return era

    # Known works, longest title first so specific entries win over generic ones.
    for work in sorted(WORK_ERA, key=len, reverse=True):
        if work in t:
            return WORK_ERA[work]

    pl = era_from_pl_volume(t)
    if pl:
        return pl

    return UNKNOWN


def genre_for(title: str) -> str:
    for pattern, genre in GENRE_RULES:
        if pattern.search(title):
            return genre
    return UNKNOWN


def verse_signal(text: str, is_scan_page: bool) -> str:
    """'verse' | 'prose' | 'unknown', from line-length statistics.

    Scan pages keep the physical line breaks of the printed page, which mimics verse, so
    the signal is reported as unknown for them rather than guessed wrongly.
    """
    if is_scan_page:
        return UNKNOWN
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if len(ln) > 3]
    if len(lines) < 10:
        return UNKNOWN
    lengths = [len(ln) for ln in lines]
    median = statistics.median(lengths)
    stdev = statistics.pstdev(lengths)
    cv = stdev / median if median else 99.0
    # Verse: short, tightly clustered line lengths.
    if 18 <= median <= 72 and cv < 0.45:
        return "verse"
    if median > 90:
        return "prose"
    return "prose" if cv >= 0.45 else UNKNOWN


def ocr_quality(text: str) -> float:
    """0..1, higher is cleaner. Fraction of characters that look like plausible Latin."""
    if not text:
        return 0.0
    sample = text[:200_000]
    suspicious = len(SUSPICIOUS_CHARS.findall(sample))
    # Words containing digits mid-token are a classic OCR/apparatus artifact.
    broken = len(re.findall(r"\b\w*\d+\w*[a-z]\w*\b", sample))
    score = 1.0 - (suspicious / len(sample)) - min(0.3, broken / max(1, len(sample.split())))
    return round(max(0.0, min(1.0, score)), 4)


def non_latin_ratio(text: str) -> float:
    """Rate of unmistakably non-Latin function words per 1000 words."""
    words = text[:200_000].split()
    if not words:
        return 0.0
    return round(1000.0 * len(NON_LATIN_WORDS.findall(text[:200_000])) / len(words), 3)


# --- Orthography ------------------------------------------------------------------------
# The corpus mixes editorial conventions: some editions print consonantal u as "v"
# ("veni"), others as "u" ("ueni"); some distinguish i/j; some mark vowel length. That
# variation is edition noise, not Latin, and a 32M model spends capacity modelling it.
# Standardising is therefore usually worth it -- but it is destructive and IRREVERSIBLE in
# the encoded data, so it is a explicit choice with levels rather than a silent default.

_MACRONS = str.maketrans("āēīōūȳăĕĭŏŭўÁÉÍÓÚáéíóú", "aeiouyaeiouyAEIOUaeiou")


def standardize_orthography(text: str, level: str) -> str:
    """Normalize editorial orthography.

    none          leave the text exactly as the edition printed it.
    conservative  strip vowel-length macrons and combining diacritics only; u/v and i/j
                  distinctions are preserved.
    classical     also fold consonantal v->u and j->i, giving the "ueni uidi uici"
                  convention used by much of this corpus already.
    modern        fold the other way, u->v and i->j in consonantal position, giving the
                  "veni vidi vici" convention.

    Note that "classical" and "modern" are not fully invertible: folding v->u loses the
    distinction the edition encoded. Choose one and stay with it, because changing level
    changes the tokenizer's effective vocabulary and invalidates comparisons with
    checkpoints trained at a different level.
    """
    if level == "none":
        return text

    # Decompose, drop combining marks, recompose: removes macrons and breves generally.
    text = text.translate(_MACRONS)
    text = "".join(c for c in unicodedata.normalize("NFD", text)
                   if not unicodedata.combining(c))
    text = unicodedata.normalize("NFC", text)

    if level == "conservative":
        return text

    if level == "classical":
        text = text.replace("v", "u").replace("V", "U")
        text = text.replace("j", "i").replace("J", "I")
        return text

    if level == "modern":
        # Consonantal u is u followed by a vowel at the start of a syllable. A full
        # treatment needs syllabification; this handles the common word-initial and
        # intervocalic cases and leaves the rest alone rather than guessing.
        text = re.sub(r"\b([Uu])(?=[aeiouAEIOU])",
                      lambda m: "V" if m.group(1).isupper() else "v", text)
        text = re.sub(r"(?<=[aeiou])u(?=[aeiou])", "v", text)
        text = re.sub(r"\b([Ii])(?=[aeou])",
                      lambda m: "J" if m.group(1).isupper() else "j", text)
        return text

    raise ValueError(f"unknown orthography level {level!r}")


# --- Fragments ---------------------------------------------------------------------------

def fragment_score(text: str) -> float:
    """0..1, higher means "more likely a fragment/stub rather than continuous prose".

    Fragments are a real problem in this corpus: incipits, one-line charters, tables of
    contents, index stubs and lacunose excerpts teach the model to start and abandon
    sentences.
    """
    stripped = text.strip()
    if not stripped:
        return 1.0

    words = stripped.split()
    n_words = len(words)
    if n_words < 40:
        return 1.0

    # Editorial lacuna markers and ellipses.
    lacunae = len(re.findall(r"\.\.\.|\[\s*\.\.\.\s*\]|…|\*\*\*|†", stripped))
    lacuna_rate = lacunae / max(1, n_words / 100)

    # Sentence-ending punctuation: continuous prose has some; lists and stubs have little.
    sentences = len(re.findall(r"[.!?;:]", stripped))
    sentence_rate = sentences / max(1, n_words / 100)

    score = 0.0
    if n_words < 120:
        score += 0.4
    if sentence_rate < 2.0:
        score += 0.3
    if lacuna_rate > 5.0:
        score += 0.3
    return round(min(1.0, score), 3)


def classify(text: str, title: str, is_scan_page: bool) -> Dict[str, object]:
    """All classification signals for one document."""
    author = author_from_title(title)
    return {
        "author": author or UNKNOWN,
        "era": era_for(author, title),
        "genre": genre_for(title),
        "form": verse_signal(text, is_scan_page),
        "ocr_quality": ocr_quality(text),
        "non_latin_per_1k": non_latin_ratio(text),
        "fragment_score": fragment_score(text),
        "source_type": "scan_page" if is_scan_page else "text",
    }
