"""Phonemize spans.

The span text stored in the index is unchanged. Phones come from a spoken-form
normalizer and then a G2P.

Vietnamese uses sea-g2p ``Normalizer`` then ``G2P``, so ``21`` and
``hai mươi mốt`` share one pronunciation. ``punc_norm`` stays off (no forced
trailing period).

English uses WeTextProcessing text normalization and then CharsiuG2P, so
``21`` and ``twenty one`` share one pronunciation. No XPhoneBERT.
"""

from __future__ import annotations

import contextlib
import os
import re
import urllib.request
from pathlib import Path
from typing import Optional, Sequence

from simphone.distance import VI_TONE_MARKS, span_phones

G2P_MODEL_ID = "charsiu/g2p_multilingual_byT5_small_100"
BYT5_ID = "google/byt5-small"
G2P_DICT_URL = "https://raw.githubusercontent.com/lingjzhu/CharsiuG2P/main/dicts/{locale}.tsv"

_EN_INSTALL = (
    'English is not installed. Install torch first, then: pip install "simphone[en]"'
)
_TORCH_INSTALL = (
    "torch is not installed. Install it yourself; this package will not install it. "
    "CPU: pip install torch --index-url https://download.pytorch.org/whl/cpu"
)

# Primary and secondary stress. sea-g2p uses ˌ together with tone digit 2 for huyền.
_STRESS = frozenset({"\u02C8", "\u02CC"})
# Vowels we attach a syllable's tone mark to. ɜ (U+025C) is a tone, not listed.
_VOWELS = frozenset("aeiouyɛɔəɪʊæɑɒʌɨʉɯøœɐɤ")

_FT = None
_EN_G2P: dict[tuple[str, str, str], object] = {}
_EN_TN = None
# sea-g2p marks English code-switching. G2P reads the markers; spoken text does not keep them.
_EN_TAG = re.compile(r"</?en>", re.IGNORECASE)


def default_cache_dir() -> Path:
    """Charsiu assets. ``SIMPHONE_CACHE`` overrides ``~/.cache/simphone``."""
    env = os.environ.get("SIMPHONE_CACHE")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "simphone"


def _missing_modules(names: Sequence[str]) -> list[str]:
    missing = []
    for name in names:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    return missing


def _require_tn() -> None:
    """WeTextProcessing only. Normalization does not need torch or Charsiu."""
    if _missing_modules(("tn",)):
        raise ImportError(_EN_INSTALL) from None


def _require_en() -> None:
    """Fail before English G2P if the extra or torch is missing."""
    if _missing_modules(("text2phonemesequence", "tn")):
        raise ImportError(_EN_INSTALL) from None
    try:
        __import__("torch")
    except ImportError:
        raise ImportError(_TORCH_INSTALL) from None


def _fork_context():
    """``fork`` so workers inherit loaded models. ``None`` where fork does not exist."""
    import multiprocessing as mp

    try:
        return mp.get_context("fork")
    except ValueError:
        return None


@contextlib.contextmanager
def _chdir(path: Path):
    prev = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(prev)


def _feature_table():
    global _FT
    if _FT is None:
        from panphon.featuretable import FeatureTable

        _FT = FeatureTable()
    return _FT


def _segment_ipa(ipa: str) -> list[str]:
    """Phones panphon recognizes. An unrecognized symbol is skipped.

    The raw string is never returned as one phone. A residue panphon did not
    consume (``fɪlᵻpiːnz``, ``lə1w``) must not enter the cost matrix.
    """
    if not ipa:
        return []
    return list(_feature_table().ipa_segs(ipa))


def _attach_tone(segs: list[str], tone: str) -> list[str]:
    if not tone or not segs:
        return segs
    idx = None
    for i, seg in enumerate(segs):
        if any(c in _VOWELS for c in seg):
            idx = i
    if idx is None:
        idx = len(segs) - 1
    out = list(segs)
    out[idx] = out[idx] + tone
    return out


def vi_syllable_tokens(syllable: str) -> list[str]:
    """One sea-g2p syllable to panphon phone tokens, tone suffixed on the vowel.

    ``mˈaɜc`` (mắc) becomes ``["m", "aɜ", "c"]``. ``mˌaː2`` (mà) becomes
    ``["m", "aː2"]``. Stress marks are dropped.
    """
    tone_chars: list[str] = []
    base_chars: list[str] = []
    for c in syllable:
        if c in _STRESS:
            continue
        if c in VI_TONE_MARKS:
            tone_chars.append(c)
            continue
        if c.isascii() and not c.isalnum():
            continue
        base_chars.append(c)
    segs = _segment_ipa("".join(base_chars))
    return _attach_tone(segs, "".join(tone_chars))


def _tokens_from_raw(raw: str) -> list[str]:
    toks: list[str] = []
    for syl in (raw or "").split():
        toks.extend(vi_syllable_tokens(syl))
    return toks


def _tokens_from_raws(raws: list[str]) -> list[list[str]]:
    return [_tokens_from_raw(raw) for raw in raws]


def _spoken_vi(spans: Sequence[str], workers: int = 0) -> list[str]:
    """sea-g2p spoken form, one string per input. ``punc_norm`` stays off."""
    from tqdm import tqdm

    # sea-g2p's Rust pool reads this at first use. Set it before importing.
    n_workers = workers if workers > 0 else (os.cpu_count() or 1)
    os.environ["RAYON_NUM_THREADS"] = str(n_workers)
    from sea_g2p import Normalizer

    normalizer = Normalizer(lang="vi")
    texts = list(spans)
    step = 8192
    spoken: list[str] = []
    for i in tqdm(range(0, len(texts), step), desc="normalize", unit="batch"):
        chunk = texts[i : i + step]
        words = normalizer.normalize(chunk, punc_norm=False)
        if isinstance(words, str):
            words = [words]
        spoken.extend(words)
    return spoken


def _strip_en_tags(text: str) -> str:
    """Remove sea-g2p ``<en>`` markers. The words inside the tag stay."""
    return _EN_TAG.sub("", text)


def _phonemize_vi(spans: Sequence[str], workers: int = 0) -> dict[str, list[str]]:
    from tqdm import tqdm

    n_workers = workers if workers > 0 else (os.cpu_count() or 1)
    unique = list(dict.fromkeys(spans))
    # Sets RAYON_NUM_THREADS before the first sea-g2p import.
    spoken = _spoken_vi(unique, workers=workers)
    from sea_g2p import G2P

    g2p = G2P(lang="vi")
    step = 8192
    batches = range(0, len(unique), step)
    raws: list[str] = []
    for i in tqdm(batches, desc="sea-g2p", unit="batch"):
        chunk = spoken[i : i + step]
        phones = g2p.convert(chunk, punc_norm=False)
        if isinstance(phones, str):
            phones = [phones]
        raws.extend(phones)

    _feature_table()
    seg_step = 1024
    pieces = [raws[i : i + seg_step] for i in range(0, len(raws), seg_step)]
    token_lists: list[list[str]] = []
    ctx = _fork_context()
    if n_workers <= 1 or len(pieces) <= 1 or ctx is None:
        for piece in tqdm(pieces, desc="segment", unit="batch"):
            token_lists.extend(_tokens_from_raws(piece))
    else:
        with ctx.Pool(processes=min(n_workers, len(pieces))) as pool:
            for part in tqdm(
                pool.imap(_tokens_from_raws, pieces),
                total=len(pieces),
                desc="segment",
                unit="batch",
            ):
                token_lists.extend(part)
    return {span: toks for span, toks in zip(unique, token_lists)}


def _resolve_device(device: str) -> str:
    import torch

    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    return device if use_cuda else "cpu"


def _build_en_g2p(locale: str, device: str, cache_dir: Path):
    """Charsiu ``Text2PhonemeSequence`` with the dict and models under ``cache_dir``."""
    from huggingface_hub import snapshot_download
    from text2phonemesequence import Text2PhonemeSequence

    dicts_dir = cache_dir / "charsiu_dicts"
    dicts_dir.mkdir(parents=True, exist_ok=True)
    tsv = dicts_dir / f"{locale}.tsv"
    if not tsv.exists():
        urllib.request.urlretrieve(G2P_DICT_URL.format(locale=locale), str(tsv))
    g2p_path = snapshot_download(G2P_MODEL_ID, cache_dir=str(cache_dir))
    tok_path = snapshot_download(BYT5_ID, cache_dir=str(cache_dir))
    is_cuda = str(device).startswith("cuda")
    with _chdir(dicts_dir):
        return Text2PhonemeSequence(
            pretrained_g2p_model=g2p_path,
            tokenizer=tok_path,
            language=locale,
            is_cuda=is_cuda,
        )


def _en_g2p(locale: str, device: str, cache_dir: Path):
    key = (locale, device, str(cache_dir))
    g = _EN_G2P.get(key)
    if g is None:
        g = _build_en_g2p(locale, device, cache_dir)
        _EN_G2P[key] = g
    return g


def _phonemize_en_units(
    units: Sequence[str], locale: str, device: str, cache_dir: Path, gen_batch_size: int
) -> dict[str, list[str]]:
    import torch

    g2p = _en_g2p(locale, device, cache_dir)
    raw: dict[str, str] = {}
    oov: list[str] = []
    for u in units:
        if u in g2p.phone_dict:
            raw[u] = g2p.phone_dict[u][0]
        elif u in g2p.punctuation:
            raw[u] = u
        else:
            oov.append(u)
    from tqdm import tqdm

    max_len = g2p.phoneme_length.get(locale + ".tsv", 50)
    for i in tqdm(range(0, len(oov), gen_batch_size), desc="charsiu", unit="batch"):
        chunk = oov[i : i + gen_batch_size]
        prompts = ["<" + locale + ">: " + w for w in chunk]
        enc = g2p.tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors="pt")
        if g2p.is_cuda:
            enc = {k: v.cuda() for k, v in enc.items()}
        with torch.no_grad():
            preds = g2p.model.generate(**enc, num_beams=1, max_length=max_len)
        phones = g2p.tokenizer.batch_decode(preds.tolist(), skip_special_tokens=True)
        for w, p in zip(chunk, phones):
            raw[w] = p
    out: dict[str, list[str]] = {}
    for u, p in tqdm(raw.items(), desc="segment", unit="word"):
        try:
            s = g2p.segment_tool(p, ipa=True)
        except Exception:
            try:
                s = g2p.segment_tool(p)
            except Exception:
                s = ""
        out[u] = (s or "").split()
    return out


def _en_normalizer():
    """One WeTextProcessing English TN. The FST is built once per process."""
    global _EN_TN
    if _EN_TN is None:
        import logging

        from tn.english.normalizer import Normalizer as EnNormalizer

        # WeTextProcessing logs "found existing fst" at INFO on its own handler.
        logging.getLogger("wetext").setLevel(logging.WARNING)
        _EN_TN = EnNormalizer()
    return _EN_TN


def _spoken_en(spans: Sequence[str]) -> list[str]:
    """Spoken words for each span. An empty normalization keeps the original."""
    from tqdm import tqdm

    norm = _en_normalizer()
    spoken: list[str] = []
    for span in tqdm(spans, desc="wetext", unit="span"):
        words = norm.normalize(span)
        if not isinstance(words, str) or not words.strip():
            words = span
        spoken.append(words)
    return spoken


def _phonemize_en(
    spans: Sequence[str],
    *,
    device: str,
    cache_dir: Path,
    en_locale: str,
) -> dict[str, list[str]]:
    device = _resolve_device(device)
    spoken = _spoken_en(spans)
    per_span = [s.split() for s in spoken]
    units = list(dict.fromkeys(u for parts in per_span for u in parts))
    unit_tokens = _phonemize_en_units(units, en_locale, device, cache_dir, 128)
    return {span: span_phones(parts, unit_tokens) for span, parts in zip(spans, per_span)}


def phonemize(
    spans: Sequence[str],
    lang: str,
    *,
    device: str = "cpu",
    cache_dir: Optional[os.PathLike] = None,
    en_locale: str = "eng-us",
    workers: int = 0,
) -> dict[str, list[str]]:
    """Map each span string to phone tokens.

    ``lang="vi"`` calls sea-g2p on the whole span (``"việt nam"`` is one call).
    ``lang="en"`` runs WeTextProcessing on the whole span, then CharsiuG2P on
    each spoken word. Tones stay on the tokens (Vietnamese suffix, English Chao
    token). The returned dict is keyed by the original span, not the spoken form.
    """
    if lang == "vi":
        return _phonemize_vi(spans, workers=workers)
    if lang == "en":
        _require_en()
        cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
        return _phonemize_en(spans, device=device, cache_dir=cache, en_locale=en_locale)
    raise ValueError(f"lang must be 'vi' or 'en', got {lang!r}")


def normalize(
    spans: Sequence[str],
    lang: str,
    *,
    workers: int = 0,
) -> list[str]:
    """Spoken form of each span, same order as ``spans``. Duplicates are kept.

    ``lang="vi"`` uses sea-g2p with ``punc_norm=False``. sea-g2p wraps English
    runs in ``<en>...</en>``; those markers are removed from this text.
    Phonemization still sees them, so those runs are read as English.
    ``lang="en"`` uses WeTextProcessing and does not need torch. ``workers``
    sets sea-g2p's Rayon thread count (``0`` = all CPUs) and is ignored for English.
    """
    if lang == "vi":
        return [_strip_en_tags(s) for s in _spoken_vi(spans, workers=workers)]
    if lang == "en":
        _require_tn()
        return _spoken_en(spans)
    raise ValueError(f"lang must be 'vi' or 'en', got {lang!r}")
