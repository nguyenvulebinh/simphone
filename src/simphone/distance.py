"""Feature-weighted phonetic edit distance.

Substitution cost is the panphon articulatory distance between base phones,
normalized to ``[0, 1]``, plus ``tone_weight`` when the tone marks differ.
panphon ignores tone, so the tone term is what keeps tone in the score.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# Charsiu English tones: Chao letters U+02E5..U+02E9, sometimes with U+02C0 (ˀ).
CHAO_TONE_CHARS = frozenset(chr(c) for c in range(0x02E5, 0x02EA))
CHAO_EXTRA = frozenset({"\u02C0"})

# sea-g2p Vietnamese tones, peeled out of the syllable before segmentation.
# Confirmed on ma/mà/mả/mã/má/mạ: huyền=2, hỏi=4, ngã=5, nặng=6, sắc=U+025C (ɜ).
# Ngang has no mark. ɜ is a tone glyph here, not the IPA vowel.
VI_TONE_MARKS = frozenset({"2", "4", "5", "6", "\u025c"})

TONE_CHARS = CHAO_TONE_CHARS | CHAO_EXTRA | VI_TONE_MARKS

_DP_FUNC = None
_FT = None


def _is_single_phone(base: str) -> bool:
    """True when panphon treats ``base`` as exactly one segment."""
    if not base:
        return False
    global _FT
    if _FT is None:
        from panphon.featuretable import FeatureTable

        _FT = FeatureTable()
    segs = list(_FT.ipa_segs(base))
    return len(segs) == 1 and segs[0] == base


def is_tone_token(token: str) -> bool:
    """True if ``token`` is only Chao tone letters (and optional ˀ)."""
    return bool(token) and any(c in CHAO_TONE_CHARS for c in token) and all(
        c in CHAO_TONE_CHARS or c in CHAO_EXTRA for c in token
    )


def split_base_tone(token: str) -> tuple[str, str]:
    """Split a phone token into ``(base_ipa, tone_str)``.

    Chao-only tokens yield ``("", tone)``. A sea-g2p vowel with a trailing
    Vietnamese tone mark (``aː2``, ``aɜ``) yields the base vowel and that mark.
    A one-character ``ɜ`` is left as a base phone so an English NURSE vowel is
    not treated as Vietnamese sắc.
    """
    if is_tone_token(token):
        return "", token
    if len(token) > 1 and token[-1] in VI_TONE_MARKS:
        return token[:-1], token[-1]
    base = "".join(c for c in token if c not in CHAO_TONE_CHARS and c not in CHAO_EXTRA)
    tone = "".join(c for c in token if c in CHAO_TONE_CHARS or c in CHAO_EXTRA)
    return base, tone


def span_phones(
    units: Sequence[str], unit_tokens: dict[str, Sequence[str]]
) -> list[str]:
    """Concatenate per-unit phone-token lists (no word-boundary token)."""
    toks: list[str] = []
    for u in units:
        toks.extend(unit_tokens.get(u, ()))
    return toks


def _get_dp():
    """Weighted Levenshtein ``dp(a, b, C, indel) -> float``. Numba when available."""
    global _DP_FUNC
    if _DP_FUNC is not None:
        return _DP_FUNC

    def _dp(a, b, C, indel):
        la = a.shape[0]
        lb = b.shape[0]
        if la == 0:
            return lb * indel
        if lb == 0:
            return la * indel
        prev = np.empty(lb + 1, dtype=np.float32)
        cur = np.empty(lb + 1, dtype=np.float32)
        for j in range(lb + 1):
            prev[j] = j * indel
        for i in range(1, la + 1):
            cur[0] = i * indel
            ai = a[i - 1]
            for j in range(1, lb + 1):
                sub = prev[j - 1] + C[ai, b[j - 1]]
                dele = prev[j] + indel
                ins = cur[j - 1] + indel
                m = sub
                if dele < m:
                    m = dele
                if ins < m:
                    m = ins
                cur[j] = m
            for j in range(lb + 1):
                prev[j] = cur[j]
        return prev[lb]

    try:
        from numba import njit

        _DP_FUNC = njit(cache=True)(_dp)
    except Exception:
        _DP_FUNC = _dp
    return _DP_FUNC


class PhoneticDistance:
    """Cost matrix and weighted edit distance over a fixed phone-token list."""

    def __init__(
        self,
        tokens: Sequence[str],
        *,
        tone_weight: float = 0.5,
        indel_cost: float = 1.0,
    ) -> None:
        self.tone_weight = float(tone_weight)
        self.indel_cost = float(indel_cost)
        self.tokens = list(dict.fromkeys(tokens))
        self.tok2id = {t: i for i, t in enumerate(self.tokens)}
        self.cost_scale = 1.0
        self._bases: list[str] = []
        self._Cb: Optional[np.ndarray] = None
        self._tok_base: list[int] = []
        self._tok_tone: list[str] = []
        self.cost = self._build_cost()
        self._dp = _get_dp()

    @classmethod
    def from_matrix(
        cls,
        tokens: Sequence[str],
        cost: np.ndarray,
        *,
        tone_weight: float,
        indel_cost: float,
        cost_scale: float,
    ) -> "PhoneticDistance":
        """Restore a matrix saved by :meth:`PhoneticIndex.save` (no panphon)."""
        obj = cls.__new__(cls)
        obj.tone_weight = float(tone_weight)
        obj.indel_cost = float(indel_cost)
        obj.tokens = list(tokens)
        obj.tok2id = {t: i for i, t in enumerate(obj.tokens)}
        obj.cost = np.asarray(cost, dtype=np.float32)
        obj.cost_scale = float(cost_scale) if cost_scale and cost_scale > 0 else 1.0
        obj._dp = _get_dp()
        obj._recover_components()
        obj._rescale_single_phones()
        return obj

    def _base_cost_matrix(self, bases: list[str]) -> tuple[np.ndarray, float]:
        """Normalized ``[0,1]`` panphon distances. Empty base is the null segment.

        ``cost_scale`` is the largest distance between two single panphon
        phones. A multi-phone string left in the inventory cannot set the
        scale, and its normalized cost is capped at 1 (one indel).
        """
        from panphon.distance import Distance

        dist = Distance()
        n = len(bases)
        raw = np.zeros((n, n), dtype=np.float64)
        from tqdm import tqdm

        real = [i for i, b in enumerate(bases) if b]
        for x in tqdm(range(len(real)), desc="panphon", unit="phone"):
            for y in range(x + 1, len(real)):
                i, j = real[x], real[y]
                try:
                    v = float(dist.weighted_feature_edit_distance(bases[i], bases[j]))
                except Exception:
                    v = -1.0
                if v != v:
                    v = -1.0
                raw[i, j] = raw[j, i] = v
        single = [i for i in real if _is_single_phone(bases[i])]
        vals = [
            raw[single[x], single[y]]
            for x in range(len(single))
            for y in range(x + 1, len(single))
            if raw[single[x], single[y]] > 0
        ]
        if vals:
            scale = float(max(vals))
        else:
            finite = raw[raw > 0]
            scale = float(finite.max()) if finite.size else 1.0
        if scale <= 0:
            scale = 1.0
        raw[raw < 0] = scale
        raw = raw / scale
        np.minimum(raw, 1.0, out=raw)
        for i, b in enumerate(bases):
            if not b:
                for j in range(n):
                    raw[i, j] = raw[j, i] = 0.0 if i == j else 1.0
        return raw, scale

    def _split_all(self, tokens: Sequence[str]) -> tuple[list[str], list[str]]:
        bases: list[str] = []
        tones: list[str] = []
        for t in tokens:
            b, tn = split_base_tone(t)
            bases.append(b)
            tones.append(tn)
        return bases, tones

    def _build_cost(self) -> np.ndarray:
        bases, tones = self._split_all(self.tokens)
        uniq = list(dict.fromkeys(bases))
        base2id = {b: i for i, b in enumerate(uniq)}
        Cb, scale = self._base_cost_matrix(uniq)
        self._bases = uniq
        self._Cb = Cb
        self.cost_scale = scale
        self._tok_base = [base2id[b] for b in bases]
        self._tok_tone = tones
        return self._cost_from_components(self._tok_base, self._tok_tone)

    def _cost_from_components(self, bidx: Sequence[int], tones: Sequence[str]) -> np.ndarray:
        assert self._Cb is not None
        k = len(bidx)
        C = np.zeros((k, k), dtype=np.float32)
        for i in range(k):
            bi = bidx[i]
            ti = tones[i]
            for j in range(k):
                c = float(self._Cb[bi, bidx[j]])
                if ti != tones[j]:
                    c += self.tone_weight
                C[i, j] = c
        np.fill_diagonal(C, 0.0)
        return C

    def _recover_components(self) -> None:
        """Rebuild the base-cost matrix from the saved full-token matrix."""
        bases, tones = self._split_all(self.tokens)
        uniq = list(dict.fromkeys(bases))
        base2id = {b: i for i, b in enumerate(uniq)}
        first: dict[str, int] = {}
        for i, b in enumerate(bases):
            first.setdefault(b, i)
        n = len(uniq)
        Cb = np.zeros((n, n), dtype=np.float64)
        for a in uniq:
            ia = first[a]
            for b in uniq:
                ib = first[b]
                c = float(self.cost[ia, ib])
                if tones[ia] != tones[ib]:
                    c -= self.tone_weight
                if c < 0.0:
                    c = 0.0
                Cb[base2id[a], base2id[b]] = c
        self._bases = uniq
        self._Cb = Cb
        self._tok_base = [base2id[b] for b in bases]
        self._tok_tone = tones

    def _rescale_single_phones(self) -> None:
        """Re-normalize a loaded matrix by the largest single-phone distance.

        Indexes saved before this rule store a scale set by unsegmented
        strings. Raw distances are ``normalized * cost_scale``, so the matrix
        can be rescaled without panphon.
        """
        assert self._Cb is not None
        old = self.cost_scale if self.cost_scale > 0 else 1.0
        single = [i for i, b in enumerate(self._bases) if _is_single_phone(b)]
        mx = 0.0
        for x in range(len(single)):
            ix = single[x]
            for y in range(x + 1, len(single)):
                raw = float(self._Cb[ix, single[y]]) * old
                if raw > mx:
                    mx = raw
        if mx <= 0.0:
            return
        factor = old / mx
        Cb = np.clip(np.asarray(self._Cb, dtype=np.float64) * factor, 0.0, 1.0)
        np.fill_diagonal(Cb, 0.0)
        n = len(self._bases)
        for i, b in enumerate(self._bases):
            if not b:
                for j in range(n):
                    Cb[i, j] = Cb[j, i] = 0.0 if i == j else 1.0
        self._Cb = Cb
        self.cost_scale = mx
        self.cost = self._cost_from_components(self._tok_base, self._tok_tone)

    def _pair_base(self, dist, a: str, b: str) -> float:
        if a == b:
            return 0.0
        if not a or not b:
            return 1.0
        try:
            v = float(dist.weighted_feature_edit_distance(a, b))
        except Exception:
            return 1.0
        if v != v or v < 0:
            return 1.0
        scale = self.cost_scale if self.cost_scale > 0 else 1.0
        out = v / scale
        return 1.0 if out > 1.0 else out

    def ensure_tokens(self, tokens: Sequence[str]) -> None:
        """Append phones that are not in the matrix.

        Existing inventory ids stay valid. A new base phone gets panphon costs
        against the bases already stored; the saved inventory block is not rebuilt.
        """
        new: list[str] = []
        for t in tokens:
            if t not in self.tok2id and t not in new:
                new.append(t)
        if not new:
            return
        assert self._Cb is not None
        from panphon.distance import Distance

        dist = Distance()
        new_bases, new_tones = self._split_all(new)
        extra: list[str] = []
        known = set(self._bases)
        for b in new_bases:
            if b not in known:
                known.add(b)
                extra.append(b)
        if extra:
            old_n = len(self._bases)
            add_n = len(extra)
            grown = np.zeros((old_n + add_n, old_n + add_n), dtype=np.float64)
            grown[:old_n, :old_n] = self._Cb
            for i, b in enumerate(extra):
                bi = old_n + i
                for j, ob in enumerate(self._bases):
                    grown[bi, j] = grown[j, bi] = self._pair_base(dist, b, ob)
                for k in range(i):
                    grown[bi, old_n + k] = grown[old_n + k, bi] = self._pair_base(
                        dist, b, extra[k]
                    )
            self._bases = list(self._bases) + extra
            self._Cb = grown
        base2id = {b: i for i, b in enumerate(self._bases)}
        old_k = len(self.tokens)
        bidx = self._tok_base + [base2id[b] for b in new_bases]
        tones = self._tok_tone + new_tones
        C = np.zeros((old_k + len(new), old_k + len(new)), dtype=np.float32)
        C[:old_k, :old_k] = self.cost
        for i in range(old_k, old_k + len(new)):
            for j in range(old_k + len(new)):
                if i == j:
                    continue
                c = float(self._Cb[bidx[i], bidx[j]])
                if tones[i] != tones[j]:
                    c += self.tone_weight
                C[i, j] = C[j, i] = c
        np.fill_diagonal(C, 0.0)
        self.cost = C
        for i, t in enumerate(new):
            self.tok2id[t] = old_k + i
        self.tokens.extend(new)
        self._tok_base = bidx
        self._tok_tone = tones

    def id_char_map(self, base: int = 0x100) -> list[str]:
        """One distinct unicode char per token id, for token-level rapidfuzz."""
        out: list[str] = []
        for idx in range(len(self.tokens)):
            c = base + idx
            if 0xD800 <= c <= 0xDFFF:
                c += 0x800
            out.append(chr(c))
        return out

    def encode(self, tokens: Sequence[str]) -> np.ndarray:
        """Phone tokens to int32 ids. Unknown tokens are dropped."""
        return np.array(
            [self.tok2id[t] for t in tokens if t in self.tok2id], dtype=np.int32
        )

    def distance(self, a_ids: np.ndarray, b_ids: np.ndarray) -> float:
        return float(self._dp(a_ids, b_ids, self.cost, self.indel_cost))

    def normalized_similarity(self, a_ids: np.ndarray, b_ids: np.ndarray) -> float:
        la, lb = int(a_ids.shape[0]), int(b_ids.shape[0])
        m = max(la, lb)
        if m == 0:
            return 1.0
        sim = 1.0 - self.distance(a_ids, b_ids) / m
        if sim < 0.0:
            return 0.0
        if sim > 1.0:
            return 1.0
        return sim
