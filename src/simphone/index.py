"""Build a phonetic index once, save it, and search queries against it.

A large query list is the fast path: queries are split across CPU processes,
and each process runs rapidfuzz on blocks of queries against the shared
inventory. A single query is split the other way, by inventory shard, so it
still uses more than one core.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from simphone.distance import PhoneticDistance
from simphone.g2p import phonemize

_INV_SEQS: list[np.ndarray] = []
_INV_STRS: list[str] = []
_INV_TEXTS: list[str] = []
_Q_SEQS: list[np.ndarray] = []
_Q_STRS: list[str] = []
_Q_TEXTS: list[str] = []
_PD: Optional[PhoneticDistance] = None
_PARAMS: dict = {}


def _fork_context():
    """``fork`` so workers share the inventory. ``None`` where fork does not exist."""
    try:
        return mp.get_context("fork")
    except ValueError:
        return None


def read_span_file(path: os.PathLike) -> list[str]:
    """One span per line. Blank lines and ``#`` comments are skipped. Exact dupes drop."""
    seen: dict[str, None] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        seen.setdefault(s, None)
    return list(seen.keys())


def _unit_count(span: str) -> int:
    return len(span.split())


def _strings(pd: PhoneticDistance, seqs: Sequence[np.ndarray]) -> list[str]:
    chars = pd.id_char_map()
    return ["".join(chars[int(t)] for t in ids) for ids in seqs]


class PhoneticIndex:
    """Inventory of spans, their phone-id sequences, and a shared cost matrix."""

    def __init__(
        self,
        spans: list[str],
        seqs: list[np.ndarray],
        distance: PhoneticDistance,
        *,
        lang: str,
    ) -> None:
        self.spans = spans
        self.seqs = seqs
        self.distance = distance
        self.lang = lang
        self.strs = _strings(distance, seqs)
        self.n_units = [_unit_count(s) for s in spans]

    @classmethod
    def build(
        cls,
        spans: Sequence[str],
        lang: str,
        *,
        tone_weight: float = 0.5,
        device: str = "cpu",
        cache_dir: Optional[os.PathLike] = None,
        en_locale: str = "eng-us",
        workers: int = 0,
    ) -> "PhoneticIndex":
        """Phonemize ``spans`` once and build the panphon cost matrix once."""
        kept = list(dict.fromkeys(s.strip() for s in spans if s and s.strip()))
        phones = phonemize(
            kept, lang, device=device, cache_dir=cache_dir, en_locale=en_locale,
            workers=workers,
        )
        seq_tokens = [phones.get(s, []) for s in kept]
        vocab = list(dict.fromkeys(t for toks in seq_tokens for t in toks))
        distance = PhoneticDistance(vocab, tone_weight=tone_weight)
        seqs = [distance.encode(toks) for toks in seq_tokens]
        return cls(kept, seqs, distance, lang=lang)

    def save(self, path: os.PathLike) -> None:
        """Write ``meta.json``, ``spans.txt``, and ``phones.npz``."""
        dest = Path(path)
        dest.mkdir(parents=True, exist_ok=True)
        meta = {
            "lang": self.lang,
            "tone_weight": self.distance.tone_weight,
            "indel_cost": self.distance.indel_cost,
            "cost_scale": self.distance.cost_scale,
            "n_spans": len(self.spans),
        }
        (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        (dest / "spans.txt").write_text(
            "\n".join(s.replace("\n", " ") for s in self.spans) + ("\n" if self.spans else ""),
            encoding="utf-8",
        )
        offsets = np.zeros(len(self.seqs) + 1, dtype=np.int64)
        chunks: list[np.ndarray] = []
        for i, seq in enumerate(self.seqs):
            chunks.append(np.asarray(seq, dtype=np.int32))
            offsets[i + 1] = offsets[i] + int(chunks[-1].shape[0])
        ids = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int32)
        np.savez(
            dest / "phones.npz",
            vocab=np.array(self.distance.tokens, dtype=object),
            ids=ids,
            offsets=offsets,
            cost=np.asarray(self.distance.cost, dtype=np.float32),
            cost_scale=np.float64(self.distance.cost_scale),
        )

    @classmethod
    def load(cls, path: os.PathLike) -> "PhoneticIndex":
        """Load a directory written by :meth:`save`. Does not phonemize the inventory."""
        dest = Path(path)
        meta = json.loads((dest / "meta.json").read_text(encoding="utf-8"))
        spans = (dest / "spans.txt").read_text(encoding="utf-8").splitlines()
        if spans and spans[-1] == "" :
            spans = spans[:-1]
        # splitlines already drops the trailing newline; keep empty-string spans.
        data = np.load(dest / "phones.npz", allow_pickle=True)
        vocab = [str(t) for t in data["vocab"].tolist()]
        ids = np.asarray(data["ids"], dtype=np.int32)
        offsets = np.asarray(data["offsets"], dtype=np.int64)
        seqs = [ids[int(offsets[i]) : int(offsets[i + 1])] for i in range(len(offsets) - 1)]
        if len(seqs) != len(spans):
            raise ValueError(
                f"{dest}: {len(spans)} spans but {len(seqs)} phone sequences"
            )
        distance = PhoneticDistance.from_matrix(
            vocab,
            np.asarray(data["cost"], dtype=np.float32),
            tone_weight=float(meta["tone_weight"]),
            indel_cost=float(meta["indel_cost"]),
            cost_scale=float(meta.get("cost_scale", data["cost_scale"])),
        )
        return cls(spans, seqs, distance, lang=str(meta["lang"]))

    def search(
        self,
        query: str,
        *,
        topk: int = 50,
        prefilter: int = 300,
        sim_min: float = 0.8,
        workers: int = 0,
        block_size: int = 512,
        device: str = "cpu",
        cache_dir: Optional[os.PathLike] = None,
        en_locale: str = "eng-us",
    ) -> list[dict]:
        """Neighbors for one query. Same arguments as :meth:`search_many`."""
        found = self.search_many(
            [query],
            topk=topk,
            prefilter=prefilter,
            sim_min=sim_min,
            workers=workers,
            block_size=block_size,
            device=device,
            cache_dir=cache_dir,
            en_locale=en_locale,
        )
        return found[0] if found else []

    def search_many(
        self,
        queries: Sequence[str],
        *,
        topk: int = 50,
        prefilter: int = 300,
        sim_min: float = 0.8,
        workers: int = 0,
        block_size: int = 512,
        device: str = "cpu",
        cache_dir: Optional[os.PathLike] = None,
        en_locale: str = "eng-us",
    ) -> list[list[dict]]:
        """One neighbor list per query, same order as ``queries``.

        When there are fewer queries than workers, the inventory is sharded.
        Otherwise the queries are sharded and each worker scans the full inventory
        in blocks of ``block_size``. A neighbor whose text equals the query is dropped.
        """
        q_texts = [q.strip() for q in queries]
        if not q_texts or not self.spans:
            return [[] for _ in q_texts]
        n_workers = workers if workers > 0 else (os.cpu_count() or 1)
        phones = phonemize(
            q_texts,
            self.lang,
            device=device,
            cache_dir=cache_dir,
            en_locale=en_locale,
        )
        tokens = [phones.get(q, []) for q in q_texts]
        n_tok = len(self.distance.tokens)
        self.distance.ensure_tokens(t for toks in tokens for t in toks)
        # New query phones are appended, so previously built inventory strings stay valid.
        if len(self.distance.tokens) != n_tok:
            self.strs = _strings(self.distance, self.seqs)
        q_seqs = [self.distance.encode(toks) for toks in tokens]
        q_strs = _strings(self.distance, q_seqs)

        global _INV_SEQS, _INV_STRS, _INV_TEXTS, _Q_SEQS, _Q_STRS, _Q_TEXTS, _PD, _PARAMS
        _INV_SEQS = self.seqs
        _INV_STRS = self.strs
        _INV_TEXTS = self.spans
        _Q_SEQS = q_seqs
        _Q_STRS = q_strs
        _Q_TEXTS = q_texts
        _PD = self.distance
        _PARAMS = {
            "topk": int(topk),
            "prefilter": int(prefilter),
            "sim_min": float(sim_min),
            "block": int(block_size),
        }

        n_q = len(q_texts)
        if n_q < n_workers and n_workers > 1 and len(self.spans) > 1:
            ranked = _search_shard_inventory(n_workers)
        else:
            use = min(n_workers, n_q)
            ranked = _search_shard_queries(use)

        out: list[list[dict]] = []
        for qi, hits in enumerate(ranked):
            qn = _unit_count(q_texts[qi])
            rows = []
            for j, sim in hits:
                rows.append({
                    "text": self.spans[j],
                    "sim": round(float(sim), 4),
                    "n_units": self.n_units[j],
                    "dlen": self.n_units[j] - qn,
                })
            out.append(rows)
        return out


def _rerank(qi: int, cand: Sequence[int]) -> list[tuple[int, float]]:
    """Feature-distance rerank. Drops the query text and applies ``sim_min``."""
    assert _PD is not None
    q_ids = _Q_SEQS[qi]
    q_text = _Q_TEXTS[qi]
    C = _PD.cost
    indel = _PD.indel_cost
    dp = _PD._dp
    la = int(q_ids.shape[0])
    sim_min = _PARAMS["sim_min"]
    topk = _PARAMS["topk"]
    scored: list[tuple[float, int]] = []
    for j in cand:
        j = int(j)
        if _INV_TEXTS[j] == q_text:
            continue
        b = _INV_SEQS[j]
        m = la if la >= int(b.shape[0]) else int(b.shape[0])
        if m == 0:
            continue
        sim = 1.0 - float(dp(q_ids, b, C, indel)) / m
        if sim < 0.0:
            sim = 0.0
        elif sim > 1.0:
            sim = 1.0
        if sim >= sim_min:
            scored.append((sim, j))
    scored.sort(key=lambda t: -t[0])
    return [(j, sim) for sim, j in scored[:topk]]


def _query_shard_worker(sid: int) -> list[tuple[int, list[tuple[int, float]]]]:
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein

    n_q = len(_Q_STRS)
    mine = list(range(sid, n_q, _worker_count_from_env()))
    return _query_rows(mine, process, Levenshtein)


def _worker_count_from_env() -> int:
    return int(_PARAMS.get("n_workers", 1))


def _query_rows(mine: list[int], process, Levenshtein) -> list[tuple[int, list]]:
    block = _PARAMS["block"]
    prefilter = _PARAMS["prefilter"]
    n_inv = len(_INV_STRS)
    kpre = min(prefilter, n_inv)
    out = []
    for s in range(0, len(mine), block):
        qi = mine[s : s + block]
        qstrs = [_Q_STRS[i] for i in qi]
        D = process.cdist(
            qstrs, _INV_STRS, scorer=Levenshtein.distance, dtype=np.uint16, workers=1
        )
        for r, i in enumerate(qi):
            if kpre <= 0:
                out.append((i, []))
                continue
            row = D[r]
            if kpre >= n_inv:
                cand = range(n_inv)
            else:
                cand = (int(j) for j in np.argpartition(row, kpre - 1)[:kpre])
            out.append((i, _rerank(i, list(cand))))
    return out


def _search_shard_queries(n_workers: int) -> list[list[tuple[int, float]]]:
    ctx = _fork_context()
    if ctx is None or n_workers <= 1:
        _PARAMS["n_workers"] = 1
        parts = [_query_shard_worker(0)]
    else:
        _PARAMS["n_workers"] = n_workers
        with ctx.Pool(processes=n_workers) as pool:
            parts = pool.map(_query_shard_worker, list(range(n_workers)))
    ranked: list[list[tuple[int, float]]] = [[] for _ in range(len(_Q_STRS))]
    for part in parts:
        for qi, hits in part:
            ranked[qi] = hits
    return ranked


def _search_shard_inventory(n_workers: int) -> list[list[tuple[int, float]]]:
    n = len(_INV_STRS)
    # Contiguous shards so each rapidfuzz call scans a tight slice.
    bounds = []
    for sid in range(n_workers):
        start = sid * n // n_workers
        end = (sid + 1) * n // n_workers
        if start < end:
            bounds.append((start, end))
    ctx = _fork_context()
    if ctx is None or len(bounds) <= 1:
        parts = [_inventory_shard_worker(b) for b in bounds]
    else:
        with ctx.Pool(processes=len(bounds)) as pool:
            parts = pool.map(_inventory_shard_worker, bounds)
    prefilter = _PARAMS["prefilter"]
    n_q = len(_Q_STRS)
    merged: list[list[tuple[int, int]]] = [[] for _ in range(n_q)]
    for part in parts:
        for qi, hits in enumerate(part):
            merged[qi].extend(hits)
    ranked = []
    for qi in range(n_q):
        hits = merged[qi]
        if len(hits) > prefilter:
            hits.sort(key=lambda t: t[1])
            hits = hits[:prefilter]
        cand = [j for j, _lev in hits]
        ranked.append(_rerank(qi, cand))
    return ranked


def _inventory_shard_worker(bounds: tuple[int, int]) -> list[list[tuple[int, int]]]:
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein

    start, end = bounds
    sub = _INV_STRS[start:end]
    n_sub = end - start
    prefilter = _PARAMS["prefilter"]
    kpre = min(prefilter, n_sub)
    D = process.cdist(
        _Q_STRS, sub, scorer=Levenshtein.distance, dtype=np.uint16, workers=1
    )
    out: list[list[tuple[int, int]]] = []
    for r in range(len(_Q_STRS)):
        row = D[r]
        if kpre <= 0:
            out.append([])
            continue
        if kpre >= n_sub:
            local = range(n_sub)
        else:
            local = (int(j) for j in np.argpartition(row, kpre - 1)[:kpre])
        out.append([(start + int(j), int(row[int(j)])) for j in local])
    return out
