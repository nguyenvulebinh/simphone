"""Command line for building and searching a phonetic index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from simphone.g2p import normalize
from simphone.index import PhoneticIndex, read_span_file


def _queries(args: argparse.Namespace) -> list[str]:
    out = list(args.query or [])
    if args.queries:
        out.extend(read_span_file(args.queries))
    return out


def _print_hits(query: str, hits: list[dict]) -> None:
    print(query)
    if not hits:
        print("  (none)")
        return
    for h in hits:
        dlen = h["dlen"]
        sign = f"{dlen:+d}"
        print(f"  {h['text']}\t{h['sim']:.4f}\tn={h['n_units']}\tdlen={sign}")


def cmd_norm(args: argparse.Namespace) -> int:
    texts = list(args.text or [])
    if args.queries:
        texts.extend(read_span_file(args.queries))
    if not texts:
        print("pass --text and/or --queries", file=sys.stderr)
        return 1
    for line in normalize(texts, args.lang, workers=args.workers):
        print(line)
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    spans = read_span_file(args.inventory)
    if not spans:
        print(f"no spans in {args.inventory}", file=sys.stderr)
        return 1
    print(f"phonemizing {len(spans)} spans ({args.lang})", flush=True)
    index = PhoneticIndex.build(
        spans,
        args.lang,
        tone_weight=args.tone_weight,
        device=args.device,
        cache_dir=args.cache_dir,
        en_locale=args.en_locale,
        workers=args.workers,
    )
    index.save(args.save)
    print(f"saved {len(index.spans)} spans -> {args.save}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    queries = _queries(args)
    if not queries:
        print("pass --query and/or --queries", file=sys.stderr)
        return 1
    if args.load:
        index = PhoneticIndex.load(args.load)
    else:
        if not args.inventory or not args.lang:
            print("pass --load, or --inventory with --lang", file=sys.stderr)
            return 1
        spans = read_span_file(args.inventory)
        print(f"phonemizing {len(spans)} spans ({args.lang})", flush=True)
        index = PhoneticIndex.build(
            spans,
            args.lang,
            tone_weight=args.tone_weight,
            device=args.device,
            cache_dir=args.cache_dir,
            en_locale=args.en_locale,
            workers=args.workers,
        )
    hits = index.search_many(
        queries,
        topk=args.topk,
        prefilter=args.prefilter,
        sim_min=args.sim_min,
        workers=args.workers,
        block_size=args.block_size,
        device=args.device,
        cache_dir=args.cache_dir,
        en_locale=args.en_locale,
    )
    rows = [{"query": q, "neighbors": h} for q, h in zip(queries, hits)]
    for row in rows:
        _print_hits(row["query"], row["neighbors"])
    sys.stdout.flush()
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False))
                fh.write("\n")
        print(f"wrote {len(rows)} rows -> {path}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="simphone",
        description="Find phonetically similar words or spans with a feature-weighted edit distance.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    n = sub.add_parser("norm", help="expand spans to their spoken form")
    n.add_argument("--lang", required=True, choices=["vi", "en"])
    n.add_argument("--text", action="append", default=None, help="repeatable span")
    n.add_argument("--queries", default=None, help="file of spans, one per line")
    n.add_argument(
        "--workers",
        type=int,
        default=0,
        help="sea-g2p Rayon threads for Vietnamese (0 = os.cpu_count()). Ignored for English.",
    )
    n.set_defaults(func=cmd_norm)

    b = sub.add_parser("build", help="phonemize an inventory and save the index")
    b.add_argument("--lang", required=True, choices=["vi", "en"])
    b.add_argument("--inventory", required=True, help="one span per line")
    b.add_argument("--save", required=True, help="directory to write")
    b.add_argument("--tone-weight", type=float, default=0.5)
    b.add_argument("--device", default="cpu", help="English out-of-dictionary G2P device")
    b.add_argument(
        "--workers",
        type=int,
        default=0,
        help="CPU workers for Vietnamese segmentation (0 = os.cpu_count()). "
             "Also sets sea-g2p's Rayon thread count.",
    )
    b.add_argument("--cache-dir", default=None, help="Charsiu model cache (English)")
    b.add_argument("--en-locale", default="eng-us")
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("search", help="search one or more queries against an inventory")
    s.add_argument("--load", default=None, help="index directory from build")
    s.add_argument("--lang", choices=["vi", "en"], help="required with --inventory")
    s.add_argument("--inventory", default=None, help="build in memory instead of --load")
    s.add_argument("--query", action="append", default=None, help="repeatable span")
    s.add_argument("--queries", default=None, help="file of query spans, one per line")
    s.add_argument("--topk", type=int, default=50)
    s.add_argument("--prefilter", type=int, default=300)
    s.add_argument("--sim-min", type=float, default=0.8)
    s.add_argument("--workers", type=int, default=0, help="0 = os.cpu_count()")
    s.add_argument("--block-size", type=int, default=512)
    s.add_argument("--tone-weight", type=float, default=0.5)
    s.add_argument("--device", default="cpu")
    s.add_argument("--cache-dir", default=None)
    s.add_argument("--en-locale", default="eng-us")
    s.add_argument("--out", default=None, help="JSONL of query rows")
    s.set_defaults(func=cmd_search)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
