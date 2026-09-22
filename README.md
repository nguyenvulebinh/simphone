# simphone

Find words or spans that sound alike. Each span is turned into phones, then ranked with a feature-weighted edit distance (panphon articulatory cost, plus a penalty when tones differ).

Vietnamese phones come from [sea-g2p](https://github.com/pnnbao97/sea-g2p). English phones come from CharsiuG2P.

## Vietnamese reading

Vietnamese build and search run sea-g2p in two steps, both with `punc_norm=False` (no forced trailing period):

1. `Normalizer` expands the spoken form: numbers, dates, units, and abbreviations. `21` becomes `hai mươi mốt`, `km` becomes `ki lô mét`, `COVID-19` becomes `covid mười chín`.
2. `G2P` turns that spoken form into phones. `21` and `hai mươi mốt` therefore share `hˈaːj mˈyəj mˈoɜt̪`.

sea-g2p wraps an English run in `<en>...</en>` (`CPU` becomes `<en>c p u</en>`). `normalize` drops those markers and keeps the words. Search still passes the marked text to G2P, so the English run is read as English and the markers never become phones.

## English reading

English build and search run two steps:

1. [WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing) English text normalization expands the spoken form (cardinals, ordinals, decimals, dates, times, measures, money). `21` becomes `twenty one`. A number plus a unit expands (`5 km` becomes `five kilometers`). A bare `km` stays `km`.
2. CharsiuG2P turns each spoken word into phones. `21` and `twenty one` therefore share those phones.

## Install

`pip install simphone` installs Vietnamese only. Vietnamese search does not need a GPU.

English is a separate extra. Install PyTorch yourself first, so a CUDA build already in the environment is kept. This package will not install torch.

```bash
# CPU. Skip this if torch is already installed.
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install "simphone[en]"
```

The English extra needs Linux x86_64. It uses WeTextProcessing, which depends on pynini, and pynini publishes wheels for that platform only.

From a checkout of this repo:

```bash
pip install -e .
pip install -e ".[en]"   # after torch, Linux x86_64
```

Calling English without the extra raises an error that tells you to install `simphone[en]`. English out-of-dictionary words use Charsiu's byT5 model; `--device cuda:0` speeds that step only.

## Python

```python
from simphone import PhoneticIndex

index = PhoneticIndex.build(
    ["mắc", "mác", "mức", "việt nam"],
    lang="vi",
    tone_weight=0.5,
)
index.save("vi.index")

index = PhoneticIndex.load("vi.index")  # skips G2P and the cost matrix
hits = index.search("mắc", topk=50, prefilter=300, sim_min=0.8)
# [{"text": "...", "sim": ..., "n_units": 1, "dlen": 0}, ...]

batch = index.search_many(
    ["mắc", "việt nam"],
    topk=50,
    workers=0,       # 0 = all CPUs
    block_size=512,
)
```

`sim` is `1 - distance / max(len_a, len_b)`. `dlen` is the difference in whitespace-separated unit counts.

Spoken form only, without phones or an index. One string per input, same order:

```python
from simphone import normalize

normalize(["21", "km"], lang="vi")
# ["hai mươi mốt", "ki lô mét"]

normalize(["21", "5 km"], lang="en")
# ["twenty one", "five kilometers"]
```

English normalization needs `simphone[en]` and does not load torch.

## Command line

Inventory and query files are one span per line. Blank lines and `#` comments are ignored.

```bash
simphone norm --lang vi --text "21" --text "km"
simphone norm --lang en --text "21" --text "5 km"
simphone norm --lang vi --queries spans.txt

simphone build --lang vi --inventory spans.txt --save vi.index --tone-weight 0.5

simphone search --load vi.index --query "mắc" --topk 50 --sim-min 0.8 --workers 0

simphone search --load vi.index --queries queries.txt --workers 0 --block-size 512 --out hits.jsonl
```

`--inventory` on `search` builds in memory when you have no saved index. `--tone-weight` is fixed at build time and stored in the index.

## Large lists

Build the inventory once and `save` it. A later `load` does not phonemize those spans and does not rebuild the panphon matrix. A new query is phonemized on its own, including the Vietnamese or English normalizer above. If that query uses a phone the inventory never saw, only that phone's cost row is added.

Searching many queries one call at a time repeats a full scan of the inventory and leaves cores idle. `search_many` / `--queries` is the batch path:

- Several queries (at least as many as workers): the queries are split across processes. Each process keeps the inventory in shared memory and runs rapidfuzz on blocks of 512 queries, then reranks the top `--prefilter` with the feature-plus-tone distance.
- One query, or fewer queries than workers: the inventory is split instead, so that query still uses more than one core.

Passing the inventory back in as the query list returns a neighbor list for every span. A span is not returned as its own neighbor.

## Tones

panphon ignores tone. Vietnamese sea-g2p writes one syllable per space, with stress and a tone mark inside the syllable (`mà` → `mˌaː2`, `má` → `mˈaːɜ`). Those marks are peeled off and attached to the vowel before the distance is computed. The six tones on `ma mà mả mã má mạ` are: no mark, `2`, `4`, `5`, `ɜ` (sắc), `6`. English Charsiu tones stay Chao letters and use the same penalty. `--tone-weight` (default `0.5`) is added when two phones differ only in tone, or in tone as well as in the base phone.
