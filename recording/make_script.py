"""Select recording prompts that maximize phoneme-pair coverage, within per-category hour quotas.

    python recording/make_script.py --pool pool_ta.tsv pool_en.tsv pool_cs.tsv \\
        --hours 15 --mix ta=0.40,en=0.35,cs=0.25 --templates 150 -o recording/prompts.tsv

Pool files are TSVs with columns ``id, category, text`` (category: ta | en | cs), or plain .txt
files with one sentence per line (the category is taken from the file name: *_ta.txt, *_en.txt,
*_cs.txt). Selection is greedy: each step takes the sentence that adds the most uncovered
units per estimated second. Units are phoneme pairs **with their language tags**, so pairs that
cross a Tamil<->English switch count as their own units. That keeps the code-switch boundaries
well covered.

The output prompts.tsv also has a ``say`` column: the fully spoken form from the Kural frontend
(numbers, ₹, dates spelled out). The speaker reads ``say``, so the audio matches what the
model is trained to produce from ``text``.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import itertools
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kural_tts.text.frontend import normalize_text
from kural_tts.text.g2p import phonemize
from kural_tts.text.symbols import PUNCTUATION, SPACE
from recording import templates
from training.data_pipeline.build import eval_texts, norm_text

PHONES_PER_SECOND = 12.0  # rough speaking rate used to estimate durations
CATEGORIES = ("ta", "en", "cs")


def read_pool(path: Path) -> list[dict]:
    if path.suffix == ".txt":
        category = path.stem.rsplit("_", 1)[-1]
        if category not in CATEGORIES:
            raise SystemExit(f"{path}: name must end in _ta/_en/_cs to infer the category")
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return [{"id": f"{path.stem}_{i:05d}", "category": category, "text": t}
                for i, t in enumerate(lines, 1)]  # fmt: skip
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def units(text: str) -> tuple[set[tuple], int]:
    """Language-tagged phoneme bigrams (word boundaries included) and the phone count."""
    phones = [(p, lang) for p, lang in phonemize(text) if p not in PUNCTUATION]
    seq = [("#", phones[0][1]) if phones else ("#", "en"), *phones, ("#", "en")]
    pairs = {(a, la, b, lb) for (a, la), (b, lb) in itertools.pairwise(seq)}
    n = sum(1 for p, _ in phones if p != SPACE)
    return pairs, n


def select(candidates: list[dict], seconds_budget: float, covered: Counter) -> list[dict]:
    """Lazy greedy selection maximizing new units per estimated second."""
    heap = [(-math.inf, i) for i in range(len(candidates))]  # inf: score lazily on first pop
    heapq.heapify(heap)
    chosen, used = [], 0.0
    while heap and used < seconds_budget:
        neg_gain, i = heapq.heappop(heap)
        c = candidates[i]
        gain = sum(1.0 / (1 + covered[u]) for u in c["units"]) / max(c["seconds"], 0.5)
        if heap and gain < -heap[0][0] - 1e-12:
            heapq.heappush(heap, (-gain, i))  # stale score: re-insert and retry
            continue
        chosen.append(c)
        used += c["seconds"]
        covered.update(c["units"])
    return chosen


def build_script(
    pool: list[dict], hours: float, mix: dict[str, float], min_sec: float = 1.0,
    max_sec: float = 12.0,
) -> tuple[list[dict], dict]:  # fmt: skip
    banned = eval_texts()
    seen: set[str] = set()
    by_cat: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
    dropped = Counter()
    for row in pool:
        key = norm_text(row["text"])
        if key in banned:
            dropped["eval_overlap"] += 1
            continue
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)
        u, n = units(row["text"])
        seconds = n / PHONES_PER_SECOND
        if not min_sec <= seconds <= max_sec:
            dropped["length"] += 1
            continue
        if row["category"] not in by_cat:
            dropped["category"] += 1
            continue
        by_cat[row["category"]].append({**row, "units": u, "seconds": seconds})

    covered: Counter = Counter()
    chosen: list[dict] = []
    for cat in CATEGORIES:  # cs last: it benefits from TA/EN coverage already in place
        chosen += select(by_cat[cat], hours * 3600 * mix.get(cat, 0.0), covered)
    stats = {
        "dropped": dict(dropped),
        "units_covered": len(covered),
        "hours_by_category": {
            c: round(sum(x["seconds"] for x in chosen if x["category"] == c) / 3600, 2)
            for c in CATEGORIES
        },
        "pool_by_category": {c: len(v) for c, v in by_cat.items()},
    }
    return chosen, stats


def write_prompts(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writerow(["id", "category", "text", "say", "est_seconds"])
        for r in rows:
            w.writerow([r["id"], r["category"], r["text"], normalize_text(r["text"]),
                        f"{r['seconds']:.1f}"])  # fmt: skip


def parse_mix(text: str) -> dict[str, float]:
    mix = {k: float(v) for k, v in (part.split("=") for part in text.split(","))}
    total = sum(mix.values())
    return {k: v / total for k, v in mix.items()}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--pool", nargs="*", type=Path, default=[])
    p.add_argument("--hours", type=float, default=15.0)
    p.add_argument("--mix", default="ta=0.40,en=0.35,cs=0.25")
    p.add_argument("--templates", type=int, default=150, help="Templated prompts per category.")
    p.add_argument("-o", "--out", type=Path, default=Path("recording/prompts.tsv"))
    args = p.parse_args(argv)

    pool = [row for path in args.pool for row in read_pool(path)]
    pool += templates.generate(args.templates)
    chosen, stats = build_script(pool, args.hours, parse_mix(args.mix))
    write_prompts(chosen, args.out)
    hours = sum(stats["hours_by_category"].values())
    print(f"{len(chosen)} prompts (~{hours:.1f} h) -> {args.out}")
    print(f"  per category (h): {stats['hours_by_category']}")
    print(f"  phoneme-pair units covered: {stats['units_covered']}")
    print(f"  dropped from pool: {stats['dropped']}")
    short = {c: h for c, h in stats["hours_by_category"].items()
             if h < args.hours * parse_mix(args.mix).get(c, 0) * 0.95}  # fmt: skip
    if short:
        print(f"  WARNING: pool too small for target hours in {sorted(short)}. Add sentences.")


if __name__ == "__main__":
    main()
