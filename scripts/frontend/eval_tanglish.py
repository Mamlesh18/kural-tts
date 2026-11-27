"""Measure Tanglish -> Tamil-script word accuracy against reviewed references.

Usage:
    uv run python scripts/frontend/eval_tanglish.py [--sentences eval/sentences/feasibility_v1.tsv]

Compares the frontend's language-ID + transliteration output (before number verbalization) with
the `reference_ta_script` column, word by word, ignoring punctuation and hyphens. The Phase 5 exit
target is >= 90% on a separate 500-sentence native-reviewed Tanglish set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))

from common import DEFAULT_SENTENCES, edit_distance, load_sentences, normalize_for_cer

from kural_tts.text.frontend import analyze


def _words(text: str) -> list[str]:
    return normalize_for_cer(text.replace("-", "")).split()


def frontend_script(text: str) -> str:
    tokens = analyze(text)
    return "".join((" " if t.space_before and not t.attached else "") + t.text for t in tokens)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sentences", type=Path, default=DEFAULT_SENTENCES)
    p.add_argument("--show-errors", action="store_true")
    args = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    rows = [r for r in load_sentences(args.sentences) if r["category"] == "tanglish"]
    total = errors = 0
    for r in rows:
        ref, hyp = _words(r["reference_ta_script"]), _words(frontend_script(r["text"]))
        dist = edit_distance(ref, hyp)
        total += len(ref)
        errors += dist
        if dist and args.show_errors:
            print(f"{r['id']}\n  ref: {' '.join(ref)}\n  hyp: {' '.join(hyp)}")
    acc = 1 - errors / max(total, 1)
    print(f"Tanglish word accuracy: {acc:.1%} ({total - errors}/{total} words, {len(rows)} rows)")


if __name__ == "__main__":
    main()
