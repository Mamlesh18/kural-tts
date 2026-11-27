"""Create a CSV for native speakers to check G2P output word by word.

Usage:
    uv run python scripts/frontend/make_g2p_review_sheet.py words.txt -o g2p_review.csv
    uv run python scripts/frontend/make_g2p_review_sheet.py --from-eval -o g2p_review.csv

The reviewer fills `correct` (y/n) and, if wrong, `corrected` with the right phonemes or a
plain-language note ("க here sounds like g"). Then run with --score to compute word accuracy
(Phase 5 exit: >= 98% on >= 500 Tamil words).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

from kural_tts.text.frontend import normalize
from kural_tts.text.g2p_en import g2p_en_word
from kural_tts.text.g2p_ta import g2p_ta_word

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "eval" / "sentences" / "feasibility_v1.tsv"
_WORD = re.compile(r"[஀-௿]+|[A-Za-z]+(?:['-][A-Za-z]+)*")


def words_from_eval() -> list[tuple[str, str]]:
    with EVAL.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    seen: dict[str, str] = {}
    for r in rows:
        for span in normalize(r["text"]):
            for w in _WORD.findall(span.text):
                seen.setdefault(w, span.lang)
    return list(seen.items())


def phones_for(word: str, lang: str) -> str:
    if lang == "ta":
        return " ".join(g2p_ta_word(word))
    return " ".join(p for p, _ in g2p_en_word(word))


def score(path: Path) -> None:
    with path.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["correct"].strip().lower() in ("y", "n")]
    for lang in ("ta", "en"):
        sub = [r for r in rows if r["lang"] == lang]
        if sub:
            ok = sum(r["correct"].strip().lower() == "y" for r in sub)
            print(f"{lang}: {ok}/{len(sub)} = {ok / len(sub):.1%} word accuracy")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("words", nargs="?", type=Path, help="UTF-8 file, one word per line")
    p.add_argument("--from-eval", action="store_true", help="Use words from the feasibility set")
    p.add_argument("-o", "--out", type=Path, default=Path("g2p_review.csv"))
    p.add_argument("--score", type=Path, help="Score a filled-in review sheet")
    args = p.parse_args()

    if args.score:
        score(args.score)
        return
    if args.from_eval:
        items = words_from_eval()
    elif args.words:
        lines = args.words.read_text(encoding="utf-8").split()
        items = [(w, "ta" if "஀" <= w[0] <= "௿" else "en") for w in lines]
    else:
        sys.exit("give a word file or --from-eval")

    with args.out.open("w", encoding="utf-8-sig", newline="") as f:  # BOM: opens in Excel
        w = csv.writer(f)
        w.writerow(["word", "lang", "phonemes", "correct", "corrected", "notes"])
        for word, lang in items:
            w.writerow([word, lang, phones_for(word, lang), "", "", ""])
    print(f"{len(items)} words -> {args.out}")


if __name__ == "__main__":
    main()
