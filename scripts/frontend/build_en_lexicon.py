"""Build the packaged English lexicon from CMUdict.

Usage:
    uv run python scripts/frontend/build_en_lexicon.py path/to/cmudict.dict

Source: https://github.com/cmusphinx/cmudict (BSD-2-Clause style license, see
src/kural_tts/data/lexicon/CMUDICT_LICENSE.txt). Keeps the first pronunciation of each word,
drops inline comments, and writes a gzip TSV: ``word<TAB>ARPAbet phones``.
"""

from __future__ import annotations

import gzip
import re
import sys
from pathlib import Path

OUT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "kural_tts"
    / "data"
    / "lexicon"
    / "cmudict.tsv.gz"
)
_WORD = re.compile(r"^[a-z][a-z'.-]*$")


def build(src: Path, out: Path = OUT) -> int:
    entries: dict[str, str] = {}
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        word, phones = line.split(" ", 1)
        if "(" in word or not _WORD.match(word):  # alternates like word(2), or odd entries
            continue
        entries.setdefault(word, phones.strip())
    body = "".join(f"{w}\t{p}\n" for w, p in sorted(entries.items()))
    # mtime=0 keeps the output byte-identical across rebuilds.
    with gzip.GzipFile(out, "wb", mtime=0) as f:
        f.write(body.encode("utf-8"))
    return len(entries)


if __name__ == "__main__":
    n = build(Path(sys.argv[1]))
    print(f"wrote {n} entries -> {OUT}")
