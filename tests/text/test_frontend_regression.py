"""Run the whole feasibility set through the frontend as a smoke/regression check.

This checks invariants only (no exact outputs), so it never tunes the frontend to the eval set.
"""

import csv
import re
from pathlib import Path

import pytest

from kural_tts.text.frontend import normalize
from kural_tts.text.tamil_script import has_tamil

SENTENCES = Path(__file__).resolve().parents[2] / "eval" / "sentences" / "feasibility_v1.tsv"
_UNSPOKEN = re.compile(r"[0-9₹$%@௦-௯]")


def _rows():
    with SENTENCES.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


@pytest.mark.parametrize("row", _rows(), ids=lambda r: r["id"])
def test_feasibility_sentence_normalizes(row):
    spans = normalize(row["text"])
    assert spans, "no output"
    joined = " ".join(s.text for s in spans)
    assert not _UNSPOKEN.search(joined), f"unverbalized symbol in: {joined}"
    for s in spans:
        assert s.lang in ("ta", "en")
        if s.lang == "en":
            assert not has_tamil(s.text)
    if row["category"] in ("ta", "tanglish"):
        assert any(s.lang == "ta" for s in spans)
    if row["category"] == "en":
        assert all(s.lang == "en" for s in spans)
