import csv
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

SENT_DIR = Path(__file__).resolve().parents[1] / "eval" / "sentences"
COLUMNS = ["id", "category", "phenomenon", "text", "reference_ta_script"]
CATEGORIES = {"en", "ta", "cs", "tanglish"}
TAMIL = range(0x0B80, 0x0C00)


def _has_tamil(s: str) -> bool:
    return any(ord(c) in TAMIL for c in s)


def _has_latin(s: str) -> bool:
    return any("a" <= c.lower() <= "z" for c in s)


def load(name: str) -> list[dict]:
    with (SENT_DIR / name).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        assert reader.fieldnames == COLUMNS
        return list(reader)


@pytest.fixture(scope="module")
def feasibility():
    return load("feasibility_v1.tsv")


def test_ids_unique_and_prefixed(feasibility):
    ids = [r["id"] for r in feasibility]
    assert len(ids) == len(set(ids))
    for r in feasibility:
        prefix = "tg" if r["category"] == "tanglish" else r["category"]
        assert r["id"].startswith(prefix + "_")


def test_balanced_categories(feasibility):
    counts = Counter(r["category"] for r in feasibility)
    assert set(counts) == CATEGORIES
    assert all(n == 20 for n in counts.values())


def test_text_is_nfc_and_nonempty(feasibility):
    for r in feasibility:
        assert r["text"].strip()
        assert unicodedata.normalize("NFC", r["text"]) == r["text"], r["id"]


def test_script_matches_category(feasibility):
    for r in feasibility:
        text, cat = r["text"], r["category"]
        if cat == "en":
            assert not _has_tamil(text), r["id"]
        elif cat == "ta":
            assert _has_tamil(text) and not _has_latin(text), r["id"]
        elif cat == "cs":
            assert _has_tamil(text), r["id"]
        elif cat == "tanglish":
            assert not _has_tamil(text), r["id"]
            assert _has_tamil(r["reference_ta_script"]), r["id"]
        if cat != "tanglish":
            assert r["reference_ta_script"] == "", r["id"]
