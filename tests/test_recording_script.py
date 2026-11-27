import csv

from recording import templates
from recording.make_script import build_script, parse_mix, read_pool, units, write_prompts


def test_templates_are_deterministic_unique_and_categorized():
    a = templates.generate(20, seed=1)
    assert a == templates.generate(20, seed=1)
    assert len({r["text"] for r in a}) == len(a) == 60
    assert {r["category"] for r in a} == {"ta", "en", "cs"}
    assert all("{" not in r["text"] for r in a)


def test_indian_amount_grouping():
    import random

    rng = random.Random(0)
    for _ in range(200):
        s = templates._amount(rng)
        groups = s.split(",")
        assert all(len(g) == 2 for g in groups[1:-1]) and len(groups[-1]) in (2, 3)


def test_units_include_cross_language_pairs():
    pairs, n = units("நாளைக்கு meeting இருக்கு")
    assert n > 10
    assert any(la != lb for _, la, _, lb in pairs)


def test_build_script_respects_budget_and_mix():
    pool = templates.generate(80, seed=3)
    chosen, stats = build_script(pool, hours=0.05, mix=parse_mix("ta=1,en=1,cs=1"))
    per_cat = stats["hours_by_category"]
    assert all(0 < h <= 0.05 / 3 + 0.01 for h in per_cat.values())
    assert len({r["text"] for r in chosen}) == len(chosen)


def test_greedy_prefers_new_coverage():
    pool = [
        {"id": "a", "category": "en", "text": "Hello there, how are you doing today?"},
        {"id": "b", "category": "en", "text": "Hello there; how are you doing today!"},
        {"id": "c", "category": "en", "text": "Quick brown foxes jump over lazy dogs."},
    ]
    chosen, stats = build_script(pool, hours=1, mix={"en": 1.0})
    assert stats["dropped"].get("duplicate") == 1
    assert {r["id"] for r in chosen} == {"a", "c"}


def test_write_and_read_prompts_roundtrip(tmp_path):
    pool = templates.generate(5, seed=2)
    chosen, _ = build_script(pool, hours=1, mix=parse_mix("ta=1,en=1,cs=1"))
    out = tmp_path / "prompts.tsv"
    write_prompts(chosen, out)
    with out.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    assert [r["id"] for r in rows] == [r["id"] for r in chosen]
    assert all(not any(ch.isdigit() for ch in r["say"]) for r in rows)  # spoken form
    txt = tmp_path / "extra_en.txt"
    txt.write_text("One sentence.\nAnother one.\n", encoding="utf-8")
    assert [r["category"] for r in read_pool(txt)] == ["en", "en"]
