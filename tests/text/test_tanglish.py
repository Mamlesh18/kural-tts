import pytest

from kural_tts.text.tamil_script import has_tamil
from kural_tts.text.tanglish import (
    load_lexicon,
    looks_tanglish,
    lookup,
    rule_transliterate,
    transliterate,
)


def test_lexicon_loads_and_is_tamil():
    lex = load_lexicon()
    assert len(lex) > 250
    assert all(has_tamil(e.tamil) for e in lex.values())
    assert all(k == k.lower() and k.isascii() for k in lex)


def test_lookup_variants_and_ambiguity():
    assert lookup("Romba").tamil == "ரொம்ப"
    assert lookup("rombha").tamil == "ரொம்ப"
    assert lookup("pal").ambiguous
    assert not lookup("irukku").ambiguous
    assert lookup("meeting") is None


@pytest.mark.parametrize(
    "word,expected",
    [
        ("naan", "நான்"),
        ("vali", "வலி"),
        ("mazhai", "மழை"),
        ("irukku", "இருக்கு"),
        ("enna", "என்ன"),
        ("romba", "ரொம்ப"),
        ("konjam", "கொஞ்சம்"),
        ("vandi", "வண்டி"),
        ("antha", "அந்த"),
        ("unga", "உங்க"),
    ],
)
def test_rule_transliterate(word, expected):
    assert rule_transliterate(word) == expected


def test_transliterate_prefers_lexicon():
    # Rules alone would give பண்றென் (short e). The lexicon has the correct long ே.
    assert transliterate("panren") == "பண்றேன்"


def test_looks_tanglish():
    assert looks_tanglish("pannittaanga")
    assert looks_tanglish("vazhakkam")
    assert not looks_tanglish("meeting")
    assert not looks_tanglish("the")
