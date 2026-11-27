import csv
from pathlib import Path

import pytest

from kural_tts.text.g2p import format_phones, phonemize, text_to_ids
from kural_tts.text.symbols import LANG_TO_ID, SPACE, SYMBOL_TO_ID


def syms(text, **kw):
    return [s for s, _ in phonemize(text, **kw)]


def test_tamil_sentence():
    assert format_phones(phonemize("நான் வரேன்.")) == "[ta] n̪ aː n | ʋ a ɾ eː n ."


def test_code_switched_language_tags():
    phones = phonemize("நாளைக்கு meeting இருக்கு")
    langs = [lang for s, lang in phones if s != SPACE]
    assert langs[0] == "ta" and "en" in langs and langs[-1] == "ta"
    assert format_phones(phones).startswith("[ta] n̪ aː ɭ ai k k ɯ | [en] m ˈ i t ɪ ŋ | [ta]")


def test_glued_suffix_has_no_word_boundary():
    phones = phonemize("நான் office-ல இருக்கேன்")
    text = format_phones(phones)
    assert "f ɪ s [ta] l a |" in text  # officeல spoken as one word


def test_numbers_and_tanglish_are_phonemized():
    assert "|" in format_phones(phonemize("naalaikku 10:30 ku varen"))
    assert all(s in SYMBOL_TO_ID for s in syms("₹1,250 pay pannunga, ok?"))


def test_punctuation_mapping_and_cleanup():
    assert syms("Hi; there...")[-1] == "."
    assert syms("Really?!")[-1] == "!"
    assert SPACE not in syms("Hi , there")[2:3]  # no space before the comma
    assert '"' not in syms('"Sorry," she said')


def test_text_to_ids_lengths_match():
    ids, langs = text_to_ids("Your OTP 5 நிமிடம் valid.")
    assert len(ids) == len(langs)
    assert set(langs) == {LANG_TO_ID["en"], LANG_TO_ID["ta"]}


SENTENCES = Path(__file__).resolve().parents[2] / "eval" / "sentences" / "feasibility_v1.tsv"


def _rows():
    with SENTENCES.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


@pytest.mark.parametrize("row", _rows(), ids=lambda r: r["id"])
def test_feasibility_sentences_encode(row):
    ids, langs = text_to_ids(row["text"])
    assert len(ids) > 4 and len(ids) == len(langs)
