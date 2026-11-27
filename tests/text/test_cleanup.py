import unicodedata

from kural_tts.text.cleanup import clean


def test_nfc_and_invisible_chars():
    decomposed = unicodedata.normalize("NFD", "கொ")  # க + ெ + ா
    assert clean(decomposed) == "கொ"
    assert clean("நான்‌கு") == "நான்கு"
    assert clean("﻿hello​ world") == "hello world"


def test_tamil_digits_and_punctuation():
    assert clean("௨௫ அத்தியாயங்கள்") == "25 அத்தியாயங்கள்"
    assert clean("“Sorry,” she said…") == '"Sorry," she said...'
    assert clean("it’s") == "it's"
    assert clean("Chennai—Madurai") == "Chennai - Madurai"


def test_sri_variant_and_spaces():
    assert clean("ஶ்ரீ கிருஷ்ணா") == "ஸ்ரீ கிருஷ்ணா"
    assert clean("  a \n\t b  ") == "a b"
