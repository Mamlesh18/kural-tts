import pytest

from kural_tts.text.normalize_ta import verbalize_ta
from kural_tts.text.tokenize import tokenize


def first(text):
    return verbalize_ta(tokenize(text), 0)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("₹2,499", "இரண்டாயிரத்து நானூற்று தொண்ணூற்றொன்பது ரூபாய்"),
        ("₹10.50", "பத்து ரூபாய் ஐம்பது பைசா"),
        ("8.5%", "எட்டு புள்ளி ஐந்து சதவீதம்"),
        ("10:30", "பத்தரை"),
        ("9:30", "ஒன்பதரை"),
        ("10:45", "பத்து நாற்பத்தைந்து"),
        ("9:00", "ஒன்பது"),
        ("15th", "பதினைந்தாம்"),
        ("044-2345-6789", "பூஜ்ஜியம் நான்கு நான்கு, இரண்டு மூன்று நான்கு ஐந்து, ஆறு ஏழு எட்டு ஒன்பது"),
        ("25", "இருபத்தைந்து"),
        ("2026", "இரண்டாயிரத்து இருபத்தாறு"),
    ],
)
def test_verbalize_numbers(text, expected):
    assert first(text) == (expected, 0)


def test_attached_suffix_is_consumed():
    assert first("15ஆம்") == ("பதினைந்தாம்", 1)
    assert first("10க்கு") == ("பத்துக்கு", 1)
    assert first("5-ஆம்") == ("ஐந்தாம்", 1)


def test_abbreviation_with_dot():
    assert first("டாக். ரமேஷ்") == ("டாக்டர்", 1)
    assert first("டாக்") == ("டாக்", 0)
