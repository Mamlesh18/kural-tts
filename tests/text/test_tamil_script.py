import pytest

from kural_tts.text.tamil_script import attach_vowel, has_tamil, is_tamil_char


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ("இருபத்து", "ஒன்று", "இருபத்தொன்று"),
        ("இருபத்து", "இரண்டு", "இருபத்திரண்டு"),
        ("இருபத்து", "ஐந்து", "இருபத்தைந்து"),
        ("இருபத்து", "ஆறு", "இருபத்தாறு"),
        ("இருபத்து", "ஏழு", "இருபத்தேழு"),
        ("இருபத்து", "எட்டு", "இருபத்தெட்டு"),
        ("தொண்ணூற்று", "ஒன்பது", "தொண்ணூற்றொன்பது"),
        ("ஐந்து", "ஆம்", "ஐந்தாம்"),
        ("ஆயிரம்", "ஆம்", "ஆயிரமாம்"),
        ("கோடி", "ஆம்", "கோடியாம்"),
        ("பத்து", "க்கு", "பத்துக்கு"),  # consonant-initial suffix: plain concatenation
    ],
)
def test_attach_vowel(left, right, expected):
    assert attach_vowel(left, right) == expected


def test_script_detection():
    assert is_tamil_char("த")
    assert not is_tamil_char("t")
    assert has_tamil("meeting மணிக்கு")
    assert not has_tamil("meeting at ten")
