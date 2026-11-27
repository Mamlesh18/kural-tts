import pytest

from kural_tts.text.normalize_en import verbalize_en
from kural_tts.text.tokenize import tokenize


def first(text):
    return verbalize_en(tokenize(text), 0)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("₹2,499", "two thousand four hundred ninety-nine rupees"),
        ("₹1,50,000", "one lakh fifty thousand rupees"),
        ("₹1", "one rupee"),
        ("₹10.50", "ten rupees and fifty paise"),
        ("$150,000", "one hundred fifty thousand dollars"),
        ("Rs. 500", "five hundred rupees"),
        ("044-2345-6789", "zero four four, two three four five, six seven eight nine"),
        ("8.5%", "eight point five percent"),
        ("15th", "fifteenth"),
        ("2026", "two thousand twenty-six"),
        ("1999", "nineteen ninety-nine"),
        ("2499", "two thousand four hundred ninety-nine"),
        ("support@example.com", "support at example dot com"),
        ("SBI", "S B I"),
        ("NASA", "nasa"),
        ("hello", "hello"),
    ],
)
def test_verbalize_single(text, expected):
    assert first(text)[0] == expected


def test_time_with_and_without_meridiem():
    assert first("10:30 AM") == ("ten thirty", 0)
    assert first("9:00 PM") == ("nine", 0)
    assert first("9:00") == ("nine o'clock", 0)
    assert first("10:05") == ("ten oh five", 0)


def test_abbreviation_consumes_dot():
    assert first("Dr. Ramesh") == ("doctor", 1)
    assert first("No. 5") == ("number", 1)
    assert first("No.")[1] == 0  # sentence-final "No." is just the word
