import pytest

from kural_tts.text.numbers_en import (
    decimal_to_en,
    digits_to_en,
    grouping_system,
    int_to_en,
    ordinal_en,
    year_en,
)


@pytest.mark.parametrize(
    "n,expected",
    [
        (0, "zero"),
        (7, "seven"),
        (15, "fifteen"),
        (26, "twenty-six"),
        (100, "one hundred"),
        (2499, "two thousand four hundred ninety-nine"),
        (150000, "one lakh fifty thousand"),
        (2500000, "twenty-five lakh"),
        (10000000, "one crore"),
        (123456789, "twelve crore thirty-four lakh fifty-six thousand seven hundred eighty-nine"),
    ],
)
def test_int_to_en_indian(n, expected):
    assert int_to_en(n) == expected


def test_int_to_en_western():
    assert int_to_en(150000, "western") == "one hundred fifty thousand"
    assert int_to_en(2_000_000, "western") == "two million"


@pytest.mark.parametrize(
    "n,expected",
    [(1, "first"), (2, "second"), (3, "third"), (5, "fifth"), (12, "twelfth"),
     (15, "fifteenth"), (20, "twentieth"), (21, "twenty-first"), (100, "one hundredth")],
)  # fmt: skip
def test_ordinal_en(n, expected):
    assert ordinal_en(n) == expected


@pytest.mark.parametrize(
    "n,expected",
    [(1999, "nineteen ninety-nine"), (1905, "nineteen oh five"), (1900, "nineteen hundred"),
     (2026, "two thousand twenty-six"), (2000, "two thousand")],
)  # fmt: skip
def test_year_en(n, expected):
    assert year_en(n) == expected


def test_digits_decimal_grouping():
    assert digits_to_en("044") == "zero four four"
    assert decimal_to_en("8.5") == "eight point five"
    assert grouping_system("1,50,000") == "indian"
    assert grouping_system("150,000") == "western"
    assert grouping_system("1,500,000") == "western"
    assert grouping_system("2,499") == "indian"
    assert grouping_system("150000") == "indian"
