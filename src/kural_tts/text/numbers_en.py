"""English number verbalization with Indian (lakh/crore) and Western grouping."""

from __future__ import annotations

from typing import Literal

System = Literal["indian", "western"]

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen"]  # fmt: skip
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

_ORDINAL_EXCEPTIONS = {
    "one": "first",
    "two": "second",
    "three": "third",
    "five": "fifth",
    "eight": "eighth",
    "nine": "ninth",
    "twelve": "twelfth",
}


def _below_100(n: int) -> str:
    if n < 20:
        return ONES[n]
    tens, unit = divmod(n, 10)
    return TENS[tens] if unit == 0 else f"{TENS[tens]}-{ONES[unit]}"


def _below_1000(n: int) -> str:
    h, rest = divmod(n, 100)
    if h == 0:
        return _below_100(rest)
    head = f"{ONES[h]} hundred"
    return head if rest == 0 else f"{head} {_below_100(rest)}"


def int_to_en(n: int, system: System = "indian") -> str:
    """Cardinal in words. Indian: 150000 -> 'one lakh fifty thousand'."""
    if n < 0:
        return f"minus {int_to_en(-n, system)}"
    if n == 0:
        return "zero"
    parts: list[str] = []
    if system == "indian":
        crore, n = divmod(n, 10**7)
        if crore:
            parts.append(f"{int_to_en(crore, system)} crore")
        lakh, n = divmod(n, 10**5)
        if lakh:
            parts.append(f"{_below_100(lakh)} lakh")
        thousand, n = divmod(n, 1000)
        if thousand:
            parts.append(f"{_below_100(thousand)} thousand")
    else:
        for value, name in ((10**12, "trillion"), (10**9, "billion"), (10**6, "million"),
                            (1000, "thousand")):  # fmt: skip
            q, n = divmod(n, value)
            if q:
                parts.append(f"{_below_1000(q)} {name}")
    if n:
        parts.append(_below_1000(n))
    return " ".join(parts)


def ordinal_en(n: int) -> str:
    """2 -> 'second', 21 -> 'twenty-first', 100 -> 'one hundredth'."""
    words = int_to_en(n)
    head, sep, last = words.rpartition(" ")
    pre, hyphen, unit = last.rpartition("-")
    if unit in _ORDINAL_EXCEPTIONS:
        unit = _ORDINAL_EXCEPTIONS[unit]
    elif unit.endswith("y"):
        unit = unit[:-1] + "ieth"
    else:
        unit += "th"
    return f"{head}{sep}{pre}{hyphen}{unit}"


def year_en(n: int) -> str:
    """Read years as spoken in Indian English: 1999 -> nineteen ninety-nine, 2026 -> two thousand
    twenty-six, 1905 -> nineteen oh five."""
    if 1100 <= n <= 1999:
        hi, lo = divmod(n, 100)
        if lo == 0:
            return f"{_below_100(hi)} hundred"
        return f"{_below_100(hi)} {'oh ' + ONES[lo] if lo < 10 else _below_100(lo)}"
    return int_to_en(n)


def digits_to_en(digits: str) -> str:
    return " ".join(ONES[int(d)] for d in digits if d.isdigit())


def decimal_to_en(text: str, system: System = "indian") -> str:
    whole, _, frac = text.replace(",", "").partition(".")
    words = int_to_en(int(whole or "0"), system)
    return f"{words} point {digits_to_en(frac)}" if frac else words


def grouping_system(raw: str) -> System:
    """'1,50,000' -> indian; '150,000' -> western; no commas -> indian."""
    groups = raw.split(".")[0].split(",")
    if len(groups) < 2:
        return "indian"
    # Western groups every 3 digits; Indian groups the part above thousands in pairs (1,50,000).
    # "2,499" is ambiguous but reads the same in both systems.
    if all(len(g) == 3 for g in groups[1:]) and (len(groups) > 2 or len(groups[0]) == 3):
        return "western"
    return "indian"
