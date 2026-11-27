"""Verbalize Tamil-context tokens: numbers, currency, dates, times, abbreviations."""

from __future__ import annotations

from kural_tts.text.normalize_en import split_currency
from kural_tts.text.numbers_ta import attach_suffix, decimal_to_ta, digits_to_ta, int_to_ta
from kural_tts.text.tamil_script import attach_vowel
from kural_tts.text.tokenize import Token

# Abbreviation (without its dot) -> spoken form. Only expanded when followed by ".".
TA_ABBREVIATIONS = {
    "டாக்": "டாக்டர்",
    "திரு": "திரு",
    "திருமதி": "திருமதி",
    "செல்வி": "செல்வி",
}

_CURRENCY_TA = {"₹": ("ரூபாய்", "பைசா"), "Rs": ("ரூபாய்", "பைசா"), "$": ("டாலர்", "சென்ட்")}
PERCENT = "சதவீதம்"
ORDINAL_SUFFIX = "ஆம்"


def _currency(raw: str) -> str:
    sym, amount = split_currency(raw)
    unit, sub_unit = _CURRENCY_TA[sym]
    whole, _, frac = amount.partition(".")
    words = f"{int_to_ta(int(whole.replace(',', '') or '0'))} {unit}"
    sub = int((frac + "00")[:2]) if frac else 0
    return f"{words} {int_to_ta(sub)} {sub_unit}" if sub else words


HALF = "அரை"


def _time(raw: str) -> str:
    """'9:00' -> ஒன்பது, '10:30' -> பத்தரை (spoken half-hour), '10:45' -> பத்து நாற்பத்தைந்து."""
    h, m = (int(x) for x in raw.split(":"))
    if m == 0:
        return int_to_ta(h)
    if m == 30:
        return attach_vowel(int_to_ta(h), HALF)
    return f"{int_to_ta(h)} {int_to_ta(m)}"


def _spoken_number(tok: Token) -> str:
    kind, raw = tok.kind, tok.text
    if kind == "currency":
        return _currency(raw)
    if kind == "phone":
        return ", ".join(digits_to_ta(g) for g in raw.lstrip("+").split("-"))
    if kind == "time":
        return _time(raw)
    if kind == "percent":
        return f"{decimal_to_ta(raw.rstrip('%'))} {PERCENT}"
    if kind == "ordinal":
        return attach_suffix(int_to_ta(int(raw[:-2])), ORDINAL_SUFFIX)
    return decimal_to_ta(raw)


def verbalize_ta(tokens: list[Token], i: int) -> tuple[str, int]:
    """Return (spoken text, number of following tokens consumed)."""
    tok = tokens[i]
    nxt = tokens[i + 1] if i + 1 < len(tokens) else None

    if tok.kind == "ta":
        if (
            tok.text in TA_ABBREVIATIONS
            and nxt is not None
            and nxt.kind == "punct"
            and nxt.text == "."
            and not nxt.space_before
        ):
            return TA_ABBREVIATIONS[tok.text], 1
        return tok.text, 0

    if tok.is_numeric:
        words = _spoken_number(tok)
        if nxt is not None and nxt.attached and nxt.kind == "ta":
            return attach_suffix(words, nxt.text), 1  # 15ஆம் -> பதினைந்தாம்
        return words, 0

    return tok.text, 0
