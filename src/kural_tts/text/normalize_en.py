"""Verbalize English-context tokens (Indian English conventions)."""

from __future__ import annotations

from kural_tts.text.numbers_en import (
    decimal_to_en,
    digits_to_en,
    grouping_system,
    int_to_en,
    ordinal_en,
    year_en,
)
from kural_tts.text.tokenize import Token

ABBREVIATIONS = {
    "dr": "doctor",
    "mr": "mister",
    "mrs": "missus",
    "ms": "miss",
    "prof": "professor",
    "sr": "senior",
    "jr": "junior",
    "vs": "versus",
    "etc": "et cetera",
}

# Upper-case acronyms that are spoken as words rather than spelled out.
ACRONYM_WORDS = frozenset(
    {"NASA", "ISRO", "UNESCO", "UNICEF", "COVID", "NEET", "PIN", "PAN", "SIM", "RAM",
     "NATO", "SEBI", "BHIM", "GIF", "LASER", "RADAR", "AIDS"}
)  # fmt: skip

_CURRENCY = {
    "₹": ("rupee", "rupees", "paisa", "paise"),
    "Rs": ("rupee", "rupees", "paisa", "paise"),
    "$": ("dollar", "dollars", "cent", "cents"),
}
_MERIDIEM = frozenset({"AM", "PM", "am", "pm", "a.m", "p.m"})


def _next(tokens: list[Token], i: int, offset: int = 1) -> Token | None:
    j = i + offset
    return tokens[j] if j < len(tokens) else None


def _is_dot(tok: Token | None) -> bool:
    return tok is not None and tok.kind == "punct" and tok.text == "." and not tok.space_before


def split_currency(raw: str) -> tuple[str, str]:
    for sym in ("₹", "$", "Rs.", "Rs"):
        if raw.startswith(sym):
            return sym.rstrip("."), raw[len(sym) :].strip()
    return "₹", raw


def _currency(raw: str) -> str:
    sym, amount = split_currency(raw)
    one, many, sub_one, sub_many = _CURRENCY[sym]
    whole, _, frac = amount.partition(".")
    n = int(whole.replace(",", "") or "0")
    words = f"{int_to_en(n, grouping_system(whole))} {one if n == 1 else many}"
    sub = int((frac + "00")[:2]) if frac else 0
    if sub:
        words += f" and {int_to_en(sub)} {sub_one if sub == 1 else sub_many}"
    return words


def _time(raw: str, followed_by_meridiem: bool) -> str:
    h, m = (int(x) for x in raw.split(":"))
    hour = int_to_en(h)
    if m == 0:
        return hour if followed_by_meridiem else f"{hour} o'clock"
    return f"{hour} oh {int_to_en(m)}" if m < 10 else f"{hour} {int_to_en(m)}"


def _number(raw: str) -> str:
    if "." in raw:
        return decimal_to_en(raw, grouping_system(raw))
    if "," not in raw and len(raw) == 4 and 1100 <= int(raw) <= 2099:
        return year_en(int(raw))
    return int_to_en(int(raw.replace(",", "")), grouping_system(raw))


def _email(raw: str) -> str:
    out = []
    for ch in raw:
        out.append({"@": " at ", ".": " dot ", "_": " underscore ", "-": " dash "}.get(ch, ch))
    return " ".join("".join(out).split())


def verbalize_en(tokens: list[Token], i: int) -> tuple[str, int]:
    """Return (spoken text, number of following tokens consumed)."""
    tok = tokens[i]
    nxt = _next(tokens, i)
    kind, raw = tok.kind, tok.text

    if kind == "latin":
        low = raw.lower()
        if low in ABBREVIATIONS and _is_dot(nxt):
            word = ABBREVIATIONS[low]
            return (word.capitalize() if raw[0].isupper() else word), 1
        if low == "no" and _is_dot(nxt) and (after := _next(tokens, i, 2)) and after.is_numeric:
            return "number", 1
        if raw.isupper() and raw.isalpha() and len(raw) >= 2:
            return (raw.lower(), 0) if raw in ACRONYM_WORDS else (" ".join(raw), 0)
        return raw, 0
    if kind == "email":
        return _email(raw), 0
    if kind == "currency":
        return _currency(raw), 0
    if kind == "phone":
        prefix = "plus " if raw.startswith("+") else ""
        return prefix + ", ".join(digits_to_en(g) for g in raw.lstrip("+").split("-")), 0
    if kind == "time":
        return _time(raw, nxt is not None and nxt.text in _MERIDIEM), 0
    if kind == "percent":
        return f"{decimal_to_en(raw.rstrip('%'))} percent", 0
    if kind == "ordinal":
        return ordinal_en(int(raw[:-2])), 0
    if kind == "number":
        return _number(raw), 0
    return raw, 0
