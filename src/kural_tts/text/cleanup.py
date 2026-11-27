"""First frontend pass: Unicode cleanup that is safe for both Tamil and English."""

from __future__ import annotations

import re
import unicodedata

from kural_tts.text.tamil_script import TAMIL_DIGITS

# Zero-width and soft characters that only affect rendering, never pronunciation.
_INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿­"))

_CHAR_MAP = str.maketrans(
    {
        **TAMIL_DIGITS,  # ௧௨ -> 12
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "…": "...",
        " ": " ",
        "–": " - ",  # en dash
        "—": " - ",  # em dash
        "−": "-",  # minus sign
        "₨": "₹",  # ₨ rupee sign -> ₹
    }
)

_VARIANTS = (
    ("ஶ்ரீ", "ஸ்ரீ"),  # SHA-based Sri -> standard Sri
)

_SPACES = re.compile(r"[ \t\r\n]+")


def clean(text: str) -> str:
    """NFC, drop invisible characters, map Tamil digits/typographic punctuation, squeeze spaces."""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_INVISIBLE).translate(_CHAR_MAP)
    for src, dst in _VARIANTS:
        text = text.replace(src, dst)
    return _SPACES.sub(" ", text).strip()
