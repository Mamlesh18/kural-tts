"""Tamil script constants and sandhi helpers shared by the frontend modules."""

from __future__ import annotations

TAMIL_BLOCK = (0x0B80, 0x0BFF)

PULLI = "்"  # ் virama

# independent vowel -> dependent vowel sign ("" for the inherent அ)
VOWEL_SIGNS: dict[str, str] = {
    "அ": "",
    "ஆ": "ா",
    "இ": "ி",
    "ஈ": "ீ",
    "உ": "ு",
    "ஊ": "ூ",
    "எ": "ெ",
    "ஏ": "ே",
    "ஐ": "ை",
    "ஒ": "ொ",
    "ஓ": "ோ",
    "ஔ": "ௌ",
}
INDEPENDENT_VOWELS = frozenset(VOWEL_SIGNS)
DEPENDENT_SIGNS = frozenset(s for s in VOWEL_SIGNS.values() if s)

# Glide inserted between a vowel-final word and a vowel-initial suffix.
_FRONT_SIGNS = frozenset("ிீெேை")  # i, ii, e, ee, ai -> ய
TAMIL_DIGITS = {chr(0x0BE6 + i): str(i) for i in range(10)}  # ௦..௯


def is_tamil_char(ch: str) -> bool:
    return TAMIL_BLOCK[0] <= ord(ch) <= TAMIL_BLOCK[1]


def has_tamil(text: str) -> bool:
    return any(is_tamil_char(c) for c in text)


def attach_vowel(left: str, right: str) -> str:
    """Join ``left`` + vowel-initial ``right`` with Tamil vowel sandhi.

    - ...u + V   -> drop u, consonant takes V's sign   (இருபத்து + ஒன்று -> இருபத்தொன்று)
    - ...C் + V  -> drop pulli, consonant takes V's sign (ஆயிரம் + ஆம் -> ஆயிரமாம்)
    - ...(i/e/ai) + V -> insert ய glide                 (கோடி + ஆம் -> கோடியாம்)
    - other vowel endings + V -> insert வ glide         (பலா + ஆம் -> பலாவாம்)

    If ``right`` does not start with an independent vowel, the strings are concatenated as-is.
    """
    if not left or not right or right[0] not in INDEPENDENT_VOWELS:
        return left + right
    sign, rest = VOWEL_SIGNS[right[0]], right[1:]
    last = left[-1]
    if last == "ு":
        return left[:-1] + sign + rest
    if last == PULLI:
        return left[:-1] + sign + rest
    if last in _FRONT_SIGNS:
        return left + "ய" + sign + rest
    if last in DEPENDENT_SIGNS:
        return left + "வ" + sign + rest
    # bare consonant with inherent அ (e.g. "வேற")
    return left + ("ய" if last in "யரலவழள" else "வ") + sign + rest
