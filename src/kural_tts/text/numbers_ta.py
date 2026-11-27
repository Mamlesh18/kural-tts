"""Tamil number verbalization (Indian system: ஆயிரம், லட்சம், கோடி).

Forms follow standard written Tamil, which is also what a Chennai speaker reads aloud. Combining
forms (இருபத்து, நூற்று, ஆயிரத்து, லட்சத்து) are used when a smaller part follows. A following
unit digit is merged by vowel sandhi (இருபத்தொன்று, நூற்றொன்று). Larger parts stay space-separated.
"""

from __future__ import annotations

from kural_tts.text.tamil_script import attach_vowel

UNITS = ["", "ஒன்று", "இரண்டு", "மூன்று", "நான்கு", "ஐந்து", "ஆறு", "ஏழு", "எட்டு", "ஒன்பது"]
TEENS = [
    "பத்து",
    "பதினொன்று",
    "பன்னிரண்டு",
    "பதின்மூன்று",
    "பதினான்கு",
    "பதினைந்து",
    "பதினாறு",
    "பதினேழு",
    "பதினெட்டு",
    "பத்தொன்பது",
]
TENS = {2: "இருபது", 3: "முப்பது", 4: "நாற்பது", 5: "ஐம்பது",
        6: "அறுபது", 7: "எழுபது", 8: "எண்பது", 9: "தொண்ணூறு"}  # fmt: skip
TENS_COMB = {2: "இருபத்து", 3: "முப்பத்து", 4: "நாற்பத்து", 5: "ஐம்பத்து",
             6: "அறுபத்து", 7: "எழுபத்து", 8: "எண்பத்து", 9: "தொண்ணூற்று"}  # fmt: skip
HUNDREDS = {1: "நூறு", 2: "இருநூறு", 3: "முந்நூறு", 4: "நானூறு", 5: "ஐநூறு",
            6: "அறுநூறு", 7: "எழுநூறு", 8: "எண்ணூறு", 9: "தொள்ளாயிரம்"}  # fmt: skip
HUNDREDS_COMB = {1: "நூற்று", 2: "இருநூற்று", 3: "முந்நூற்று", 4: "நானூற்று", 5: "ஐநூற்று",
                 6: "அறுநூற்று", 7: "எழுநூற்று", 8: "எண்ணூற்று", 9: "தொள்ளாயிரத்து"}  # fmt: skip

DIGITS = ["பூஜ்ஜியம்", *UNITS[1:]]
ZERO = DIGITS[0]
POINT = "புள்ளி"


def _join(left: str, right: str) -> str:
    """Join a combining form with what follows; merge only single unit words by sandhi."""
    if not left:
        return right
    if not right:
        return left
    if right in UNITS[1:] and left.endswith("ு"):
        return attach_vowel(left, right) if right[0] in "அஆஇஈஉஊஎஏஐஒஓஔ" else f"{left} {right}"
    return f"{left} {right}"


def _below_100(n: int) -> str:
    if n < 10:
        return UNITS[n]
    if n < 20:
        return TEENS[n - 10]
    tens, unit = divmod(n, 10)
    return TENS[tens] if unit == 0 else _join(TENS_COMB[tens], UNITS[unit])


def _below_1000(n: int) -> str:
    h, rest = divmod(n, 100)
    if h == 0:
        return _below_100(rest)
    return HUNDREDS[h] if rest == 0 else _join(HUNDREDS_COMB[h], _below_100(rest))


def _thousands(q: int, combining: bool) -> str:
    """q in 1..99 thousands, e.g. 2 -> இரண்டாயிரம், 21 -> இருபத்தோராயிரம்."""
    if q == 1:
        word = "ஆயிரம்"
    else:
        w = _below_100(q)
        if w.endswith("ொன்று"):  # 11, 21, 31 ... -> ...ோராயிரம்
            word = w[: -len("ொன்று")] + "ோராயிரம்"
        elif w.endswith("மூன்று"):  # 3, 23 ... -> ...மூவாயிரம்
            word = w[: -len("மூன்று")] + "மூவாயிரம்"
        else:
            word = attach_vowel(w, "ஆயிரம்")
    return word[: -len("ம்")] + "த்து" if combining else word


def int_to_ta(n: int) -> str:
    """Cardinal number in Tamil words. ``int_to_ta(2499)`` -> இரண்டாயிரத்து நானூற்று தொண்ணூற்றொன்பது."""
    if n < 0:
        return f"மைனஸ் {int_to_ta(-n)}"
    if n == 0:
        return ZERO
    parts: list[str] = []
    crore, n = divmod(n, 10**7)
    lakh, n = divmod(n, 10**5)
    thousand, n = divmod(n, 1000)
    rest = n
    if crore:
        head = "ஒரு" if crore == 1 else int_to_ta(crore)
        parts.append(f"{head} {'கோடியே' if (lakh or thousand or rest) else 'கோடி'}")
    if lakh:
        head = "ஒரு" if lakh == 1 else _below_100(lakh)
        parts.append(f"{head} {'லட்சத்து' if (thousand or rest) else 'லட்சம்'}")
    if thousand:
        parts.append(_thousands(thousand, combining=bool(rest)))
    if rest:
        tail = _below_1000(rest)
        if parts:
            return " ".join([*parts[:-1], _join(parts[-1], tail)])
        return tail
    return " ".join(parts)


def digits_to_ta(digits: str) -> str:
    """Read a digit string one digit at a time (phone numbers, PINs)."""
    return " ".join(DIGITS[int(d)] for d in digits if d.isdigit())


def decimal_to_ta(text: str) -> str:
    """'8.5' -> எட்டு புள்ளி ஐந்து (fraction read digit by digit)."""
    whole, _, frac = text.replace(",", "").partition(".")
    words = int_to_ta(int(whole or "0"))
    return f"{words} {POINT} {digits_to_ta(frac)}" if frac else words


def attach_suffix(number_words: str, suffix: str) -> str:
    """Attach a Tamil case/ordinal suffix to the last word: (ஐந்து, ஆம்) -> ஐந்தாம்."""
    head, _, last = number_words.rpartition(" ")
    joined = attach_vowel(last, suffix)
    return f"{head} {joined}" if head else joined
