"""Rule-based Tamil G2P (Chennai standard).

Tamil spelling is close to phonemic, but the plain stops க ச ட த ப have context-dependent
allophones that generic phonemizers get wrong. The rules below follow standard descriptions of
spoken Tamil and still need native validation (Phase 5 exit criterion: >= 98% word accuracy).

    position                   க    ச    ட    த    ப
    word-initial               k    s    ʈ    t̪    p     (Chennai: initial ச is [s])
    geminate (க்க ...)          k    tʃ   ʈ    t̪    p
    after a nasal (ங்க ...)     ɡ    dʒ   ɖ    d̪    b
    after ய ர ல வ ழ ள          ɡ    s    ɖ    d̪    b
    between vowels             ɣ    s    ɖ    ð    b

Other rules: ற்ற -> t r, ன்ற -> n d r, ஃப -> f, word-final consonant+உ -> ɯ (kutriyalukaram).
"""

from __future__ import annotations

from dataclasses import dataclass

from kural_tts.text.tamil_script import PULLI, VOWEL_SIGNS

CONSONANTS = frozenset("கஙசஞடணதநபமயரலவழளறனஜஷஸஹஶ")
STOPS = frozenset("கசடதப")
NASALS = frozenset("ஙஞணநமன")
SONORANTS = frozenset("யரலவழள")
AYTHAM = "ஃ"

BASE = {
    "க": "k", "ங": "ŋ", "ச": "tʃ", "ஞ": "ɲ", "ட": "ʈ", "ண": "ɳ", "த": "t̪", "ந": "n̪",
    "ப": "p", "ம": "m", "ய": "j", "ர": "ɾ", "ல": "l", "வ": "ʋ", "ழ": "ɻ", "ள": "ɭ",
    "ற": "r", "ன": "n", "ஜ": "dʒ", "ஷ": "ʃ", "ஸ": "s", "ஹ": "h", "ஶ": "ʃ",
}  # fmt: skip
VOICELESS = {"க": "k", "ச": "tʃ", "ட": "ʈ", "த": "t̪", "ப": "p"}
INITIAL = {"க": "k", "ச": "s", "ட": "ʈ", "த": "t̪", "ப": "p"}
AFTER_NASAL = {"க": "ɡ", "ச": "dʒ", "ட": "ɖ", "த": "d̪", "ப": "b"}
AFTER_SONORANT = {"க": "ɡ", "ச": "s", "ட": "ɖ", "த": "d̪", "ப": "b"}
INTERVOCALIC = {"க": "ɣ", "ச": "s", "ட": "ɖ", "த": "ð", "ப": "b"}

VOWELS = {
    "அ": "a", "ஆ": "aː", "இ": "i", "ஈ": "iː", "உ": "u", "ஊ": "uː",
    "எ": "e", "ஏ": "eː", "ஐ": "ai", "ஒ": "o", "ஓ": "oː", "ஔ": "au",
}  # fmt: skip
_SIGN_TO_VOWEL = {sign: vowel for vowel, sign in VOWEL_SIGNS.items() if sign}
REDUCED_U = "ɯ"


@dataclass(frozen=True)
class Unit:
    """One orthographic syllable unit. ``vowel=None`` means a dead consonant (with pulli)."""

    cons: str | None
    vowel: str | None


def parse(word: str) -> list[Unit]:
    units: list[Unit] = []
    i = 0
    while i < len(word):
        ch = word[i]
        nxt = word[i + 1] if i + 1 < len(word) else ""
        if ch in CONSONANTS:
            if nxt == PULLI:
                units.append(Unit(ch, None))
                i += 2
            elif nxt in _SIGN_TO_VOWEL:
                units.append(Unit(ch, _SIGN_TO_VOWEL[nxt]))
                i += 2
            else:
                units.append(Unit(ch, "அ"))
                i += 1
        elif ch in VOWELS:
            units.append(Unit(None, ch))
            i += 1
        elif ch == AYTHAM:
            units.append(Unit(AYTHAM, None))
            i += 1
        else:  # stray signs, Tamil symbols, non-Tamil characters
            i += 1
    return units


def _consonant(units: list[Unit], idx: int) -> list[str]:
    u = units[idx]
    c = u.cons
    prev = units[idx - 1] if idx > 0 else None
    nxt = units[idx + 1] if idx + 1 < len(units) else None
    prev_dead = prev is not None and prev.vowel is None

    if c == AYTHAM:
        return [] if nxt is not None and nxt.cons == "ப" else ["h"]
    if c in STOPS:
        if c == "ப" and prev is not None and prev.cons == AYTHAM:
            return ["f"]
        if u.vowel is None and nxt is not None and nxt.cons == c:
            return [VOICELESS[c]]  # first half of a geminate
        if prev_dead and prev.cons == c:
            return [VOICELESS[c]]  # second half of a geminate
        if prev_dead and prev.cons in NASALS:
            return [AFTER_NASAL[c]]
        if prev_dead and prev.cons in SONORANTS:
            return [AFTER_SONORANT[c]]
        if prev_dead:
            return [VOICELESS[c]]
        if prev is None:
            return [INITIAL[c]]
        return [INTERVOCALIC[c]]
    if c == "ற":
        if u.vowel is None and nxt is not None and nxt.cons == "ற":
            return ["t"]  # ற்ற -> t r
        if prev_dead and prev.cons == "ற":
            return ["r"]
        if prev_dead and prev.cons == "ன":
            return ["d", "r"]  # ன்ற -> n d r
    return [BASE[c]]


def g2p_ta_word(word: str) -> list[str]:
    units = parse(word)
    phones: list[str] = []
    last = len(units) - 1
    for idx, u in enumerate(units):
        if u.cons is not None:
            phones += _consonant(units, idx)
        if u.vowel is not None:
            if idx == last and idx > 0 and u.cons is not None and u.vowel == "உ":
                phones.append(REDUCED_U)
            else:
                phones.append(VOWELS[u.vowel])
    return phones
