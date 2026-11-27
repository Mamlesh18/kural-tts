"""Tanglish (romanized Tamil) -> Tamil script.

v0 strategy: the curated lexicon handles frequent colloquial words, and a greedy rule-based
transliterator handles everything else. The rules guess vowel length and retroflexion, so the
lexicon matters most. A learned ONNX transliterator replaces the rule fallback in Phase 5 v1.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

from kural_tts.text.tamil_script import PULLI, VOWEL_SIGNS


@dataclass(frozen=True)
class LexEntry:
    tamil: str
    ambiguous: bool  # also a common English word or name


@lru_cache(maxsize=1)
def load_lexicon() -> dict[str, LexEntry]:
    text = files("kural_tts.data").joinpath("lexicon/tanglish.tsv").read_text(encoding="utf-8")
    lex: dict[str, LexEntry] = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        romans, tamil, ambiguous = line.split("\t")
        for roman in romans.split("|"):
            lex[roman.strip().lower()] = LexEntry(tamil.strip(), ambiguous.strip() == "1")
    return lex


def lookup(word: str) -> LexEntry | None:
    return load_lexicon().get(word.lower())


# Particles that attach to the previous word when spoken (office ku -> office-க்கு).
ATTACHING_PARTICLES = frozenset({"la", "ku", "kku", "nu", "nnu", "ah", "aa", "oda"})

# Endings typical of Tamil verbs/nouns, used only to *suggest* Tanglish for unknown words.
_TANGLISH_ENDINGS = (
    "unga", "inga", "anga", "eenga", "aanga", "ren", "rom", "dhu", "thu", "chu",
    "kku", "num", "nam", "lam", "iya", "ttu", "nga", "zhai", "zhi",
)  # fmt: skip


def looks_tanglish(word: str) -> bool:
    w = word.lower()
    if not w.isalpha() or len(w) < 4:
        return False
    return "zh" in w or w.endswith(_TANGLISH_ENDINGS)


# ---------------------------------------------------------------- rule-based fallback
_CONSONANTS = [
    ("zh", "ழ"), ("sh", "ஷ"), ("ch", "ச"), ("th", "த"), ("dh", "த"), ("nj", "ஞ்ச"),
    ("ng", "ங்க"), ("k", "க"), ("g", "க"), ("c", "க"), ("q", "க"), ("s", "ச"), ("j", "ஜ"),
    ("t", "ட"), ("d", "ட"), ("n", "N"), ("p", "ப"), ("b", "ப"), ("m", "ம"), ("y", "ய"),
    ("r", "ர"), ("l", "ல"), ("v", "வ"), ("w", "வ"), ("h", "ஹ"), ("f", "ஃப"), ("x", "க்ஸ"),
    ("z", "ஜ"),
]  # fmt: skip
_VOWELS = [
    ("aa", "ஆ"), ("ee", "ஈ"), ("ii", "ஈ"), ("oo", "ஊ"), ("uu", "ஊ"), ("ai", "ஐ"),
    ("ae", "ஏ"), ("au", "ஔ"), ("a", "அ"), ("i", "இ"), ("u", "உ"), ("e", "எ"), ("o", "ஒ"),
]  # fmt: skip


def _parse(word: str) -> list[tuple[str, str]]:
    units: list[tuple[str, str]] = []
    i = 0
    while i < len(word):
        for table, kind in ((_VOWELS, "V"), (_CONSONANTS, "C")):
            match = next(((r, t) for r, t in table if word.startswith(r, i)), None)
            if match:
                units.append((kind, match[1]))
                i += len(match[0])
                break
        else:
            i += 1  # skip characters we cannot map (digits, apostrophes)
    return units


def _resolve_n(units: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """'n' becomes ண before ட, ந before த or word-initially, otherwise ன."""
    out = []
    for idx, (kind, t) in enumerate(units):
        if t == "N":
            nxt = units[idx + 1][1] if idx + 1 < len(units) else ""
            if nxt.startswith("ட"):
                t = "ண"
            elif nxt.startswith("த") or idx == 0:
                t = "ந"
            else:
                t = "ன"
        out.append((kind, t))
    return out


def rule_transliterate(word: str) -> str:
    units = _resolve_n(_parse(word.lower()))
    out: list[str] = []
    for idx, (kind, t) in enumerate(units):
        prev_kind = units[idx - 1][0] if idx else None
        nxt = units[idx + 1] if idx + 1 < len(units) else None
        if kind == "C":
            out.append(t + (VOWEL_SIGNS[nxt[1]] if nxt and nxt[0] == "V" else PULLI))
        elif prev_kind != "C":  # vowel not consumed by a consonant
            out.append(t)
    return "".join(out)


def transliterate(word: str) -> str:
    entry = lookup(word)
    return entry.tamil if entry else rule_transliterate(word)
