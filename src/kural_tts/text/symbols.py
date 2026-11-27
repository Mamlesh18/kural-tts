"""Phoneme inventory shared by the frontend and the model.

The order of ``SYMBOLS`` defines the model's embedding ids. **Never reorder or remove entries.**
Append new symbols at the end and bump ``SYMBOLS_VERSION``, because checkpoints depend on these ids.
"""

from __future__ import annotations

SYMBOLS_VERSION = 1

PAD, BOS, EOS, SPACE = "_", "^", "$", " "
SPECIALS = [PAD, BOS, EOS, SPACE]
PUNCTUATION = [",", ".", "?", "!"]
STRESS = ["ˈ", "ˌ"]  # placed before the stressed English vowel

# General American values for CMUdict ARPAbet. Sahithi's Indian English realization is learned
# from the recordings, so these act as categories rather than exact sound targets.
EN_PHONES = [
    "ɑ", "æ", "ʌ", "ə", "ɔ", "aʊ", "aɪ", "ɛ", "ɝ", "ɚ", "eɪ", "ɪ", "i", "oʊ", "ɔɪ", "ʊ", "u",
    "b", "tʃ", "d", "ð", "f", "ɡ", "h", "dʒ", "k", "l", "m", "n", "ŋ", "p", "ɹ", "s", "ʃ",
    "t", "θ", "v", "w", "j", "z", "ʒ",
]  # fmt: skip

# Chennai-standard Tamil, including allophones produced by g2p_ta's rules.
TA_PHONES = [
    "a", "aː", "i", "iː", "u", "uː", "e", "eː", "o", "oː", "ai", "au", "ɯ",
    "k", "ɡ", "ɣ", "ŋ", "tʃ", "dʒ", "s", "ʃ", "ɲ", "ʈ", "ɖ", "ɳ", "t̪", "d̪", "ð", "n̪",
    "p", "b", "m", "j", "ɾ", "r", "l", "ʋ", "ɻ", "ɭ", "n", "t", "d", "h", "f",
]  # fmt: skip


def _unique(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    return [s for s in seq if not (s in seen or seen.add(s))]


PHONES = _unique(EN_PHONES + TA_PHONES)
SYMBOLS = SPECIALS + PUNCTUATION + STRESS + PHONES
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}

LANGS = ["en", "ta"]
LANG_TO_ID = {lang: i for i, lang in enumerate(LANGS)}


def encode(phones: list[tuple[str, str]]) -> tuple[list[int], list[int]]:
    """[(symbol, lang), ...] -> (symbol ids, language ids), wrapped in BOS/EOS."""
    if not phones:
        return [SYMBOL_TO_ID[BOS], SYMBOL_TO_ID[EOS]], [0, 0]
    first_lang, last_lang = phones[0][1], phones[-1][1]
    seq = [(BOS, first_lang), *phones, (EOS, last_lang)]
    unknown = [s for s, _ in seq if s not in SYMBOL_TO_ID]
    if unknown:
        raise KeyError(f"symbols not in inventory: {unknown}")
    return [SYMBOL_TO_ID[s] for s, _ in seq], [LANG_TO_ID[lang] for _, lang in seq]
