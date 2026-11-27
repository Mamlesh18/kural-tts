"""English G2P: CMUdict lookup -> IPA, with a Tamil-route fallback for unknown words.

Words that are not in CMUdict in this domain are mostly Indian names and places (Velachery,
Tiruchirappalli, Aadhaar). A Tamil speaker reads those natively, so they are transliterated to
Tamil script and run through Tamil G2P, and their phonemes are tagged ``ta``. A learned English
G2P for true English out-of-vocabulary words comes in Phase 5 v1.
"""

from __future__ import annotations

import gzip
from functools import lru_cache
from importlib.resources import files

from kural_tts.text.g2p_ta import g2p_ta_word
from kural_tts.text.tanglish import rule_transliterate

ARPABET_TO_IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ", "EH": "ɛ", "ER": "ɝ",
    "EY": "eɪ", "IH": "ɪ", "IY": "i", "OW": "oʊ", "OY": "ɔɪ", "UH": "ʊ", "UW": "u",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f", "G": "ɡ", "HH": "h", "JH": "dʒ",
    "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ", "P": "p", "R": "ɹ", "S": "s",
    "SH": "ʃ", "T": "t", "TH": "θ", "V": "v", "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}  # fmt: skip
_UNSTRESSED = {"AH": "ə", "ER": "ɚ"}  # reduced forms when stress is 0
_STRESS = {"1": "ˈ", "2": "ˌ"}

# Letter names as spoken in Indian English (Z = "zed").
LETTERS = {
    "A": "EY1", "B": "B IY1", "C": "S IY1", "D": "D IY1", "E": "IY1", "F": "EH1 F",
    "G": "JH IY1", "H": "EY1 CH", "I": "AY1", "J": "JH EY1", "K": "K EY1", "L": "EH1 L",
    "M": "EH1 M", "N": "EH1 N", "O": "OW1", "P": "P IY1", "Q": "K Y UW1", "R": "AA1 R",
    "S": "EH1 S", "T": "T IY1", "U": "Y UW1", "V": "V IY1", "W": "D AH1 B AH0 L Y UW0",
    "X": "EH1 K S", "Y": "W AY1", "Z": "Z EH1 D",
}  # fmt: skip


@lru_cache(maxsize=1)
def load_cmudict() -> dict[str, str]:
    raw = files("kural_tts.data").joinpath("lexicon/cmudict.tsv.gz").read_bytes()
    lex = {}
    for line in gzip.decompress(raw).decode("utf-8").splitlines():
        word, phones = line.split("\t")
        lex[word] = phones
    return lex


def arpabet_to_ipa(arpabet: str) -> list[str]:
    out: list[str] = []
    for ph in arpabet.split():
        base, stress = ph.rstrip("012"), ph[len(ph.rstrip("012")) :]
        if stress in _STRESS:
            out.append(_STRESS[stress])
        out.append(
            _UNSTRESSED[base] if stress == "0" and base in _UNSTRESSED else ARPABET_TO_IPA[base]
        )
    return out


def _lookup(word: str) -> list[str] | None:
    arpabet = load_cmudict().get(word.lower())
    return arpabet_to_ipa(arpabet) if arpabet else None


def g2p_en_word(word: str) -> list[tuple[str, str]]:
    """Phonemes for one English-span word as ``[(phone, lang), ...]``."""
    if len(word) == 1 and word.isupper():
        return [(p, "en") for p in arpabet_to_ipa(LETTERS[word])]
    word = word.strip("'")
    if not word:
        return []
    phones = _lookup(word)
    if phones is not None:
        return [(p, "en") for p in phones]
    if "-" in word:  # passport-size, Wi-Fi variants
        out: list[tuple[str, str]] = []
        for part in filter(None, word.split("-")):
            out += g2p_en_word(part)
        return out
    if "'" in word:  # unknown contraction: drop the clitic rather than mispronounce
        return g2p_en_word(word.split("'")[0])
    return [(p, "ta") for p in g2p_ta_word(rule_transliterate(word))]
