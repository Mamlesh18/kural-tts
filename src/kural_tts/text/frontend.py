"""Frontend entry point: raw text -> normalized, language-tagged spans.

    clean -> tokenize -> language ID (+ Tanglish transliteration) -> verbalize -> spans

G2P (phonemes with per-phoneme language IDs) builds on these spans in the next stage.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from kural_tts.text.cleanup import clean
from kural_tts.text.langid import LangMode, assign_languages
from kural_tts.text.normalize_en import verbalize_en
from kural_tts.text.normalize_ta import verbalize_ta
from kural_tts.text.tokenize import Token, tokenize


@dataclass(frozen=True)
class Span:
    text: str
    lang: str  # "ta" | "en"


def analyze(text: str, lang: LangMode = "auto") -> list[Token]:
    """Tokenize and language-tag ``text`` (Tanglish already converted to Tamil script)."""
    return assign_languages(tokenize(clean(text)), lang)


def verbalize(tokens: list[Token]) -> list[Token]:
    """Replace numbers, abbreviations, etc. with spoken words. Returns new tokens."""
    out: list[Token] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.lang == "ta":
            text, skip = verbalize_ta(tokens, i)
        elif tok.lang == "en":
            text, skip = verbalize_en(tokens, i)
        else:
            text, skip = tok.text, 0
        out.append(replace(tok, text=text))
        i += 1 + skip
    return out


def _joiner(tok: Token) -> str:
    return " " if tok.space_before and not tok.attached else ""


def to_spans(tokens: list[Token]) -> list[Span]:
    """Group tokens into maximal same-language spans. Punctuation joins the preceding span."""
    spans: list[list] = []  # [text, lang]
    for tok in tokens:
        lang = tok.lang or (spans[-1][1] if spans else None)
        if lang is None:  # leading punctuation: attach to the first real token later
            lang = next((t.lang for t in tokens if t.lang), "en")
        if spans and spans[-1][1] == lang:
            spans[-1][0] += _joiner(tok) + tok.text
        else:
            spans.append([tok.text, lang])
    return [Span(text.strip(), lang) for text, lang in spans if text.strip()]


def normalize(text: str, lang: LangMode = "auto") -> list[Span]:
    """Normalize ``text`` into language-tagged spans ready for G2P."""
    return to_spans(verbalize(analyze(text, lang)))


def normalize_text(text: str, lang: LangMode = "auto") -> str:
    """Normalized text as a single string (for debugging and the CLI)."""
    tokens = verbalize(analyze(text, lang))
    return "".join(_joiner(t) + t.text for t in tokens).strip()
