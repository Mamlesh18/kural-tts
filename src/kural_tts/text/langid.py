"""Token-level language ID: Tamil script / English / Tanglish, plus number language from context.

After ``assign_languages`` every word and number token has ``lang`` in {"ta", "en"}. Tanglish
words are rewritten into Tamil script with ``source="tanglish"``. Punctuation keeps ``lang=None``.
"""

from __future__ import annotations

from typing import Literal

from kural_tts.text.tanglish import ATTACHING_PARTICLES, looks_tanglish, lookup, transliterate
from kural_tts.text.tokenize import Token

LangMode = Literal["auto", "ta", "en"]

_STRONG, _WEAK, _EN, _AMBIG = "strong", "weak", "en", "ambiguous"
_WINDOW = 3
_SENTENCE_END = frozenset(".?!")


def _classify_latin(tok: Token) -> str:
    entry = lookup(tok.text)
    if entry is not None:
        return _AMBIG if entry.ambiguous else _STRONG
    if looks_tanglish(tok.text):
        return _WEAK
    return _EN


def _context_score(labels: list[str | None], tokens: list[Token], i: int) -> float:
    score = 0.0
    lo, hi = max(0, i - _WINDOW), min(len(tokens), i + _WINDOW + 1)
    for j in range(lo, hi):
        if j == i:
            continue
        lab, tok = labels[j], tokens[j]
        if lab == _STRONG:
            score += 1.0
        elif tok.kind == "ta":
            score += 0.5
        elif lab == _EN and tok.text[:1].islower():
            score -= 0.5  # capitalised unknown words are usually names: neutral
    return score


def assign_languages(tokens: list[Token], mode: LangMode = "auto") -> list[Token]:
    labels: list[str | None] = [_classify_latin(t) if t.kind == "latin" else None for t in tokens]
    if mode == "en":
        labels = [_EN if lab else None for lab in labels]

    has_strong = _STRONG in labels
    for i in range(len(tokens)):
        if labels[i] in (_AMBIG, _WEAK):
            score = _context_score(labels, tokens, i)
            tanglish = score > 0 or (score == 0 and (has_strong or mode == "ta"))
            if labels[i] == _WEAK:  # unknown words need positive evidence
                tanglish = score > 0 and has_strong
            labels[i] = _STRONG if tanglish else _EN

    for i, tok in enumerate(tokens):
        if tok.kind == "ta":
            tok.lang = "ta"
        elif tok.kind == "latin":
            if labels[i] == _STRONG:
                tok.source, tok.lang = "tanglish", "ta"
                if tok.text.lower() in ATTACHING_PARTICLES and i > 0 and tokens[i - 1].is_word:
                    tok.attached = True
                tok.text = transliterate(tok.text)
                tok.kind = "ta"
            else:
                tok.lang = "en"
        elif tok.kind == "email":
            tok.lang = "en"

    _assign_number_languages(tokens, mode)
    return tokens


def _assign_number_languages(tokens: list[Token], mode: LangMode) -> None:
    words = [t.lang for t in tokens if t.lang]
    default = mode if mode != "auto" else _majority(words)
    for i, tok in enumerate(tokens):
        if not tok.is_numeric:
            continue
        if i + 1 < len(tokens) and tokens[i + 1].attached and tokens[i + 1].lang == "ta":
            tok.lang = "ta"  # 15ஆம், 10க்கு
            continue
        tok.lang = _neighbour_lang(tokens, i) or default


def _neighbour_lang(tokens: list[Token], i: int) -> str | None:
    """Language of the next word in the same sentence, else of the previous word."""
    for j in range(i + 1, len(tokens)):
        t = tokens[j]
        if t.kind == "punct" and t.text in _SENTENCE_END:
            break
        if t.is_word and t.lang:
            return t.lang
    for j in range(i - 1, -1, -1):
        t = tokens[j]
        if t.kind == "punct" and t.text in _SENTENCE_END:
            break
        if t.is_word and t.lang:
            return t.lang
    return None


def _majority(langs: list[str]) -> str:
    return "ta" if langs.count("ta") > langs.count("en") else "en"
