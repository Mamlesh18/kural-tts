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
# In a sentence that already has unambiguous Tanglish, an ambiguous word stays Tanglish unless
# its neighbours are clearly English (e.g. "meeting, ok ah?" -> ah is the Tamil particle ஆ).
_AMBIG_TOLERANCE = -1.5
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


def split_hyphen_particles(tokens: list[Token]) -> list[Token]:
    """'account-la' -> [account, la(attached)] when the tail is a Tanglish particle."""
    out: list[Token] = []
    for tok in tokens:
        head, sep, tail = tok.text.rpartition("-")
        if tok.kind == "latin" and sep and head and tail.lower() in ATTACHING_PARTICLES:
            out.append(Token(head, "latin", space_before=tok.space_before))
            out.append(Token(tail, "latin", attached=True))
        else:
            out.append(tok)
    return out


def assign_languages(tokens: list[Token], mode: LangMode = "auto") -> list[Token]:
    if mode != "en":
        tokens = split_hyphen_particles(tokens)
    labels: list[str | None] = [_classify_latin(t) if t.kind == "latin" else None for t in tokens]
    if mode == "en":
        labels = [_EN if lab else None for lab in labels]
    # A particle glued with a hyphen ("account-la") is Tamil regardless of context.
    labels = [
        _STRONG if t.attached and t.kind == "latin" else lab
        for t, lab in zip(tokens, labels, strict=True)
    ]

    has_strong = _STRONG in labels
    for i in range(len(tokens)):
        if labels[i] in (_AMBIG, _WEAK):
            score = _context_score(labels, tokens, i)
            if labels[i] == _WEAK:  # unknown words need positive local evidence
                tanglish = score > 0 and has_strong
            else:  # lexicon words that are also English: lean on the sentence when it is Tanglish
                tanglish = (
                    score > 0
                    or (has_strong and score > _AMBIG_TOLERANCE)
                    or (mode == "ta" and score >= 0)
                )
            labels[i] = _STRONG if tanglish else _EN

    for i, tok in enumerate(tokens):
        if tok.kind == "ta":
            tok.lang = "ta"
        elif tok.kind == "latin":
            if labels[i] == _STRONG:
                tok.source, tok.lang = "tanglish", "ta"
                prev = tokens[i - 1] if i > 0 else None
                if (
                    tok.text.lower() in ATTACHING_PARTICLES
                    and prev
                    and (prev.is_word or prev.is_numeric)
                ):
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
