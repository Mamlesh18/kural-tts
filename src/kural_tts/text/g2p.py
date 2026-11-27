"""Spans -> phoneme sequence with a language id on every symbol (the model's input).

Word boundaries become ``SPACE``. Punctuation maps to the pause symbols ``, . ? !``.
Boundary and punctuation symbols take the language of the preceding phoneme.
"""

from __future__ import annotations

import re

from kural_tts.text.frontend import Span, normalize
from kural_tts.text.g2p_en import g2p_en_word
from kural_tts.text.g2p_ta import g2p_ta_word
from kural_tts.text.langid import LangMode
from kural_tts.text.symbols import PUNCTUATION, SPACE, encode

Phone = tuple[str, str]  # (symbol, lang)

_PIECE = re.compile(r"[஀-௿]+|[A-Za-z]+(?:['-][A-Za-z]+)*|\S")
# Other punctuation -> pause class; anything unlisted (quotes, brackets) is dropped.
_PUNCT_MAP = {";": ",", ":": ",", "-": ",", "…": ".", "(": ",", ")": ","}


def _punct(ch: str) -> str | None:
    if ch in PUNCTUATION:
        return ch
    return _PUNCT_MAP.get(ch)


def _span_phones(span: Span) -> list[Phone]:
    out: list[Phone] = []
    for piece in _PIECE.findall(span.text):
        first = piece[0]
        if "஀" <= first <= "௿":
            word = [(p, "ta") for p in g2p_ta_word(piece)]
        elif first.isascii() and first.isalpha():
            word = g2p_en_word(piece)
        else:
            mark = _punct(piece)
            if mark:
                out.append((mark, out[-1][1] if out else span.lang))
            continue
        if word:
            if out:
                out.append((SPACE, out[-1][1]))
            out += word
    return out


def _cleanup(phones: list[Phone]) -> list[Phone]:
    """Collapse repeated boundaries and drop a space next to punctuation."""
    out: list[Phone] = []
    for sym, lang in phones:
        if sym == SPACE and (not out or out[-1][0] == SPACE or out[-1][0] in PUNCTUATION):
            continue
        if sym in PUNCTUATION and out and out[-1][0] == SPACE:
            out.pop()
        if sym in PUNCTUATION and out and out[-1][0] in PUNCTUATION:
            out[-1] = (sym, lang)  # keep the stronger (last) mark: "?!" -> "!"
            continue
        out.append((sym, lang))
    while out and out[-1][0] == SPACE:
        out.pop()
    return out


def phonemize_spans(spans: list[Span]) -> list[Phone]:
    phones: list[Phone] = []
    for span in spans:
        part = _span_phones(span)
        if not part:
            continue
        if phones and not span.joined:
            phones.append((SPACE, phones[-1][1]))
        phones += part
    return _cleanup(phones)


def phonemize(text: str, lang: LangMode = "auto") -> list[Phone]:
    """Raw text -> ``[(symbol, lang), ...]``."""
    return phonemize_spans(normalize(text, lang))


def text_to_ids(text: str, lang: LangMode = "auto") -> tuple[list[int], list[int]]:
    """Raw text -> (symbol ids, language ids) with BOS/EOS, ready for the model."""
    return encode(phonemize(text, lang))


def format_phones(phones: list[Phone]) -> str:
    """Human-readable form: words separated by ' | ', language switches marked as [ta]/[en]."""
    out, lang = [], None
    for sym, sym_lang in phones:
        if sym_lang != lang and sym != SPACE and sym not in PUNCTUATION:
            out.append(f"[{sym_lang}]")
            lang = sym_lang
        out.append("|" if sym == SPACE else sym)
    return " ".join(out)
