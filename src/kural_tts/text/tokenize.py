"""Split cleaned text into typed tokens for language ID and normalization."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Order matters: more specific number shapes first.
_TOKEN_RE = re.compile(
    r"""
    (?P<email>[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)
  | (?P<currency>(?:₹|\$|Rs\.?\ ?)\d[\d,]*(?:\.\d+)?)
  | (?P<phone>\+?\d{2,5}(?:-\d{3,5}){2,3}|\+\d{10,13}|\b\d{10}\b)
  | (?P<time>\d{1,2}:\d{2})
  | (?P<percent>\d[\d,]*(?:\.\d+)?%)
  | (?P<ordinal>\d+(?:st|nd|rd|th)\b)
  | (?P<number>\d[\d,]*(?:\.\d+)?)
  | (?P<ta>[஀-௿]+)
  | (?P<latin>[A-Za-z]+(?:'[A-Za-z]+)*(?:-[A-Za-z]+)*)
  | (?P<space>\s+)
  | (?P<punct>.)
    """,
    re.VERBOSE,
)

NUMERIC_KINDS = frozenset({"currency", "phone", "time", "percent", "ordinal", "number"})
WORD_KINDS = frozenset({"ta", "latin"})


@dataclass
class Token:
    text: str
    kind: str  # email|currency|phone|time|percent|ordinal|number|ta|latin|punct
    lang: str | None = None  # "ta" | "en", assigned by langid
    space_before: bool = False
    attached: bool = False  # Tamil suffix glued to the previous token (15ஆம், office-ல)
    source: str | None = None  # e.g. "tanglish" when transliterated from romanized Tamil

    @property
    def is_word(self) -> bool:
        return self.kind in WORD_KINDS

    @property
    def is_numeric(self) -> bool:
        return self.kind in NUMERIC_KINDS


def tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    space = False
    for m in _TOKEN_RE.finditer(text):
        kind = m.lastgroup
        if kind == "space":
            space = True
            continue
        tokens.append(Token(m.group(), kind, space_before=space))
        space = False
    return _mark_attached_suffixes(tokens)


def _mark_attached_suffixes(tokens: list[Token]) -> list[Token]:
    """Mark Tamil tokens glued to a number or Latin word, dropping a joining hyphen.

    "15ஆம்" -> [15, ஆம்(attached)];  "office-ல" -> [office, ல(attached)].
    """
    out: list[Token] = []
    for tok in tokens:
        if tok.kind == "ta" and not tok.space_before and out:
            prev = out[-1]
            if prev.is_numeric or prev.kind == "latin":
                tok.attached = True
            elif (
                prev.kind == "punct"
                and prev.text == "-"
                and not prev.space_before
                and len(out) >= 2
                and (out[-2].is_numeric or out[-2].kind == "latin")
            ):
                out.pop()  # drop the hyphen
                tok.attached = True
        out.append(tok)
    return out
