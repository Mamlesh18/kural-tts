"""Sentence splitting used for chunked streaming."""

from __future__ import annotations

import re

# Split after . ? ! (and the Devanagari/Tamil-adjacent danda forms some writers use), keeping the
# punctuation attached. Avoid splitting inside decimals like "10.5" or common abbreviations.
_ABBREVIATIONS = {
    "dr",
    "mr",
    "mrs",
    "ms",
    "prof",
    "st",
    "no",
    "vs",
    "etc",
    "e.g",
    "i.e",
    "டாக்",
    "திரு",
}
_SPLIT_RE = re.compile(r"(?<=[.?!।])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for streaming synthesis."""
    parts = _SPLIT_RE.split(text.strip())
    sentences: list[str] = []
    for part in parts:
        if not part:
            continue
        if sentences and _ends_with_abbreviation(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {part}"
        else:
            sentences.append(part)
    return sentences


def _ends_with_abbreviation(sentence: str) -> bool:
    last = sentence.rstrip(".").rsplit(maxsplit=1)[-1].lower() if sentence.strip() else ""
    return sentence.endswith(".") and last in _ABBREVIATIONS
