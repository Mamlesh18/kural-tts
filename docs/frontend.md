# Text frontend (v0)

The frontend turns raw user or LLM text into **language-tagged, fully spoken-form spans**. The model
(and, next, G2P) only ever sees words, never digits or symbols.

```
raw text
  └─ cleanup.clean          NFC, zero-width chars, Tamil digits ௦-௯ → 0-9, quotes/dashes, ஶ்ரீ → ஸ்ரீ
  └─ tokenize.tokenize      email | currency | phone | time | percent | ordinal | number | ta | latin | punct
                            + marks Tamil suffixes glued to numbers/English words (15ஆம், office-ல)
  └─ langid.assign_languages
        Tamil script        → ta
        Latin, Tanglish     → transliterated to Tamil script, lang=ta, source="tanglish"
        Latin, other        → en
        numbers             → language of attached suffix, else next word, else previous word
  └─ frontend.verbalize     normalize_ta.verbalize_ta / normalize_en.verbalize_en
  └─ frontend.to_spans      [Span(text, lang), ...]
```

Try it:

```bash
uv run kural-tts "naalaikku 10:30 ku meeting, ok ah?" --normalize
[ta] நாளைக்கு பத்தரைக்கு
[en] meeting, ok
[ta] ஆ?
```

```python
from kural_tts.text import normalize

normalize("Total ₹1,250 ஆகும்")
# [Span('Total', 'en'), Span('ஆயிரத்து இருநூற்று ஐம்பது ரூபாய் ஆகும்', 'ta')]
```

## Modules

| Module | Responsibility |
|---|---|
| `tamil_script.py` | Unicode constants, `attach_vowel` sandhi (இருபத்து+ஒன்று → இருபத்தொன்று) |
| `numbers_ta.py` | Tamil cardinals with ஆயிரம்/லட்சம்/கோடி, combining forms, decimals, digit strings, suffix attachment |
| `numbers_en.py` | English cardinals (Indian lakh/crore and Western), ordinals, years, decimals |
| `cleanup.py` | Unicode cleanup safe for both languages |
| `tokenize.py` | Typed tokens and attached-suffix detection |
| `tanglish.py` | Tanglish lexicon (`data/lexicon/tanglish.tsv`) and rule-based fallback transliteration |
| `langid.py` | Token language ID with context resolution for ambiguous words |
| `normalize_en.py` / `normalize_ta.py` | Verbalizers for currency, time, phone, %, ordinals, abbreviations, acronyms |
| `frontend.py` | Pipeline and span grouping |

## Key decisions

- **Spoken Chennai forms where they matter:** `10:30` → பத்தரை, and Tanglish maps to colloquial
  spellings (பண்றேன், இருக்கு), because that is what the voice actually says.
- **Numbers follow their context:** `meeting 10 மணிக்கு` → பத்து; `at 10:30 AM` → ten thirty.
- **Indian digit grouping:** `1,50,000` → lakh. Western grouping (`150,000`) is detected and read
  Western-style in English.
- **Acronyms are spelled out** (`SBI` → S B I) unless listed as word-acronyms (`NASA`, `PAN`).
- **Ambiguous Tanglish words** (`pal`, `naan`, `sari`, `anna`, particles `la`/`ku`) are decided by a ±3-word
  context window. Unknown words that only *look* Tamil (`-unga`, `-ren`, `zh`) need positive context.

## Known limitations (v0), tracked for v1

1. **Tanglish fallback is rule-based.** It guesses vowel length and retroflexes (பண்றென் instead of
   பண்றேன் for unknown words). It will be replaced by an ONNX seq2seq model trained on Aksharantar.
2. The lexicon has about 270 spellings. It needs to grow to about 5k colloquial forms with native review.
3. Tanglish accusative `-ah` is rendered as ஆ (`password ah` → passwordஆ). That matches casual speech,
   but the formal reference is -ஐ.
4. ~~English words with Tamil suffixes become two spans.~~ Resolved: spans carry `joined`, and G2P
   emits no word boundary between `office` and `க்கு` (see [g2p.md](g2p.md)).
5. Brand and mixed-case tokens (`GPay`, `iPhone`) are passed through unchanged. G2P looks them up
   in CMUdict and falls back to the Tamil route.
6. Tamil numerals ௰ ௱ ௲ are not converted.
7. The measured Tanglish accuracy of 99.1% (`scripts/frontend/eval_tanglish.py`) is **optimistic**,
   because the lexicon and the feasibility sentences have the same author. The Phase 5 exit criterion
   needs a separate 500-sentence native-reviewed set.
