# G2P and model input (v0)

The frontend's spans become the model's input, a sequence of **(phoneme, language)** pairs:

```bash
uv run kural-tts "Sir, unga account-la KYC update pannunga" --phonemes
[en] s ˈ ɝ , [ta] u ŋ ɡ a | [en] ə k ˈ aʊ n t [ta] l a | [en] k ˈ eɪ | w ˈ aɪ | s ˈ i | ə p d ˈ eɪ t | [ta] p a ɳ ɳ u ŋ ɡ a
```

`|` is a word boundary (`SPACE`), `[ta]`/`[en]` mark language switches, and `, . ? !` are pause
symbols. `account` + `ல` has no boundary because the suffix is glued to the word.

```python
from kural_tts.text import text_to_ids
ids, lang_ids = text_to_ids("நாளைக்கு meeting 10 மணிக்கு")   # BOS ... EOS, equal lengths
```

## Symbol inventory (`symbols.py`, version 1, 75 symbols)

| Group | Symbols |
|---|---|
| Specials | `_` pad, `^` BOS, `$` EOS, ` ` word boundary |
| Pauses | `, . ? !` (`; : - ( )` map to `,`, `…` maps to `.`, quotes are dropped) |
| Stress | `ˈ` primary, `ˌ` secondary (English only, placed before the vowel) |
| English | 41 IPA phones from CMUdict ARPAbet (General American categories) |
| Tamil | 13 vowels including `ɯ` and 31 consonants including allophones `ɣ ð ɖ d̪ b ɡ dʒ`, with `t r` for ற்ற |

Shared phones (`m`, `s`, `k` ...) use one symbol for both languages. The **per-symbol language
id** lets the model learn that, for example, `t` in English and `t` in Tamil (ற்ற) differ in
Sahithi's speech. **The ids are frozen:** new symbols are appended and `SYMBOLS_VERSION` is bumped.

## Tamil G2P (`g2p_ta.py`)

Rule-based, Chennai standard. Each word is parsed into units (consonant + vowel sign / pulli /
independent vowel), then the plain stops get position-dependent allophones:

| Position | க | ச | ட | த | ப |
|---|---|---|---|---|---|
| word-initial | k | **s** | ʈ | t̪ | p |
| geminate (க்க) | k | tʃ | ʈ | t̪ | p |
| after nasal (ங்க) | ɡ | dʒ | ɖ | d̪ | b |
| after ய ர ல வ ழ ள | ɡ | s | ɖ | d̪ | b |
| between vowels | ɣ | s | ɖ | ð | b |

Also: ற்ற → `t r`, ன்ற → `n d r`, ஃப → `f`, word-final consonant+உ → `ɯ` (kutriyalukaram).

## English G2P (`g2p_en.py`)

1. **CMUdict** (126k words, packaged as `data/lexicon/cmudict.tsv.gz`, BSD-style license) → IPA
   with stress. AH0/ER0 become `ə`/`ɚ`.
2. **Single capital letters** (from spelled acronyms, `S B I`) use Indian English letter names (Z = "zed").
3. **Hyphenated** words are split and looked up per part.
4. **Unknown words** (mostly Indian names and places: Velachery, Aadhaar, Coimbatore) are
   transliterated to Tamil script and passed to Tamil G2P, tagged `ta`, which is how Sahithi says them.

## Native review

```bash
uv run python scripts/frontend/make_g2p_review_sheet.py --from-eval -o g2p_review.csv
# reviewer fills correct=y/n, corrected, notes (opens in Excel)
uv run python scripts/frontend/make_g2p_review_sheet.py --score g2p_review.csv
```

Phase 5 exit: Tamil word accuracy ≥ 98% on ≥ 500 native-reviewed words.

## Known limitations (v0)

1. The Tamil rules are unvalidated until the native review is done. Open questions to settle there:
   - intervocalic க as `ɣ` vs `h` in Chennai speech
   - word-initial ச as `s` vs `tʃ` in careful speech
   - word-final ஐ as `ai` vs `e`
2. English out-of-vocabulary words that are not Indian (new brands, typos) get Tamil-style
   pronunciation. A learned English G2P (ONNX) is planned for v1.
3. CMUdict gives one pronunciation per word, so heteronyms (read/read, live/live) are not resolved.
4. Indian English realization (e.g. retroflex `t`/`d`, no `ɹ` coda) is left for the model to learn
   from Sahithi's recordings rather than encoded in the symbols.
