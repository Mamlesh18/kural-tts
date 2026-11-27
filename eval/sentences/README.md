# Evaluation sentence sets

These sets are **never used for training, recording scripts, or hyper-parameter selection**.
`tests/test_eval_sets.py` checks their schema, and later the data pipeline will reject any
training utterance whose normalized text matches an eval sentence.

| File | Purpose | Status |
|---|---|---|
| `feasibility_v1.tsv` | Phase 1 baseline comparison: 20 sentences each of EN, TA, CS and Tanglish | **DRAFT, needs native review** |

## Columns

- `id`: stable ID, `<category>_<nnn>`
- `category`: `en`, `ta`, `cs` (code-switched, Tamil script plus Latin English) or `tanglish` (romanized Tamil)
- `phenomenon`: what the sentence is designed to test (numbers, grantha, zha, question intonation, and so on)
- `text`: the input exactly as a user or LLM would write it
- `reference_ta_script`: Tanglish only. This is the expected Tamil-script form (English words stay in Latin
  script). It serves as the target for the transliteration eval and as the input baseline models receive.

## Review checklist (native Chennai Tamil speaker)

1. Is every Tamil and CS sentence natural, grammatical, and correctly spelled?
2. Are the colloquial forms (`இருக்கேன்`, `பண்ணுங்க`, `புரிஞ்சுது`) how a Chennai speaker would write and say them?
3. Are the Tanglish spellings ones people actually type? Add common spelling variants as notes.
4. Does each `reference_ta_script` match the Tanglish sentence word for word?
5. Fill in the review log below. Once approved, the file is frozen. Any later change creates `feasibility_v2.tsv`.

## Review log

| Date | Reviewer | Result |
|---|---|---|
| — | — | pending |
