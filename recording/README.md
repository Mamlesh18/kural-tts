# Recording Sahithi

The workflow, from sentences to training data:

```
sentence pools ─┐
templates ──────┼─> make_script.py ─> prompts.tsv ─> recorder.py ─> recordings/session_XX/*.wav
                │   (coverage-greedy,   (id, category,    (48 kHz/24-bit,
eval sets ──────┘    excludes eval)      text, say)         one file per prompt)
                                                                  │
                     training/data_pipeline/build.py <────────────┘
                     (QC, trim, loudness, dedupe, split) ─> data/sahithi_v1/manifests/*.jsonl
```

## 1. Build the sentence pool

`make_script.py` selects from whatever sentences you give it, so most of the effort goes into the pool.
Aim for **3–5× more sentences than you will record**, so the selector has room to pick for
coverage.

| Category | Sources (check licenses; reading aloud is fine for CC-BY-SA) |
|---|---|
| `ta` | Tamil Wikipedia sentences, news-style text you write, **colloquial Chennai lines** like those a voice agent would say (`இருக்கு`, `பண்ணுங்க`) |
| `en` | Conversational and customer-support lines, Indian names and places, product terms |
| `cs` | Real code-mixed lines: English words in Tamil sentences (both scripts: `மீட்டிங்` and `meeting`), and Tamil words in English sentences |

Format: `*_ta.txt`, `*_en.txt`, `*_cs.txt` with one sentence per line, or TSV with `id, category, text`.
Drafting lines with an LLM is fine **if a Chennai Tamil speaker reads through them** before recording.

```bash
uv run python recording/make_script.py --pool pools/*.txt --hours 1 -o recording/pilot_prompts.tsv   # 1 h pilot
uv run python recording/make_script.py --pool pools/*.txt --hours 15 -o recording/prompts.tsv        # full script
```

The output has a **`say`** column: the exact spoken form produced by the Kural frontend
(`₹2,499` becomes இரண்டாயிரத்து நானூற்று தொண்ணூற்றொன்பது ரூபாய், and `10:30` becomes பத்தரை). Sahithi reads `say`.
If she would naturally say a number differently, change the **frontend rule**, not just the
prompt, so training and inference stay consistent.

## 2. Studio setup

- A treated room with a noise floor below −60 dBFS. Turn off fans and AC during takes.
- A large-diaphragm condenser mic (AT2020, Rode NT1 class), a pop filter, and an audio interface.
- **Fixed** mic distance (15–20 cm, mark it) and **fixed** interface gain for the whole project.
  Peaks should land around −12 to −6 dBFS on normal speech.
- Headphones for playback. Water at room temperature, and no dairy before sessions.

## 3. Sessions

```bash
uv sync --group record
uv run python recording/recorder.py --list-devices
uv run python recording/recorder.py --prompts recording/prompts.tsv --out recordings/session_01 --device 3
```

- Each session starts with **3 s of room tone**, so stay silent.
- Per prompt: Enter to record, Enter to stop, then Enter to keep, **r** to retake, **p** to play back,
  **s** to skip, or **q** to quit. The recorder warns on clipping and on low levels.
- Resuming is automatic: prompts that already have a file are skipped.
- Sessions of 2.5–3 h with a break every 45 min. Expect about 1 finished hour per session.
- Keep a **consistent voice**: same energy, pace and distance. Natural and friendly, not a news anchor.
- Re-record the same 10 **anchor prompts** at the start of every session (put them first in the
  prompt file). The pipeline compares them across sessions to catch drift.

## 4. Build the dataset

```bash
uv run --group data python -m training.data_pipeline.build \
    --raw recordings/ --prompts recording/prompts.tsv --out data/sahithi_v1
```

Read `data/sahithi_v1/qc_report.md`. Listen to everything under *Needs listening review*. If a
line was misread, either re-record it or add the actual reading to a `corrections.tsv`
(`id<TAB>text`) and rebuild with `--corrections`.

## 5. Privacy

Raw recordings and the consent form **never go in git** (`.gitignore` blocks `data/` and `*.wav`).
Keep them in a private, access-controlled bucket or Drive folder. See `consent_template.md`.
