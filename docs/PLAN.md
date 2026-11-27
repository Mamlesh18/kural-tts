# Kural-TTS: Research Findings and Implementation Plan (v2)

**Goal:** an **open-source, pip-installable** Python TTS package that speaks Tamil, English, Tamil-English code-switching and **Tanglish (romanized Tamil)** in one consistent female voice, **Sahithi**, and runs locally on the user's own machine (CPU, or GPU when available).

**Status (2026-11-27):** the repo is empty. No recordings, no dataset, no model.
- Dev machine: Windows 11, Python 3.11.9, **no NVIDIA GPU**.
- Training will run on **Google Colab**.

### Decisions locked (2026-11-27)

| # | Decision | Consequence |
|---|---|---|
| 1 | **Commercial use and open source.** Code and model weights will be released publicly. | Every component, including the weights, the training data, the warm-start checkpoint and runtime dependencies, must allow commercial redistribution. **GPL dependencies such as espeak-ng are excluded from the runtime.** The voice talent must consent to an *irrevocable public* release (§Phase 2). |
| 2 | **Training on Colab.** A small local dataset is used for development. | Training must be resumable and checkpoint to Google Drive. The local Windows machine (CPU) is for the data pipeline, the frontend, unit tests and tiny overfit tests, not real training. |
| 3 | **No talent contacts.** | The plan includes a full sourcing and audition procedure. |
| 4 | **Standard Chennai Tamil** and natural Indian English. | Recording script, G2P rules and native reviewers all target Chennai standard. |
| 5 | **Tanglish input is in scope.** | The frontend needs a romanized-Tamil detector and Tanglish→Tamil-script transliteration (§Phase 5). |
| 6 | **Distribution as a Python package, not a hosted API.** | The model must be small and CPU-friendly (ONNX). Weights download from Hugging Face Hub on first use. An optional local server or streaming API is provided for voice agents, but it is not the main product. |

---

## Progress

| Step | Status | Where |
|---|---|---|
| 1. Repo and package scaffold, multi-OS CI | ✅ done | `pyproject.toml`, `src/kural_tts/`, `.github/workflows/ci.yml` |
| 2. Feasibility sentence set (80: EN/TA/CS/Tanglish) | ✅ drafted, ⏳ native review | `eval/sentences/feasibility_v1.tsv` |
| 3. Baseline runners, scoring, listening test, Colab notebook | ✅ code done, ⏳ run on Colab | `scripts/baseline/`, `notebooks/01_baselines.ipynb` |
| 4. Frontend v0: normalization, Tamil/English numbers, LID, Tanglish | ✅ done (v0) | `src/kural_tts/text/`, `docs/frontend.md` |
| 5. Talent call and consent contract | ⏳ owner: you + lawyer | Phase 2 |
| 6. G2P v0: Tamil rules, CMUdict English, frozen symbol inventory, model input ids | ✅ done (v0), ⏳ native review | `src/kural_tts/text/g2p*.py`, `docs/g2p.md` |
| Next: recording script generator (coverage-based sentence selection) | 🔜 | Phase 3 |

---

## 0. Executive summary

| Decision | Recommendation | Why |
|---|---|---|
| Primary architecture | **VITS** (end-to-end, ~30–40M params) with **per-phoneme language IDs**, trained on Sahithi data and exported to **ONNX** | Runs in real time on an ordinary laptop CPU with onnxruntime. It is mature, single-speaker-friendly, and 10–20 h of data is enough. Its lineage can be kept fully commercial-clean. |
| Challenger | **Matcha-TTS** (MIT code, flow matching, ONNX export supported) + HiFi-GAN vocoder | Also small and CPU-friendly, often with better prosody. Head-to-head in Phase 8. |
| Dropped from shippable candidates | svara-TTS (3B, needs a GPU), Indic Parler-TTS (0.9B, autoregressive) | Too heavy for a pip package that runs locally. They remain **quality references** in the feasibility study. |
| Baseline-only (non-commercial) | MMS-TTS-tam, XTTS-v2, F5-TTS base, and IndicF5 (lineage unclear) | Their licenses block an open commercial release. |
| Data | 15 h from one consenting speaker: ~40% Tamil, ~35% English, ~25% code-switched. **1 h pilot first.** | The pilot validates the full pipeline before most of the talent budget is spent. |
| Text frontend | Pure-Python normalization, **our own Tamil rule-based G2P**, English lexicon G2P with a small ONNX G2P fallback, and **Tanglish transliteration via a small ONNX seq2seq** | No system dependencies, works the same on Windows, macOS and Linux, and contains no GPL code. |
| Compute | **Colab Pro or Pro+** (L4 or A100) for training. Free T4 only for smoke tests. | VITS needs about 150–300k steps, which is not realistic within free-tier session limits. |
| Streaming | **Sentence-chunked streaming** through a Python generator | VITS cannot stream incrementally within a sentence, and the plan says so. |

---

## 1. Feasibility study

### 1.1 Candidate landscape (shippable column reflects decisions 1 and 6)

| Model | Arch | Tamil | English | CS | Size, runs on CPU? | License | Role |
|---|---|---|---|---|---|---|---|
| Coqui VITS en (LJSpeech) via `coqui-tts` (Idiap) | VITS | – | Yes | – | 36M, yes | Code MPL-2.0. LJSpeech is public domain. **Checkpoint license to be verified.** | **Warm-start candidate** |
| Matcha-TTS (LJSpeech) | Flow matching + HiFi-GAN | – | Yes | – | ~18M + vocoder, yes (ONNX) | Code MIT. Weights inherit dataset terms (LJSpeech is PD). | **Challenger warm start** |
| facebook/mms-tts-tam | VITS | Yes | – | – | 36M, yes | CC-BY-NC-4.0 | Baseline only |
| Indic Parler-TTS | Parler (AR) + DAC | Yes | Yes | Partial | 0.9B, slow on CPU | Apache-2.0 | Quality reference |
| svara-TTS v1 | Orpheus 3B + SNAC | Yes | Yes | Claimed | 3B, needs GPU | Apache-2.0 | Quality reference |
| IndicF5 | F5 flow matching | Yes | Yes | Partial | 330M, slow on CPU | Tagged MIT, but training-data terms disputed | Quality reference |
| XTTS-v2 | GPT + HiFiGAN | – | Yes | – | 470M | CPML (non-commercial) | Reference only |
| Chatterbox + Indic LoRA | LLM + flow | Community LoRA | Yes | ? | 0.5B | Base MIT, LoRA unclear | Reference only |

**Fallback if no warm-start checkpoint has a clean license:** train from scratch on Sahithi data plus LJSpeech (public domain). This is feasible for VITS but takes about twice as many Colab hours.

### 1.2 Baseline protocol (Phase 1 deliverable)
- **Fixed feasibility set:** `eval/sentences/feasibility_v1.tsv`, with 80 sentences: 20 EN, 20 TA, 20 CS and 20 Tanglish. It covers questions, numbers, dates, ₹ amounts, names and abbreviations. A native speaker reviews it, then it is frozen.
- **Runner:** run every candidate on one Colab GPU session through `scripts/baseline/run_<model>.py`. Outputs go to `outputs/baseline/<model>/<id>.wav` plus `meta.json` (latency, RTF on GPU **and on CPU**, peak memory, model version, and the license as it appeared on the download date).
- **Tanglish handling:** candidates cannot read Tanglish natively, so it is passed through our Tanglish→Tamil transliterator (frontend v0). This also tests the frontend early.
- **Scoring:**
  - CER/WER with IndicConformer (TA) and Whisper large-v3 (EN).
  - UTMOS (EN).
  - Native-listener MOS (1–5) and a pronunciation-error tally.
- **Output:** `reports/feasibility.md` and `docs/decisions/001-architecture.md`.
- **Exit:** every candidate has been scored, the warm-start checkpoint license is confirmed in writing, and the architecture decision is recorded.

---

## 2. Repository and package structure

The package name is `kural-tts` on PyPI, imported as `kural_tts`.

```
kural-tts/
├─ pyproject.toml           # core deps: numpy, onnxruntime, soundfile, huggingface_hub
│                           # extras: [gpu] onnxruntime-gpu, [train] torch etc., [server] fastapi, [agents] livekit
├─ src/kural_tts/
│  ├─ __init__.py           # from kural_tts import KuralTTS
│  ├─ api.py                # KuralTTS class: synthesize(), stream(), save()
│  ├─ cli.py                # `kural-tts "text" -o out.wav`, `kural-tts serve`
│  ├─ hub.py                # download + sha256-verify weights from HF Hub, local cache
│  ├─ runtime/              # onnxruntime session mgmt, CPU/GPU provider selection
│  ├─ text/                 # FRONTEND (shipped, pure Python + tiny ONNX models)
│  │  ├─ normalize_en.py  normalize_ta.py  numbers_ta.py
│  │  ├─ langid.py          # Tamil-script / English / Tanglish token tagging
│  │  ├─ tanglish.py        # romanized Tamil → Tamil script (ONNX seq2seq + lexicon)
│  │  ├─ g2p_ta.py          # rule-based Tamil G2P (Chennai standard)
│  │  ├─ g2p_en.py          # lexicon + ONNX G2P fallback
│  │  └─ symbols.py         # phoneme inventory, language IDs
│  ├─ data/lexicon/         # shipped pronunciation + Tanglish dictionaries (TSV)
│  └─ server/               # optional extra: FastAPI + WebSocket streaming
├─ training/                # NOT shipped in the wheel
│  ├─ data_pipeline/        # ingest, QC, alignment, manifest, split
│  ├─ models/               # Kural-VITS (adapted), Matcha challenger
│  ├─ train.py  export_onnx.py
│  ├─ eval/                 # ASR CER/WER, speaker sim, MOS tooling
│  └─ configs/
├─ notebooks/               # Colab: 01_baselines, 02_train_vits, 03_eval, 04_export
├─ recording/               # prompt scripts, recording app, consent template
├─ eval/sentences/          # frozen eval sets (never train on these)
├─ tests/                   # pytest; frontend tests dominate
├─ docs/                    # PLAN.md, decisions/, MODEL_CARD.md, experiments.md
└─ data/ checkpoints/ logs/ outputs/   (gitignored)
```

**Tooling:**
- `uv` + `pyproject.toml` + `uv.lock`, ruff, pre-commit.
- GitHub Actions CI: pytest on Windows, macOS and Linux for Python 3.10–3.13, and a wheel-build smoke test.

**Where artifacts live:**
- Recordings and raw data sit in a private Google Drive folder, which Colab mounts.
- Weights are published to a Hugging Face model repo (e.g. `<org>/kural-tts-sahithi`) with a model card.
- Training code stays in this repo.

---

## 3. Phased plan

Each phase has an exit criterion that must be met before the next phase starts. **Track A** (code and models: Phases 1, 5, 6, 7) runs in parallel with **Track B** (voice and data: Phases 2, 3, 4).

### Phase 1: Feasibility study (weeks 1–2)
See §1.2.

### Phase 2: Voice talent sourcing and consent (weeks 1–5)
You have no existing contacts, so sourcing runs as follows.

- **Channels:**
  - Chennai dubbing artists (the dubbing artistes' union and dubbing studios in Vadapalani, Kodambakkam and Saligramam).
  - FM and AIR RJs and newsreaders.
  - Voice123, Voices.com, Upwork and Fiverr, filtered for Tamil and English, female, with home-studio samples.
  - Theatre groups, and media or visual-communication departments at Chennai colleges.
- **Job post:** native Chennai Tamil speaker with natural, clear Indian English. Must be comfortable reading Tanglish-style code-mixed text. Total commitment about 20 sessions of 3 h. Paid. Must accept an **open-source voice release** (state this up front to filter early).
- **Audition (paid, ~30 minutes):** 25 lines (8 EN, 8 TA, 6 CS, 3 long sentences) recorded in a real studio or a good home setup. Shortlist 3–5 candidates and score blind:
  - clarity
  - steadiness of pitch and pace
  - whether TA and EN sound like the same person
  - natural code-switching
  - neutrality of the Chennai accent
  - listener appeal (survey 5–10 target users)
- **Consent and contract (must be reviewed by a lawyer; open release raises the stakes):**
  - Explicit consent to recording, model training, and **public open-source release of a model that anyone can use to generate this voice, commercially, worldwide, indefinitely.** State clearly that the release **cannot be recalled once published**.
  - Use of the persona name "Sahithi" rather than her real name. Her anonymity is protected: her name is not in the model card unless she wants credit.
  - Compensation that reflects a perpetual release (typically a higher one-off buy-out).
  - Whether the **raw recordings** are released (default: **no**, only the weights).
  - The project's obligations: a responsible-use license on the weights, a ban on impersonation and fraud in the license terms, an audio watermark, and a takedown and contact process.
  - Signatures from both parties, with ID verification. Store the signed copy with the dataset records.
- **Studio:** rent a Chennai recording or dubbing studio by the hour (most consistent), or approve the talent's home studio after a test recording.
- **Exit:** a signed contract and an approved audition sample (you plus one native reviewer).

### Phase 3: Recording script and sessions (weeks 3–6, pilot first)
- **Script construction:**
  - Sentence sources: your own conversational and voice-agent dialogue writing, permissively licensed text (Wikipedia CC-BY-SA is fine to *read aloud*), and templated numbers, dates, currency, phone numbers and abbreviations.
  - **Greedy phoneme and diphone coverage selection** using frontend v0, including diphones **across language boundaries**.
  - Mix: about 40% TA, 35% EN, 25% CS. CS goes in both directions, with English words written in both Tamil script (மீட்டிங்) and Latin script (meeting).
  - **Tanglish needs no separate recordings.** The frontend converts Tanglish to Tamil script before synthesis, so Tamil-script recordings cover it. Prompts are always shown to the speaker in Tamil script.
  - Styles: statements, wh- and yes/no questions, exclamations, lists, short acknowledgements ("சரி", "Okay, got it"), and long sentences of 20+ words.
  - Target utterance length is 3–12 s, so 15 h ≈ 8–10k utterances. A native reviewer checks every TA and CS line before recording.
- **Setup:**
  - Treated room, noise floor below −60 dBFS.
  - Large-diaphragm condenser mic (e.g. AT2020 or Rode NT1), pop filter, audio interface.
  - Fixed 15–20 cm mic distance, and fixed gain for the whole project.
- **Format:** mono WAV, 48 kHz, 24-bit, **one file per prompt**, named by prompt ID, plus 2 s of room tone per session.
- **Recording app (`recording/app`):** a local web app that shows the prompt, a level meter, record, play back and retake controls, and saves `<prompt_id>.wav` plus a session log.
- **Sessions:**
  - 2.5–3 h each, yielding about 1 finished hour.
  - Breaks every 45 minutes.
  - About 10 "anchor" sentences re-recorded at every session to track voice drift.
- **Pilot:** **1 h (~500 prompts across all categories)** first. It runs through Phases 4–7 before the remaining sessions are booked.
- **Exit:** the pilot passes QC and the pilot model is intelligible in both languages.

### Phase 4: Data pipeline (`training/data_pipeline/`)
This pipeline runs locally on Windows (CPU is fine) or in Colab. It is idempotent, config-driven, and writes a report at each step.

1. **Ingest:** keep the raw files read-only and record a sha256 for each.
2. **Convert:** ffmpeg/soxr to 22.05 kHz, 16-bit, mono.
3. **Loudness and clipping:** pyloudnorm to about −23 LUFS with a −1 dBFS peak. Flag runs of clipped samples, and SNR below 35 dB measured against the room tone.
4. **Silence:** Silero VAD trims leading and trailing silence to 150 ms and flags internal pauses over 1.5 s.
5. **Segmentation:** only needed for long-form takes.
6. **Alignment check:** `torchaudio.pipelines.MMS_FA` CTC forced alignment on uromanized text. Items with low alignment scores, or ASR CER against the prompt above 10%, go to review.
7. **Transcript correction:** a human listens to every flagged item and fixes the text to match what was said. **A native reviewer checks all TA and CS items.**
8. **Duplicates:** exact text duplicates, then near-duplicate audio by embedding.
9. **Quality gates:** duration 1–15 s, characters per second within 3σ, SNR, clipping, and speaker-embedding similarity to the session centroid. Each item gets status `pass`, `review` or `reject`.
10. **Split:** stratified by language and length, 95/2/3. The test IDs are locked in `test_ids.lock` and are disjoint from the eval sentences.

**Manifest** (`data/manifests/{train,val,test}.jsonl`):
```json
{"id":"cs_s03_0412","audio":"data/processed/cs_s03_0412.wav","text":"நாளைக்கு மீட்டிங் 10 மணிக்கு.","text_norm":"நாளைக்கு மீட்டிங் பத்து மணிக்கு.","lang":"cs","spans":[["ta",0,8],["ta",9,16],["ta",17,30]],"speaker":"sahithi","duration":2.84,"sr":22050,"source":"studio_s03","quality":"pass","snr_db":41.2,"reviewer":"r2"}
```

**Local dev dataset:**
- Before Sahithi's data exists, a tiny public, permissively licensed set (e.g. 200 LJSpeech clips plus 200 clips from a CC-BY Tamil set such as IndicVoices-R) is used for code tests only and never ships.
- After the pilot, a 30-minute Sahithi subset is used.

**Exit:**
- Two builds from the same raw files produce identical manifests.
- At least 95% of pilot items pass.
- The QC report has been generated.

### Phase 5: Text frontend, including Tanglish (weeks 2–9, starts first)
All of this ships in the package and runs as pure Python plus tiny ONNX models. **No espeak-ng (GPL) and no system packages.**

**Pipeline:** `raw → NFC + cleanup → token LID {ta-script, en, tanglish, num, sym} → Tanglish→Tamil-script transliteration → per-span normalization → G2P (lexicon first, then rules or model) → phonemes with per-phoneme lang IDs + punctuation/pause tokens`

- **Token language ID:**
  - Tamil script is unambiguous (U+0B80–U+0BFF).
  - For Latin-script tokens, the system must decide between **English and Tanglish**. It uses a small char-n-gram classifier (logistic regression or a tiny CNN exported to ONNX) plus context features (neighbouring tokens' languages, an English lexicon hit, Tamil suffix patterns such as `-ku`, `-la`, `-nga`, `-ren`, `-dhu` and `-aa`).
  - Training data: Aksharantar romanized Tamil (CC0/CC-BY, verify) plus English word lists and sentences.
  - Ambiguous high-frequency words (e.g. "pal", "man", "seri") are resolved by sentence context and an override lexicon.
- **Tanglish → Tamil script:**
  - A small char-level seq2seq transformer (≤5M params) trained on Aksharantar Tamil pairs in Colab and exported to ONNX.
  - A curated Tanglish lexicon of the ~5k most frequent colloquial forms takes priority (e.g. "enna" → என்ன, "panren" → பண்றேன், "romba" → ரொம்ப).
  - The spoken-colloquial forms matter here: voice-agent LLMs write *spoken* Chennai Tamil in Tanglish, so both the transliteration and the recordings should cover colloquial forms.
  - IndicXlit (MIT) is a reference and baseline. It is not a runtime dependency because it depends on fairseq.
  - English words inside a Tanglish sentence stay English ("naan office ku late ah varen" → நான் [office] க்கு late ஆ வரேன், with the English tokens tagged EN).
- **English normalization:**
  - Numbers, ordinals, years, dates, times, ₹ and $ with the **lakh/crore system**, phone numbers, units, and acronym spell-out vs. word lists ("SBI" vs. "NASA").
  - Pure-Python implementation. NeMo text processing is a reference only, since it is too heavy as a dependency.
- **Tamil normalization:**
  - NFC; strip spurious zero-width characters (ZWJ/ZWNJ); standardize ஸ்ரீ.
  - Tamil numerals ௦–௯, ௰, ௱, ௲ → digits → words, with **case and sandhi** handling ("10 மணிக்கு" → "பத்து மணிக்கு", "5-ஆம் தேதி" → "ஐந்தாம் தேதி").
  - Dates, ரூபாய், abbreviations (டாக்., திரு.), and English acronyms read Tamil-style.
- **Tamil G2P (rule-based, Chennai standard):**
  - Stop allophony (voiceless when initial or geminate, voiced after nasals, voiced or fricative between vowels).
  - ச → [s] word-initially, [tʃː] when geminated (ச்ச), [ɲdʒ] for ஞ்ச.
  - ற / ற்ற / ன்ற (colloquial ன்ற → [nd]); ழ → [ɻ]; ண/ந/ன.
  - Kutriyalukaram; Grantha letters (ஜ ஷ ஸ ஹ க்ஷ).
  - Chennai colloquial features, such as final -ம் nasalization ("panrom").
  - Validated by a native speaker on 500+ words.
- **English G2P:** CMUdict (BSD-style license) plus an Indian-English and names lexicon. Out-of-vocabulary words go through a small ONNX G2P trained on CMUdict. The `g2p_en` package (Apache-2.0) is an acceptable interim option. Shared IPA symbols are used where the sounds really are the same.
- **Lexicon files** in `src/kural_tts/data/lexicon/`: names, places, brands and tech terms, all versioned. The lexicon overrides the rules.
- **Exit:**
  - ≥400 frontend unit tests pass on Windows, macOS and Linux.
  - Tamil G2P word accuracy is ≥98% on the native-reviewed set.
  - Normalization accuracy is ≥99% on 200 number, date and currency strings per language.
  - **Tanglish→Tamil word accuracy is ≥90% on a 500-sentence native-reviewed Tanglish set, and token LID F1 is ≥95%.**

### Phase 6: Model architecture (Kural-VITS)
Start from the `coqui-tts` VITS code. It is MPL-2.0: modified files stay MPL, and the package itself can be Apache-2.0. The **shipped runtime is ONNX only**, so there is no Coqui dependency at inference.

- **Training path:**
  1. Phoneme IDs plus **per-phoneme language embeddings** go into a transformer text encoder. This is our modification: Coqui uses a single per-sentence language ID.
  2. The encoder produces the prior (mean and variance per phoneme).
  3. The posterior encoder runs over the linear spectrogram.
  4. A normalizing flow maps z → z_p.
  5. **Monotonic Alignment Search (MAS)** finds the alignment and duration targets.
  6. The stochastic duration predictor learns the durations.
  7. A HiFi-GAN decoder turns a random latent slice into a waveform.
  8. Multi-period and multi-scale discriminators.
  9. Losses: mel L1 (×45), KL, duration, adversarial, and feature matching.
- **Inference path (what the package runs):** frontend → encoder → duration prediction (`noise_w`, `length_scale` = speed) → expand → sample (`noise_scale`) → inverse flow → decoder → waveform. The posterior encoder, MAS and discriminators are not used.
- **Speaker embedding:** a single slot. The API takes `voice="sahithi"`, so more voices can be added later without breaking changes.
- **Warm start:** English VITS weights, with the phoneme table extended for Tamil IPA (new symbols initialized from their nearest shared phone) and a fresh language-embedding table.
- **ONNX export:** export only the inference graph, at opset ≥17, with dynamic text length. Verify parity against PyTorch (mel L1 below a small epsilon).
- **Exit:** a design doc, a unit test that runs a forward and backward pass on a dummy batch, and an ONNX export of a randomly initialized model that passes the parity test on CPU.

### Phase 7: Training environment on Colab
- **Tier:**
  - **Colab Pro or Pro+** is recommended (L4 or A100, longer sessions, background execution).
  - The free T4 is fine for smoke tests and the 1 h pilot, but too slow for full training within session limits.
- **Colab-proofing:**
  - Mount Drive and copy the dataset tarball to local `/content` disk at session start, because Drive I/O is too slow for training.
  - Write a checkpoint to Drive every 2–5k steps (generator, discriminator, optimizer and scheduler state, step, RNG state) and keep the latest three.
  - `train.py --resume auto` picks up the newest checkpoint, so training survives disconnects.
  - Use bf16 on A100 or L4 (fp16 with a grad scaler on T4).
  - Log to TensorBoard on Drive (or W&B), with audio samples every N steps.
  - Each notebook pins its dependency versions in its first cell.
- **Local machine:** runs unit tests, the data pipeline, the frontend, and a **tiny-config overfit test on CPU** (10 utterances, a small model, a few hundred steps) to catch bugs before spending Colab time.
- **Smoke tests (must pass first):**
  - One full dataloader epoch.
  - Overfitting 10 utterances until reproduction is clear.
  - Kill and resume with no loss jump.
  - Stable mixed precision.
- **Rough budget:** warm-started VITS at 150–300k steps is about 10–30 A100 hours, or 3–6 sessions. The cost depends on Colab compute units and should be measured on the pilot.
- **Exit:** all smoke tests pass, and the pilot model (1 h of Sahithi data) produces intelligible TA and EN.

### Phase 8: Staged training
Every run is logged in `docs/experiments.md` with git sha, config hash, manifest hash and metrics.

| Stage | Data | Init | Exit |
|---|---|---|---|
| A: English | LJSpeech subset (or Sahithi EN) | VITS en | Pipeline proven. EN WER ≤ 3%. |
| B: Tamil | Sahithi TA | A + extended phones | TA CER ≤ 5%, native MOS ≥ 3.5 |
| C: Shared identity | Sahithi TA+EN, per-phoneme lang IDs | B | Speaker similarity of TA vs. EN output is close to that of real Sahithi TA vs. EN |
| D: Code-switch + Tanglish | + Sahithi CS. Tanglish eval goes through the frontend. | C | CS CER ≤ 7%, Tanglish eval CER ≤ 8%, no accent bleed at boundaries (human tally) |
| E: Refinement | all | best D | Controlled sweeps (lr, segment size, noise_w, duration predictor, loss weights). A change is kept only if validation and human A/B improve. |
| Challenger | same data | Matcha-TTS en | Matcha vs. VITS on the eval set, comparing MOS **and CPU RTF** |

### Phase 9: Evaluation (`training/eval/`)
- **Frozen eval set** `eval/sentences/eval_v1.tsv`: 150 EN, 150 TA, 150 CS and 100 Tanglish sentences, bucketed by difficulty (numbers, names, long, questions).
- **Automatic metrics:**
  - CER/WER (IndicConformer for TA, Whisper large-v3 for EN, both plus a human for CS).
  - Speaker similarity (ECAPA or WavLM-SV).
  - UTMOS (EN), duration ratio, F0 variance.
  - **CPU RTF and time to first audio on a reference laptop**, plus GPU RTF.
- **Human metrics:** MOS, SMOS, pronunciation-error tally, and CS-boundary naturalness, from ≥8 bilingual Chennai listeners. Use a blind, randomized web listening test with real recordings as the upper anchor.
- **Report:** `reports/eval_<ckpt>.md` compares each checkpoint against the previous best and the Phase 1 baselines.
- **v1 release bar:**
  - MOS ≥ 4.0 per category, SMOS ≥ 4.0.
  - TA CER ≤ 4%, EN WER ≤ 3%, CS CER ≤ 6%, Tanglish CER ≤ 7%.
  - No severe errors on the 50 most common agent phrases.
  - **CPU RTF < 0.3 on a 4-core laptop.**

### Phase 10: Inference optimization (package runtime)
- **ONNX:**
  - Graph optimization in onnxruntime.
  - Optional INT8 dynamic quantization of the decoder. Keep it only if MOS drops by at most 0.1.
  - fp16 model variant for GPU.
- **Execution providers:** CUDA when `kural-tts[gpu]` is installed; DirectML (Windows) and CoreML (macOS) as optional experiments.
- **Chunked streaming:** `KuralTTS.stream(text)` splits text at sentence and clause boundaries and yields PCM chunks. Time to first audio = frontend + first chunk.
  - Targets: **< 300 ms on CPU** and < 100 ms on GPU for a 10-word first sentence.
  - Documented honestly as sentence-level streaming.
- **Footprint:**
  - Target an install size under 50 MB for the wheel plus dependencies (excluding onnxruntime), and under 100 MB for the downloaded weights.
  - Cold start under 2 s after the first download.
- **Exit:** `reports/perf.md` covering Windows, macOS (Apple Silicon) and Linux on CPU, plus one GPU.

### Phase 11: Python package (main deliverable, replaces the hosted API)
- **Python API:**
  ```python
  from kural_tts import KuralTTS

  tts = KuralTTS(voice="sahithi", device="auto")  # downloads + caches weights on first use
  audio, sr = tts.synthesize("நாளைக்கு meeting 10 மணிக்கு. Don't be late!")
  tts.save("naalaikku office ku varen, ok va?", "out.wav")  # Tanglish works
  for chunk in tts.stream(long_text):  # sentence-chunked streaming
      play(chunk)
  ```
  Options: `lang="auto"|"ta"|"en"`, `speed`, `noise_scale`, `noise_w`, `sample_rate`, plus `phonemize(text)` for debugging pronunciation.
- **CLI:**
  - `kural-tts "text" -o out.wav`
  - `kural-tts --file input.txt`
  - `kural-tts --phonemes "text"`
  - `kural-tts download` (pre-fetch weights for offline use)
- **Weights:**
  - Hugging Face Hub repo with a pinned revision per package version and sha256 checks.
  - Respects `HF_HOME` and `KURAL_TTS_CACHE`, and supports `KURAL_TTS_OFFLINE=1` and a local model path.
- **Optional extras:**
  - `kural-tts[server]`: `kural-tts serve` starts a local FastAPI app with `POST /tts`, `WS /tts/stream`, `GET /health` and `GET /model-info` for voice agents and non-Python users. It binds to localhost by default and needs no API key.
  - `kural-tts[agents]`: a LiveKit Agents TTS plugin.
- **Quality:**
  - Typed public API, docstrings, and a docs site (mkdocs) with examples.
  - Semantic versioning, with a CHANGELOG.
  - `pip install kural-tts` is tested in CI on Windows, macOS and Linux for Python 3.10–3.13.
- **Exit:**
  - A clean virtualenv on each OS can `pip install kural-tts`, run the CLI and produce correct audio for the eval smoke set.
  - Published to TestPyPI first, then PyPI.

### Phase 12: Voice-agent integration
- Write a LiveKit Agents plugin using `KuralTTS.stream()` in-process, or the local `WS /tts/stream`. It must support **cancellation** for barge-in.
- Reference pipeline: Silero VAD / turn detection → Tamil-capable ASR → LLM (often emitting Tanglish or CS) → Kural-TTS → LiveKit.
- Timestamp every hop: end of speech, ASR final, LLM first token, TTS first audio, client playback.
- Test barge-in, silence, long responses, TA↔EN↔Tanglish switching, and 1/5/10 concurrent sessions on one machine.
- **Exit:** a latency report. Target the median from end of speech to first audio below 1.2 s, with TTS contributing < 300 ms on CPU.

### Phase 13: Open-source release
- **Licenses:**
  - **Code: Apache-2.0.** Modified Coqui files stay MPL-2.0 under `training/`, and that is documented in NOTICE.
  - **Weights:** a commercial-permitting **responsible-use license** (an OpenRAIL-M-style license with use restrictions: no impersonating the voice talent, no fraud or deceptive robocalls, no non-consensual content, disclosure of synthetic audio where the law requires it). This must be consistent with the talent contract and checked by a lawyer.
  - **Dataset:** not released by default.
- **Artifacts:**
  - GitHub repo with README, MODEL_CARD.md (data summary, languages, metrics, limitations, intended and prohibited uses, consent statement), CONTRIBUTING and a security policy.
  - Hugging Face model repo with ONNX weights and the model card.
  - A Hugging Face Space demo (CPU).
  - PyPI release through a GitHub Actions trusted publisher.
- **Watermark:** embed an inaudible watermark in generated audio by default (e.g. AudioSeal, MIT, if its runtime cost is acceptable on CPU; otherwise a lightweight spread-spectrum mark), and document it.
- **Telemetry:** none. The package never sends text or audio anywhere.
- **Technical report:** `docs/technical_report.md` covering architecture, data, frontend (including Tanglish), experiments, results and limitations.
- **Exit:** v1.0.0 is tagged and published on GitHub, PyPI and Hugging Face, and a stranger can follow the README from zero to audio in under 5 minutes.

---

## 4. Timeline (indicative: 1 engineer plus part-time native reviewers)

| Weeks | Track A (code and models) | Track B (voice and data) |
|---|---|---|
| 1–2 | Repo and package scaffold, Phase 1 baselines on Colab | Talent job post, auditions |
| 2–6 | Frontend v0→v1 (normalization, Tamil G2P, Tanglish LID and transliteration), Phase 6 design + ONNX parity | Contract, recording script, studio booking, **1 h pilot** |
| 6–8 | Phases 4 and 7 on the pilot, Stage A/B on Colab | Full recording (~4 sessions per week) |
| 8–12 | Stages B→D as data arrives, Matcha challenger | Recording done (15 h), transcript review |
| 12–14 | Stage E, Phase 9 eval and listening tests, Phase 10 optimization | – |
| 14–17 | Phase 11 package, Phase 12 agent integration, Phase 13 release and report | – |

---

## 5. Key risks and mitigations

| Risk | Mitigation |
|---|---|
| The talent hesitates over an irrevocable open release | Disclose the open release in the job post, pay a fair buy-out, protect her anonymity, and add the responsible-use license and watermark. |
| A license problem in the warm-start checkpoint or a frontend dependency | Verify in Phase 1 and run a dependency license scan (`pip-licenses`) in CI. Fallback: train from scratch on Sahithi + LJSpeech. |
| Tamil G2P errors | Rules plus a lexicon, native review and unit tests. Also compare a grapheme-input model in Stage B. |
| Tanglish is ambiguous (inconsistent spellings, English/Tamil homographs) | A colloquial Tanglish lexicon, a context-aware LID, and a native-reviewed eval set. Document the limitations. Users can force `lang=` or write Tamil script. |
| Colab disconnects or quota limits | Auto-resume checkpoints on Drive and a local-disk data copy. Budget Pro compute units, measured on the pilot. |
| CPU speed is too slow on low-end machines | ONNX, INT8 quantization, a smaller decoder variant ("kural-tts-small"), and sentence-chunked streaming. |
| Voice drift across sessions | Anchor sentences, monitoring of the speaker-embedding centroid, and fixed gain and distance. |
| Accent bleed in code-switching | Per-phoneme language IDs, CS recording coverage, and a human boundary tally. |

---

## 6. Open questions (non-blocking)
1. Recording budget: a rented Chennai studio (more consistent, costs more) or the talent's home studio?
2. Colab tier: is Pro or Pro+ approved?
3. Should Sahithi receive public credit in the model card, or stay anonymous?

---

## 7. Immediate next steps
1. **Scaffold** `pyproject.toml` (`src/kural_tts`, extras), the `training/` and `tests/` layout, `.gitignore`, pre-commit, and multi-OS CI.
2. **Write the feasibility set** `eval/sentences/feasibility_v1.tsv` (EN, TA, CS, Tanglish) and get it native-reviewed.
3. **Write baseline runners** and `notebooks/01_baselines.ipynb` for Colab. Run them, score them, and write `reports/feasibility.md` and ADR-001.
4. **Start frontend v0:** Tamil normalization and numbers, script-based LID, and a lexicon-only Tanglish transliterator, with tests.
5. **Track B:** post the talent call and get the lawyer to draft the consent and open-release contract.

---

## Sources (research, 2026-11-27)
- MMS-TTS Tamil (CC-BY-NC-4.0): https://huggingface.co/facebook/mms-tts-tam
- Indic Parler-TTS (Apache-2.0): https://huggingface.co/ai4bharat/indic-parler-tts
- IndicF5 (AIKosh, MIT tag): https://aikosh.indiaai.gov.in/home/models/details/aibharat_indicf5.html
- Dhee-Indic-F5 (argues for non-commercial inheritance): https://huggingface.co/dheeyantra/dhee-indic-f5
- Tamil IndicF5 fine-tune: https://huggingface.co/ragunath-ravi/indic5
- F5 Indic vocabulary extension: https://arxiv.org/pdf/2505.20693
- svara-TTS v1: https://huggingface.co/kenpath/svara-tts-v1, https://huggingface.co/blog/kenpath/svara-tts-open-multilingual-speech-for-india
- svara cloning caveat: https://blaizzy.github.io/mlx-audio/models/tts/svara/
- Chatterbox: https://huggingface.co/ResembleAI/chatterbox/blob/main/README.md, Indic LoRA: https://huggingface.co/reenigne314/chatterbox-indic-lora
- F5-TTS OpenBible Tamil: https://huggingface.co/multilingual-tts/F5-TTS-OpenBible-Tamil
- IndicVoices-R (CC-BY-4.0): https://ai4bharat.iitm.ac.in/indicvoices_r
- IIT Madras IndicTTS license: https://www.iitm.ac.in/donlab/indictts/database
- Aksharantar (CC0 mined / CC-BY manual; an older card revision said NC, so verify): https://huggingface.co/datasets/ai4bharat/Aksharantar, paper https://arxiv.org/abs/2205.03018
- Matcha-TTS ONNX export and inference: https://www.mintlify.com/shivammehta25/Matcha-TTS/advanced/onnx-export, https://www.mintlify.com/shivammehta25/Matcha-TTS/advanced/onnx-inference
- Matcha VCTK ONNX (weights inherit dataset license): https://huggingface.co/Akjava/matcha-tts_vctk-onnx
- Code-switching accent control: https://arxiv.org/pdf/2609.01016
- Cross-lingual Indic speaker adaptation: https://chandastts.github.io/
