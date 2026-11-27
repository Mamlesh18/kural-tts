---
language: [ta, en]
library_name: kural-tts
pipeline_tag: text-to-speech
tags: [tts, tamil, tanglish, code-switching, vits, onnx]
license: other  # responsible-use license: see "License" below (fill in before release)
---

# Kural-TTS: Sahithi voice

A bilingual **Tamil + English** text-to-speech voice. It handles Tamil-English code-switching and
**Tanglish** (romanized Tamil) input. It runs locally on CPU or GPU through the `kural-tts` Python package.

```bash
pip install kural-tts
kural-tts "நாளைக்கு meeting 10 மணிக்கு, ok va?" -o out.wav
```

## Model

| | |
|---|---|
| Architecture | Kural-VITS: VITS with per-phoneme language embeddings, deterministic duration predictor, HiFi-GAN decoder |
| Input | IPA phonemes (Tamil rule-based G2P, CMUdict English) plus a language id per phoneme |
| Output | 16-bit mono, {{sample_rate}} Hz |
| Format | ONNX (`model.onnx`) + `config.json` |
| Training step | {{training_step}} |
| Streaming | Sentence-level (each sentence is generated whole) |

## Training data

About `[N]` hours of studio recordings from **one consenting voice talent** ("Sahithi" is a persona
name): `[x]` h Tamil, `[y]` h English, `[z]` h code-switched. The recordings are **not** released.
The talent consented to the training and to the public, commercial release of this model.

## Evaluation (frozen eval set, see `reports/`)

| Metric | Tamil | English | Code-switched | Tanglish |
|---|---|---|---|---|
| MOS (native listeners) | | | | |
| CER / WER (ASR) | | | | |
| Speaker similarity | | | | |

CPU real-time factor (4-core laptop): `[ ]`. GPU real-time factor: `[ ]`.

## Intended use

Voice agents, accessibility, reading assistants and other applications needing natural Tamil/English speech.

## Prohibited use

- Impersonating the voice talent or any real person, or presenting output as their statements.
- Fraud, scams, phishing, and deceptive or unsolicited robocalls.
- Sexual content, harassment, and political campaigning or election content.
- Any use that hides synthetic speech where the law requires disclosure.

## Limitations

- Tamil pronunciation comes from rule-based G2P. Rare words, names and dialect forms can be mispronounced.
- Tanglish spelling is highly variable. Uncommon spellings may be read incorrectly, so write Tamil
  script for critical text.
- One speaker, one speaking style; there is no emotion control.

## License

`[Responsible-use license, e.g. OpenRAIL-M-style, permitting commercial use with the restrictions
above. Must match the talent agreement.]`

## Contact and misuse reports

`[email / issue tracker]`
