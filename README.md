# Kural-TTS

Open-source text-to-speech for **Tamil, English, Tamil-English code-switching and Tanglish**, in one
consistent female voice: **Sahithi**. Runs locally on CPU (ONNX Runtime) or GPU.

> **Status: pre-alpha.** The full pipeline is implemented and tested end to end: text frontend,
> model, training, ONNX export, runtime, server and LiveKit plugin. **The Sahithi voice has not been
> recorded or trained yet**, so `KuralTTS()` has nothing to download. Until then, train your own
> with the runbook or pass a local `model_path`. See [docs/PLAN.md](docs/PLAN.md).

## Planned usage

```bash
pip install kural-tts
kural-tts "நாளைக்கு meeting 10 மணிக்கு. Don't be late!" -o out.wav
```

```python
from kural_tts import KuralTTS

tts = KuralTTS(voice="sahithi")
audio, sr = tts.synthesize("naalaikku office ku varen, ok va?")  # Tanglish works too
for chunk in tts.stream(long_text):  # sentence-level streaming
    play(chunk)
```

## Try the text frontend today

The model isn't trained yet, but the bilingual text frontend already works. It handles numbers,
currency, dates, abbreviations, code-switching and Tanglish:

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

Inspect exactly what the model will receive, as phonemes with a language tag on each:

```bash
uv run kural-tts "Sir, unga account-la KYC update pannunga" --phonemes
[en] s ˈ ɝ , [ta] u ŋ ɡ a | [en] ə k ˈ aʊ n t [ta] l a | [en] k ˈ eɪ | w ˈ aɪ | s ˈ i | ...
```

See [docs/frontend.md](docs/frontend.md) and [docs/g2p.md](docs/g2p.md) for details and known limitations.

## Documentation

| Doc | What's in it |
|---|---|
| [docs/PLAN.md](docs/PLAN.md) | Research, decisions, phased plan, progress |
| [docs/training.md](docs/training.md) | **Runbook**: script → record → dataset → train on Colab → export → evaluate → publish |
| [recording/README.md](recording/README.md) | Sentence pools, studio setup, session workflow |
| [docs/frontend.md](docs/frontend.md), [docs/g2p.md](docs/g2p.md) | Text normalization, Tanglish, pronunciation |
| [docs/voice-agents.md](docs/voice-agents.md) | In-process streaming, LiveKit plugin, local server |

## Repository layout

| Path | Contents |
|---|---|
| `src/kural_tts/` | The installable package: text frontend, ONNX runtime, CLI, server, LiveKit plugin |
| `training/` | Kural-VITS model, data pipeline, trainer, ONNX export (not shipped) |
| `notebooks/` | Google Colab notebooks for baselines, training, evaluation |
| `recording/` | Recording scripts, recording app, consent templates |
| `eval/sentences/` | Frozen evaluation sentence sets (never used for training) |
| `docs/` | Plan, architecture decisions, model card, experiment log |

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
```

## License

Code: Apache-2.0. Model weights will be released under a separate responsible-use license that
permits commercial use. Impersonation, fraud and deceptive use of the voice are prohibited.
