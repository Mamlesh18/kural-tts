# Kural-TTS

Open-source text-to-speech for **Tamil, English, Tamil-English code-switching and Tanglish**, in one
consistent female voice: **Sahithi**. Runs locally on CPU (ONNX Runtime) or GPU.

> **Status: pre-alpha.** No model has been released yet. See [docs/PLAN.md](docs/PLAN.md) for the roadmap.

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

## Repository layout

| Path | Contents |
|---|---|
| `src/kural_tts/` | The installable package (runtime + text frontend) |
| `training/` | Data pipeline, model training, evaluation, ONNX export (not shipped) |
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
