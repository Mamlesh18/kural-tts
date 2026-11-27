"""Public Python API.

The model is not trained yet, so ``synthesize`` raises ``ModelNotAvailableError``. The signature is
fixed now so the rest of the stack (CLI, server, agent plugin, tests) can be built against it.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import numpy as np

LangMode = Literal["auto", "ta", "en"]
Device = Literal["auto", "cpu", "cuda"]

SUPPORTED_VOICES = ("sahithi",)


class ModelNotAvailableError(RuntimeError):
    """Raised when no trained Kural-TTS model is available yet."""


class KuralTTS:
    """Text-to-speech synthesizer.

    Example::

        tts = KuralTTS(voice="sahithi")
        audio, sr = tts.synthesize("நாளைக்கு meeting 10 மணிக்கு.")
    """

    def __init__(
        self,
        voice: str = "sahithi",
        device: Device = "auto",
        model_path: str | Path | None = None,
    ) -> None:
        if voice not in SUPPORTED_VOICES:
            raise ValueError(f"Unknown voice {voice!r}; available: {', '.join(SUPPORTED_VOICES)}")
        self.voice = voice
        self.device = device
        self.model_path = Path(model_path) if model_path is not None else None
        self._session = None  # onnxruntime.InferenceSession, loaded lazily

    @property
    def sample_rate(self) -> int:
        return 22050

    def synthesize(
        self,
        text: str,
        lang: LangMode = "auto",
        speed: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
    ) -> tuple[np.ndarray, int]:
        """Synthesize ``text`` and return ``(float32 mono waveform, sample_rate)``."""
        _validate_text(text)
        if speed <= 0:
            raise ValueError("speed must be > 0")
        raise ModelNotAvailableError(
            "No Kural-TTS model has been released yet. Training is in progress; see docs/PLAN.md."
        )

    def normalize(self, text: str, lang: LangMode = "auto") -> list:
        """Run the text frontend only: returns language-tagged, verbalized ``Span`` objects.

        Works without a model, so it is useful for checking how text will be read.
        """
        _validate_text(text)
        from kural_tts.text.frontend import normalize

        return normalize(text, lang)

    def phonemize(self, text: str, lang: LangMode = "auto") -> list[tuple[str, str]]:
        """Return the model input as ``[(phoneme, lang), ...]``. Works without a model."""
        _validate_text(text)
        from kural_tts.text.g2p import phonemize

        return phonemize(text, lang)

    def stream(self, text: str, lang: LangMode = "auto", **kwargs) -> Iterator[np.ndarray]:
        """Yield audio sentence by sentence (sentence-level streaming, not intra-sentence)."""
        _validate_text(text)
        from kural_tts.text.segment import split_sentences

        for sentence in split_sentences(text):
            audio, _ = self.synthesize(sentence, lang=lang, **kwargs)
            yield audio

    def save(self, text: str, path: str | Path, **kwargs) -> Path:
        """Synthesize ``text`` and write a 16-bit PCM WAV file."""
        import soundfile as sf

        audio, sr = self.synthesize(text, **kwargs)
        path = Path(path)
        sf.write(path, audio, sr, subtype="PCM_16")
        return path


MAX_CHARS = 5000


def _validate_text(text: str) -> None:
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    if not text.strip():
        raise ValueError("text is empty")
    if len(text) > MAX_CHARS:
        raise ValueError(f"text longer than {MAX_CHARS} characters; split it or use stream()")
