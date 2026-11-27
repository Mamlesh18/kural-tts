"""Public Python API."""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import numpy as np

LangMode = Literal["auto", "ta", "en"]
Device = Literal["auto", "cpu", "cuda"]

SUPPORTED_VOICES = ("sahithi",)
SENTENCE_PAUSE_SEC = 0.12  # silence inserted between sentences in synthesize()
MAX_CHARS = 5000


class ModelNotAvailableError(RuntimeError):
    """Raised when no trained Kural-TTS voice can be found or downloaded."""


class KuralTTS:
    """Text-to-speech synthesizer.

    Example::

        tts = KuralTTS(voice="sahithi")
        audio, sr = tts.synthesize("நாளைக்கு meeting 10 மணிக்கு.")

    The voice is loaded on first use, from ``model_path`` (or the ``KURAL_TTS_MODEL`` environment
    variable) if given, otherwise downloaded from the Hugging Face Hub and cached.
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
        env_path = os.environ.get("KURAL_TTS_MODEL")
        chosen = model_path if model_path is not None else env_path
        self.model_path = Path(chosen) if chosen else None
        self._voice = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ model loading
    def _load(self):
        if self._voice is None:
            with self._lock:
                if self._voice is None:
                    from kural_tts.runtime import OnnxVoice

                    self._voice = OnnxVoice(self._resolve_model_dir(), self.device)
        return self._voice

    def _resolve_model_dir(self) -> Path:
        if self.model_path is not None:
            if not (self.model_path / "model.onnx").exists():
                raise ModelNotAvailableError(f"no model.onnx in {self.model_path}")
            return self.model_path
        from kural_tts import hub

        try:
            return hub.download_model()
        except Exception as e:  # not released yet, offline, or network error
            raise ModelNotAvailableError(
                f"Could not load the {self.voice!r} voice: {e}. Pass model_path= or set "
                "KURAL_TTS_MODEL to a directory containing model.onnx and config.json."
            ) from e

    @property
    def sample_rate(self) -> int:
        return self._load().sample_rate

    @property
    def providers(self) -> list[str]:
        """ONNX Runtime execution providers in use (e.g. CUDA, CPU)."""
        return self._load().providers

    # ------------------------------------------------------------------ synthesis
    def _synth_sentence(
        self, sentence: str, lang: LangMode, speed: float, noise_scale: float | None
    ) -> np.ndarray:
        from kural_tts.text.g2p import text_to_ids

        ids, langs = text_to_ids(sentence, lang)
        if len(ids) <= 2:  # only BOS/EOS: nothing speakable (e.g. lone punctuation)
            return np.zeros(0, dtype=np.float32)
        return self._load().run(ids, langs, noise_scale=noise_scale, length_scale=1.0 / speed)

    def synthesize(
        self,
        text: str,
        lang: LangMode = "auto",
        speed: float = 1.0,
        noise_scale: float | None = None,
    ) -> tuple[np.ndarray, int]:
        """Synthesize ``text`` and return ``(float32 mono waveform, sample_rate)``.

        ``speed`` > 1 speaks faster. ``noise_scale`` controls expressiveness (default comes from
        the voice; 0 gives the most stable, flattest reading).
        """
        _validate_text(text)
        if speed <= 0:
            raise ValueError("speed must be > 0")
        from kural_tts.text.segment import split_sentences

        voice = self._load()
        pause = np.zeros(int(SENTENCE_PAUSE_SEC * voice.sample_rate), dtype=np.float32)
        pieces: list[np.ndarray] = []
        for sentence in split_sentences(text):
            audio = self._synth_sentence(sentence, lang, speed, noise_scale)
            if audio.size:
                if pieces:
                    pieces.append(pause)
                pieces.append(audio)
        out = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        return out, voice.sample_rate

    def stream(
        self,
        text: str,
        lang: LangMode = "auto",
        speed: float = 1.0,
        noise_scale: float | None = None,
    ) -> Iterator[np.ndarray]:
        """Yield audio sentence by sentence (sentence-level streaming, not intra-sentence).

        The first chunk is ready after the first sentence is synthesized, which keeps time to
        first audio low for voice agents.
        """
        _validate_text(text)
        from kural_tts.text.segment import split_sentences

        for sentence in split_sentences(text):
            audio = self._synth_sentence(sentence, lang, speed, noise_scale)
            if audio.size:
                yield audio

    def save(self, text: str, path: str | Path, **kwargs) -> Path:
        """Synthesize ``text`` and write a 16-bit PCM WAV file."""
        import soundfile as sf

        audio, sr = self.synthesize(text, **kwargs)
        path = Path(path)
        sf.write(path, audio, sr, subtype="PCM_16")
        return path

    # ------------------------------------------------------------------ frontend only
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


def _validate_text(text: str) -> None:
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    if not text.strip():
        raise ValueError("text is empty")
    if len(text) > MAX_CHARS:
        raise ValueError(f"text longer than {MAX_CHARS} characters; split it or use stream()")
