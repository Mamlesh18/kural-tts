"""ONNX Runtime wrapper around an exported Kural-TTS voice bundle (model.onnx + config.json)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kural_tts.text.symbols import SYMBOLS_VERSION

MODEL_FILE = "model.onnx"
CONFIG_FILE = "config.json"
SUPPORTED_FORMATS = ("kural-tts-onnx/1",)


class IncompatibleModelError(RuntimeError):
    """The voice bundle was exported for a different phoneme inventory or format."""


def _providers(device: str) -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    if device in ("auto", "cuda") and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "cuda":
        raise RuntimeError("CUDA requested but onnxruntime-gpu is not installed (pip install "
                           "'kural-tts[gpu]')")  # fmt: skip
    return ["CPUExecutionProvider"]


class OnnxVoice:
    def __init__(self, model_dir: str | Path, device: str = "auto"):
        import onnxruntime as ort

        model_dir = Path(model_dir)
        self.config = json.loads((model_dir / CONFIG_FILE).read_text(encoding="utf-8"))
        fmt = self.config.get("format")
        if fmt not in SUPPORTED_FORMATS:
            raise IncompatibleModelError(f"unsupported bundle format {fmt!r}")
        if self.config.get("symbols_version") != SYMBOLS_VERSION:
            raise IncompatibleModelError(
                f"voice uses symbols v{self.config.get('symbols_version')}, this kural-tts "
                f"expects v{SYMBOLS_VERSION}; upgrade kural-tts or download a matching voice"
            )
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_dir / MODEL_FILE), sess_options=options, providers=_providers(device)
        )
        self.sample_rate: int = int(self.config["sample_rate"])
        defaults = self.config.get("inference", {})
        self.default_noise_scale = float(defaults.get("noise_scale", 0.667))
        self.default_length_scale = float(defaults.get("length_scale", 1.0))

    @property
    def providers(self) -> list[str]:
        return self.session.get_providers()

    def run(
        self,
        ids: list[int],
        langs: list[int],
        noise_scale: float | None = None,
        length_scale: float | None = None,
    ) -> np.ndarray:
        """Phoneme/language ids (with BOS/EOS) -> float32 mono waveform."""
        x = np.asarray([ids], dtype=np.int64)
        scales = np.asarray(
            [
                self.default_noise_scale if noise_scale is None else noise_scale,
                self.default_length_scale if length_scale is None else length_scale,
            ],
            dtype=np.float32,
        )
        (audio,) = self.session.run(
            None,
            {
                "input": x,
                "input_lengths": np.asarray([len(ids)], dtype=np.int64),
                "lang": np.asarray([langs], dtype=np.int64),
                "scales": scales,
            },
        )
        return audio[0].astype(np.float32, copy=False)
