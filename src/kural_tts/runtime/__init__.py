"""Inference runtimes for exported voices."""

from kural_tts.runtime.onnx_voice import IncompatibleModelError, OnnxVoice

__all__ = ["IncompatibleModelError", "OnnxVoice"]
