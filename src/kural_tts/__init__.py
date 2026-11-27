"""Kural-TTS: bilingual Tamil-English (and Tanglish) text-to-speech with the Sahithi voice."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("kural-tts")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "0.0.0+unknown"

from kural_tts.api import KuralTTS

__all__ = ["KuralTTS", "__version__"]
