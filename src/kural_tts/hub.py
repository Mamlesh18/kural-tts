"""Model download and caching from the Hugging Face Hub."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

# Filled in at first model release. Pinned per package version for reproducibility.
HF_REPO_ID = "kural-tts/kural-tts-sahithi"
HF_REVISION: str | None = None
MODEL_FILES: dict[str, str] = {}  # filename -> sha256


def cache_dir() -> Path:
    override = os.environ.get("KURAL_TTS_CACHE")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "kural-tts"


def is_offline() -> bool:
    return os.environ.get("KURAL_TTS_OFFLINE", "0") == "1"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def download_model() -> Path:
    """Download the released model files and verify their checksums. Returns the model directory."""
    if HF_REVISION is None or not MODEL_FILES:
        raise RuntimeError("No Kural-TTS model has been released yet.")
    from huggingface_hub import snapshot_download

    local = Path(
        snapshot_download(
            repo_id=HF_REPO_ID,
            revision=HF_REVISION,
            cache_dir=cache_dir(),
            allow_patterns=list(MODEL_FILES),
            local_files_only=is_offline(),
        )
    )
    for name, expected in MODEL_FILES.items():
        actual = sha256_file(local / name)
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for {name}: expected {expected}, got {actual}")
    return local
