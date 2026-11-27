"""Audio loading, resampling, trimming, loudness normalization and quality measurements."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import soundfile as sf

FRAME_SEC = 0.02  # analysis frame for energy-based measurements


def load_mono(path, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Read any WAV/FLAC (any bit depth/channels) -> float32 mono, optionally resampled."""
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if target_sr is not None and sr != target_sr:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")
        sr = target_sr
    return wav.astype(np.float32), sr


def _frame_db(wav: np.ndarray, sr: int) -> np.ndarray:
    hop = max(1, int(FRAME_SEC * sr))
    n = len(wav) // hop
    if n == 0:
        return np.array([-120.0])
    frames = wav[: n * hop].reshape(n, hop)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def trim_silence(
    wav: np.ndarray, sr: int, top_db: float = 40.0, pad_sec: float = 0.15
) -> np.ndarray:
    """Trim leading/trailing audio quieter than (peak frame - top_db), keeping pad_sec margins."""
    db = _frame_db(wav, sr)
    voiced = np.flatnonzero(db > db.max() - top_db)
    if voiced.size == 0:
        return wav[:0]
    hop = int(FRAME_SEC * sr)
    pad = int(pad_sec * sr)
    start = max(0, voiced[0] * hop - pad)
    end = min(len(wav), (voiced[-1] + 1) * hop + pad)
    return wav[start:end]


def loudness_lufs(wav: np.ndarray, sr: int) -> float:
    import pyloudnorm

    if len(wav) < int(0.4 * sr):  # pyloudnorm needs at least one 400 ms block
        return float("-inf")
    return float(pyloudnorm.Meter(sr).integrated_loudness(wav.astype(np.float64)))


def normalize_loudness(
    wav: np.ndarray, sr: int, target_lufs: float = -23.0, peak_dbfs: float = -1.0
) -> np.ndarray:
    """Gain to target integrated loudness, then cap the peak (no clipping introduced)."""
    current = loudness_lufs(wav, sr)
    if np.isfinite(current):
        wav = wav * (10 ** ((target_lufs - current) / 20))
    peak = np.abs(wav).max() if wav.size else 0.0
    limit = 10 ** (peak_dbfs / 20)
    if peak > limit:
        wav = wav * (limit / peak)
    return wav.astype(np.float32)


def clipping_runs(wav: np.ndarray, threshold: float = 0.999, min_run: int = 3) -> int:
    """Number of runs of >= min_run consecutive samples at full scale."""
    hot = np.abs(wav) >= threshold
    if not hot.any():
        return 0
    edges = np.diff(np.concatenate([[0], hot.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return int(np.sum((ends - starts) >= min_run))


def estimate_snr_db(wav: np.ndarray, sr: int) -> float:
    """Speech level (loudest 50% of frames) vs noise floor (quietest 10% of frames)."""
    db = np.sort(_frame_db(wav, sr))
    if db.size < 10:
        return 0.0
    noise = np.mean(db[: max(1, db.size // 10)])
    speech = np.mean(db[db.size // 2 :])
    return float(speech - noise)


def longest_pause_sec(wav: np.ndarray, sr: int, top_db: float = 40.0) -> float:
    db = _frame_db(wav, sr)
    silent = db <= db.max() - top_db
    longest = run = 0
    for s in silent:
        run = run + 1 if s else 0
        longest = max(longest, run)
    return longest * FRAME_SEC


@dataclass
class AudioStats:
    duration: float
    peak_dbfs: float
    loudness_lufs: float
    clip_runs: int
    snr_db: float
    longest_pause: float

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def measure(wav: np.ndarray, sr: int) -> AudioStats:
    peak = float(np.abs(wav).max()) if wav.size else 0.0
    return AudioStats(
        duration=len(wav) / sr,
        peak_dbfs=20 * np.log10(peak + 1e-12),
        loudness_lufs=loudness_lufs(wav, sr),
        clip_runs=clipping_runs(wav),
        snr_db=estimate_snr_db(wav, sr),
        longest_pause=longest_pause_sec(wav, sr),
    )
