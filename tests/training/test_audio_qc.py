import numpy as np
import pytest

pytest.importorskip("pyloudnorm")
pytest.importorskip("librosa")

import soundfile as sf

from training.data_pipeline.audio_qc import (
    clipping_runs,
    estimate_snr_db,
    load_mono,
    longest_pause_sec,
    loudness_lufs,
    measure,
    normalize_loudness,
    trim_silence,
)

SR = 22050


def tone(seconds, amp=0.3, freq=220, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds, noise=1e-4, sr=SR, seed=0):
    return (noise * np.random.default_rng(seed).standard_normal(int(seconds * sr))).astype(
        np.float32
    )


def test_load_mono_resamples_and_downmixes(tmp_path):
    stereo = np.stack([tone(1.0, sr=48000), tone(1.0, sr=48000)], axis=1)
    sf.write(tmp_path / "s.wav", stereo, 48000, subtype="PCM_24")
    wav, sr = load_mono(tmp_path / "s.wav", target_sr=SR)
    assert sr == SR and wav.ndim == 1 and abs(len(wav) - SR) <= 2


def test_trim_keeps_padding():
    wav = np.concatenate([silence(1.0), tone(1.0), silence(1.0)])
    trimmed = trim_silence(wav, SR, pad_sec=0.15)
    assert 1.2 <= len(trimmed) / SR <= 1.4


def test_loudness_normalization_hits_target_without_clipping():
    wav = normalize_loudness(tone(2.0, amp=0.05), SR, target_lufs=-23.0)
    assert abs(loudness_lufs(wav, SR) - (-23.0)) < 0.5
    loud = normalize_loudness(tone(2.0, amp=0.05), SR, target_lufs=0.0, peak_dbfs=-1.0)
    assert np.abs(loud).max() <= 10 ** (-1 / 20) + 1e-6


def test_clipping_snr_and_pause_detection():
    clean = tone(1.0)
    clipped = np.clip(clean * 10, -1, 1)
    assert clipping_runs(clean) == 0 and clipping_runs(clipped) > 10
    noisy = np.concatenate([silence(0.5, noise=0.01), tone(1.0), silence(0.5, noise=0.01)])
    quiet = np.concatenate([silence(0.5), tone(1.0), silence(0.5)])
    assert estimate_snr_db(quiet, SR) > estimate_snr_db(noisy, SR) + 20
    gap = np.concatenate([tone(0.5), silence(2.0), tone(0.5)])
    assert 1.9 <= longest_pause_sec(gap, SR) <= 2.1


def test_measure_reports_all_fields():
    stats = measure(tone(1.0), SR).to_dict()
    assert set(stats) == {"duration", "peak_dbfs", "loudness_lufs", "clip_runs", "snr_db",
                          "longest_pause"}  # fmt: skip
    assert stats["duration"] == 1.0
