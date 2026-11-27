"""Spectrogram features for Kural-VITS training (matches the VITS reference STFT setup)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 22050
    filter_length: int = 1024  # n_fft
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float | None = None

    @property
    def spec_channels(self) -> int:
        return self.filter_length // 2 + 1


@lru_cache(maxsize=8)
def _hann(win_length: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    return torch.hann_window(win_length, device=device, dtype=dtype)


@lru_cache(maxsize=8)
def _mel_basis(cfg: AudioConfig, device: str, dtype: torch.dtype) -> torch.Tensor:
    import librosa

    mel = librosa.filters.mel(
        sr=cfg.sample_rate, n_fft=cfg.filter_length, n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax
    )
    return torch.from_numpy(mel).to(device=device, dtype=dtype)


def spectrogram(wav: torch.Tensor, cfg: AudioConfig) -> torch.Tensor:
    """Linear magnitude spectrogram. wav: [B, T] in [-1, 1] -> [B, n_fft//2+1, frames].

    Frames = T // hop_length exactly (reflect padding of (n_fft - hop) / 2 on both sides).
    """
    pad = (cfg.filter_length - cfg.hop_length) // 2
    wav = torch.nn.functional.pad(wav.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    spec = torch.stft(
        wav.float(),
        cfg.filter_length,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=_hann(cfg.win_length, str(wav.device), torch.float32),
        center=False,
        return_complex=True,
    )
    return torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)


def spec_to_mel(spec: torch.Tensor, cfg: AudioConfig) -> torch.Tensor:
    """Linear spectrogram -> log-mel (natural log, clamped)."""
    mel = torch.matmul(_mel_basis(cfg, str(spec.device), spec.dtype), spec)
    return torch.log(torch.clamp(mel, min=1e-5))


def mel_spectrogram(wav: torch.Tensor, cfg: AudioConfig) -> torch.Tensor:
    return spec_to_mel(spectrogram(wav, cfg), cfg)
