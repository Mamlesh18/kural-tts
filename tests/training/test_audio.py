import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("librosa")

from training.audio import AudioConfig, mel_spectrogram, spectrogram  # noqa: E402


def test_frame_count_matches_hop():
    cfg = AudioConfig()
    wav = torch.randn(2, cfg.hop_length * 50).clamp(-1, 1)
    spec = spectrogram(wav, cfg)
    assert spec.shape == (2, cfg.spec_channels, 50)
    mel = mel_spectrogram(wav, cfg)
    assert mel.shape == (2, cfg.n_mels, 50)
    assert torch.isfinite(mel).all()


def test_sine_energy_lands_in_expected_bin():
    cfg = AudioConfig()
    t = torch.arange(cfg.sample_rate) / cfg.sample_rate
    wav = (0.5 * torch.sin(2 * torch.pi * 1000 * t)).unsqueeze(0)
    spec = spectrogram(wav, cfg)
    peak_bin = spec.mean(-1).argmax().item()
    expected = round(1000 * cfg.filter_length / cfg.sample_rate)
    assert abs(peak_bin - expected) <= 1
