"""Training configuration: YAML file -> typed dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

from kural_tts.text.symbols import LANGS, SYMBOLS, SYMBOLS_VERSION
from training.audio import AudioConfig
from training.models.vits.synthesizer import ModelConfig


@dataclass
class DataConfig:
    train_manifest: str = "data/manifests/train.jsonl"
    val_manifest: str = "data/manifests/val.jsonl"
    audio_root: str = "."  # manifest audio paths are relative to this
    text_field: str = "text"
    min_phonemes: int = 2
    max_phonemes: int = 400
    max_seconds: float = 15.0


@dataclass
class TrainConfig:
    batch_size: int = 16
    learning_rate: float = 2e-4
    betas: tuple[float, float] = (0.8, 0.99)
    eps: float = 1e-9
    lr_decay: float = 0.999875  # per epoch
    segment_size: int = 8192  # waveform samples per decoder training window
    c_mel: float = 45.0
    c_kl: float = 1.0
    epochs: int = 10000
    max_steps: int | None = None
    precision: str = "auto"  # auto | bf16 | fp16 | fp32
    num_workers: int = 2
    seed: int = 1234
    log_interval: int = 100
    eval_interval: int = 2000
    checkpoint_interval: int = 2000
    keep_checkpoints: int = 3
    eval_texts: list[str] = field(
        default_factory=lambda: [
            "வணக்கம், நான் சஹிதி பேசுறேன்.",
            "Hello, this is Sahithi. How can I help you today?",
            "நாளைக்கு meeting பத்து மணிக்கு இருக்கு, please join on time.",
            "konjam wait pannunga, naan check panni solren.",
        ]
    )


@dataclass
class Config:
    data: DataConfig
    audio: AudioConfig
    model: ModelConfig
    train: TrainConfig
    symbols_version: int = SYMBOLS_VERSION

    def to_dict(self) -> dict:
        return {
            "data": asdict(self.data),
            "audio": asdict(self.audio),
            "model": self.model.to_dict(),
            "train": asdict(self.train),
            "symbols_version": self.symbols_version,
        }


def _pick(cls, values: dict | None):
    values = values or {}
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return values


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for section, values in (overrides or {}).items():
        raw.setdefault(section, {}).update(values)
    audio = AudioConfig(**_pick(AudioConfig, raw.get("audio")))
    train = TrainConfig(**_pick(TrainConfig, raw.get("train")))
    model_values = _pick(ModelConfig, raw.get("model"))
    hop = audio.hop_length
    model = ModelConfig(
        n_vocab=len(SYMBOLS),
        n_langs=len(LANGS),
        spec_channels=audio.spec_channels,
        segment_frames=train.segment_size // hop,
        **{k: v for k, v in model_values.items() if k not in ("n_vocab", "n_langs")},
    )
    if model.segment_frames * hop != train.segment_size:
        raise ValueError("train.segment_size must be a multiple of audio.hop_length")
    upsample = 1
    for r in model.upsample_rates:
        upsample *= r
    if upsample != hop:
        raise ValueError(f"product of upsample_rates ({upsample}) must equal hop_length ({hop})")
    return Config(DataConfig(**_pick(DataConfig, raw.get("data"))), audio, model, train)
