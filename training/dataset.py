"""Manifest-driven dataset, collate function and length-bucketed batch sampler."""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, Sampler

from kural_tts.text.g2p import text_to_ids
from training.audio import AudioConfig, spectrogram
from training.config import DataConfig


def read_manifest(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class TextAudioDataset(Dataset):
    """Items: (phoneme ids, language ids, linear spectrogram, waveform).

    Text is phonemized once at construction with the same frontend the package ships, so
    training and inference see identical inputs.
    """

    def __init__(self, manifest: str | Path, data_cfg: DataConfig, audio_cfg: AudioConfig):
        self.audio_cfg = audio_cfg
        self.root = Path(data_cfg.audio_root)
        self.items: list[dict] = []
        self.skipped: dict[str, int] = {}
        for row in read_manifest(manifest):
            reason = self._reject_reason(row, data_cfg)
            if reason is None:
                ids, langs = text_to_ids(row[data_cfg.text_field])
                if not data_cfg.min_phonemes <= len(ids) <= data_cfg.max_phonemes:
                    reason = "phoneme_length"
                else:
                    self.items.append({**row, "ids": ids, "langs": langs})
            if reason is not None:
                self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def _reject_reason(self, row: dict, cfg: DataConfig) -> str | None:
        if row.get("quality", "pass") != "pass":
            return "quality"
        if row.get("sr") not in (None, self.audio_cfg.sample_rate):
            return "sample_rate"
        if float(row.get("duration", 0)) > cfg.max_seconds:
            return "too_long"
        return None

    def frame_lengths(self) -> list[int]:
        sr, hop = self.audio_cfg.sample_rate, self.audio_cfg.hop_length
        return [int(float(it.get("duration", 1.0)) * sr) // hop for it in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        wav, sr = sf.read(self.root / item["audio"], dtype="float32", always_2d=False)
        if sr != self.audio_cfg.sample_rate:
            raise ValueError(f"{item['audio']}: {sr} Hz, expected {self.audio_cfg.sample_rate}")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        hop = self.audio_cfg.hop_length
        frames = len(wav) // hop
        wav = torch.from_numpy(np.ascontiguousarray(wav[: frames * hop]))
        spec = spectrogram(wav.unsqueeze(0), self.audio_cfg).squeeze(0)
        return (
            torch.tensor(item["ids"], dtype=torch.long),
            torch.tensor(item["langs"], dtype=torch.long),
            spec,
            wav,
        )


def collate(batch):
    ids, langs, specs, wavs = zip(*batch, strict=True)
    b = len(batch)
    x_len = torch.tensor([len(t) for t in ids])
    y_len = torch.tensor([s.size(1) for s in specs])
    w_len = torch.tensor([len(w) for w in wavs])
    x = torch.zeros(b, int(x_len.max()), dtype=torch.long)
    lang = torch.zeros_like(x)
    spec = torch.zeros(b, specs[0].size(0), int(y_len.max()))
    wav = torch.zeros(b, 1, int(w_len.max()))
    for i in range(b):
        x[i, : x_len[i]] = ids[i]
        lang[i, : x_len[i]] = langs[i]
        spec[i, :, : y_len[i]] = specs[i]
        wav[i, 0, : w_len[i]] = wavs[i]
    return x, x_len, lang, spec, y_len, wav, w_len


class BucketBatchSampler(Sampler[list[int]]):
    """Batches of similar-length utterances (less padding), reshuffled every epoch."""

    def __init__(self, lengths: list[int], batch_size: int, seed: int = 0, bucket_mult: int = 50):
        self.lengths = lengths
        self.batch_size = batch_size
        self.seed = seed
        self.bucket_size = batch_size * bucket_mult
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        order = list(range(len(self.lengths)))
        rng.shuffle(order)
        batches = []
        for start in range(0, len(order), self.bucket_size):
            bucket = sorted(order[start : start + self.bucket_size], key=self.lengths.__getitem__)
            batches += [
                bucket[i : i + self.batch_size] for i in range(0, len(bucket), self.batch_size)
            ]
        rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._batches())

    def __len__(self) -> int:
        return len(self._batches())
