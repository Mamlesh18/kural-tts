"""End-to-end CPU smoke test: dataset -> train -> checkpoint -> resume (tiny model)."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("librosa")
pytest.importorskip("yaml")

import soundfile as sf  # noqa: E402

from training.checkpoint import list_checkpoints  # noqa: E402
from training.config import load_config  # noqa: E402
from training.dataset import BucketBatchSampler, TextAudioDataset, collate  # noqa: E402
from training.train import main  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
TINY = REPO / "training" / "configs" / "tiny.yaml"
TEXTS = [
    "வணக்கம், நான் சஹிதி.",
    "Hello, how are you?",
    "நாளைக்கு meeting பத்து மணிக்கு.",
    "konjam wait pannunga.",
    "Your OTP is 4521.",
    "சரி, புரிஞ்சுது.",
]


def make_dataset(root: Path, sr: int = 22050) -> Path:
    rng = np.random.default_rng(0)
    (root / "wavs").mkdir(parents=True)
    rows = []
    for i, text in enumerate(TEXTS):
        n = int(sr * (0.8 + 0.1 * i))
        t = np.arange(n) / sr
        wav = 0.3 * np.sin(2 * np.pi * (150 + 20 * i) * t) * np.hanning(n)
        wav += 0.01 * rng.standard_normal(n)
        path = f"wavs/utt_{i}.wav"
        sf.write(root / path, wav.astype(np.float32), sr, subtype="PCM_16")
        rows.append({"id": f"utt_{i}", "audio": path, "text": text, "duration": n / sr, "sr": sr,
                     "quality": "pass"})  # fmt: skip
    rows.append({**rows[0], "id": "bad", "quality": "reject"})
    manifest = root / "train.jsonl"
    manifest.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), "utf-8")
    return manifest


def test_dataset_and_sampler(tmp_path):
    manifest = make_dataset(tmp_path)
    cfg = load_config(TINY, {"data": {"audio_root": str(tmp_path)}})
    ds = TextAudioDataset(manifest, cfg.data, cfg.audio)
    assert len(ds) == len(TEXTS) and ds.skipped == {"quality": 1}
    ids, langs, spec, wav = ds[2]
    assert len(ids) == len(langs) and set(langs.tolist()) == {0, 1}  # code-switched item
    assert spec.size(1) * cfg.audio.hop_length == len(wav)
    sampler = BucketBatchSampler(ds.frame_lengths(), batch_size=2, seed=0)
    batches = list(sampler)
    assert sorted(i for b in batches for i in b) == list(range(len(ds)))
    x, x_len, lang, spec, spec_len, wav, w_len = collate([ds[i] for i in batches[0]])
    assert x.shape == lang.shape and wav.shape[1] == 1


def test_train_resume_smoke(tmp_path):
    manifest = make_dataset(tmp_path / "data")
    run = tmp_path / "run"
    args = ["--config", str(TINY), "--run-dir", str(run), "--device", "cpu",
            "--train-manifest", str(manifest), "--val-manifest", "",
            "--audio-root", str(tmp_path / "data"), "--no-tensorboard"]  # fmt: skip

    trainer = main([*args, "--max-steps", "3"])
    assert trainer.step == 3
    assert [p.name for p in list_checkpoints(run)] == ["ckpt_2.pt", "ckpt_3.pt"]
    assert (run / "config.json").exists()

    trainer = main([*args, "--max-steps", "5"])  # simulated Colab reconnect
    assert trainer.step == 5
    assert [p.name for p in list_checkpoints(run)] == ["ckpt_4.pt", "ckpt_5.pt"]
    for p in trainer.net_g.parameters():
        assert torch.isfinite(p).all()
