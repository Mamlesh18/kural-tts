import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyloudnorm")
pytest.importorskip("librosa")

import soundfile as sf

from training.data_pipeline.build import QCRules, build, split_for

RAW_SR = 48000


def speechy(seconds, seed, amp=0.3):
    """Tone bursts separated by short gaps, padded with quiet room tone."""
    rng = np.random.default_rng(seed)
    n = int(seconds * RAW_SR)
    t = np.arange(n) / RAW_SR
    env = (np.sin(2 * np.pi * 3 * t) > -0.3).astype(np.float32)
    wav = amp * np.sin(2 * np.pi * (180 + seed) * t) * env
    pad = 1e-4 * rng.standard_normal(int(0.5 * RAW_SR))
    return np.concatenate([pad, wav, pad]).astype(np.float32)


@pytest.fixture
def raw(tmp_path) -> tuple[Path, Path]:
    raw = tmp_path / "raw" / "session_01"
    raw.mkdir(parents=True)
    prompts = [
        ("p001", "ta", "இன்று மாலை சந்திக்கலாம்."),
        ("p002", "en", "Please confirm your booking for tomorrow."),
        ("p003", "cs", "நாளைக்கு meeting இருக்கு."),
        ("p004", "ta", "சரி, புரிஞ்சுது."),  # also in the eval set -> must be excluded
        ("p005", "en", "This one is clipped."),
        ("p006", "en", "Please confirm your booking for tomorrow."),  # duplicate text
    ]
    for i, (pid, _, _) in enumerate(prompts):
        wav = speechy(2.0, i)
        if pid == "p005":
            wav = np.clip(wav * 20, -1, 1)
        sf.write(raw / f"{pid}.wav", wav, RAW_SR, subtype="PCM_24")
    noisy = speechy(2.0, 0) + 0.05 * np.random.default_rng(9).standard_normal(
        len(speechy(2.0, 0))
    ).astype(np.float32)
    sf.write(raw / "p001_take2.wav", noisy, RAW_SR, subtype="PCM_24")  # worse take
    tsv = tmp_path / "prompts.tsv"
    tsv.write_text(
        "id\tcategory\ttext\n" + "\n".join("\t".join(p) for p in prompts) + "\n", "utf-8"
    )
    return tmp_path / "raw", tsv


def test_build_pipeline(raw, tmp_path):
    raw_dir, prompts = raw
    out = tmp_path / "out"
    rows = {r["id"]: r for r in build(raw_dir, prompts, out)}

    assert rows["p001"]["n_takes"] == 2 and rows["p001"]["raw_file"].endswith("p001.wav")
    assert rows["p005"]["status"] == "reject" and "clipping" in rows["p005"]["reasons"]
    assert "eval_overlap" in rows["p004"]["reasons"]
    dup = [rows["p002"], rows["p006"]]
    assert sorted(r["status"] == "reject" for r in dup) == [False, True]
    assert rows["p003"]["status"] in ("pass", "review")

    wav, sr = sf.read(out / rows["p001"]["audio"])
    assert sr == 22050 and wav.ndim == 1 and np.abs(wav).max() <= 1.0
    assert rows["p001"]["source"] == "session_01" and len(rows["p001"]["raw_sha256"]) == 64

    all_rows = [
        json.loads(x) for x in (out / "manifests" / "all.jsonl").read_text("utf-8").split("\n") if x
    ]
    assert len(all_rows) == 6
    split_rows = []
    for split in ("train", "val", "test"):
        lines = (out / "manifests" / f"{split}.jsonl").read_text("utf-8").splitlines()
        split_rows += [json.loads(x) for x in lines]
    assert all(r["status"] == "pass" for r in split_rows)
    assert (out / "qc_report.md").read_text("utf-8").startswith("# Data QC report")
    assert (out / "manifests" / "test_ids.lock").exists()


def test_rebuild_is_deterministic(raw, tmp_path):
    raw_dir, prompts = raw
    a = build(raw_dir, prompts, tmp_path / "a")
    b = build(raw_dir, prompts, tmp_path / "b")
    assert a == b


def test_split_is_stable_and_respects_lock():
    rules = QCRules(test_frac=0.1, val_frac=0.1)
    ids = [f"u{i:04d}" for i in range(2000)]
    splits = [split_for(i, set(), rules) for i in ids]
    assert splits == [split_for(i, set(), rules) for i in ids]
    test_share = splits.count("test") / len(ids)
    assert 0.07 < test_share < 0.13
    assert split_for("u0000", {"u0000"}, QCRules(test_frac=0.0)) == "test"
