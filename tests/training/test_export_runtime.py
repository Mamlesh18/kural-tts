"""Checkpoint -> ONNX bundle -> pip-package runtime, end to end (tiny random model)."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("librosa")

import soundfile as sf  # noqa: E402

from kural_tts import KuralTTS  # noqa: E402
from kural_tts.cli import main as cli_main  # noqa: E402
from kural_tts.runtime import IncompatibleModelError, OnnxVoice  # noqa: E402
from training.config import load_config  # noqa: E402
from training.export_onnx import export  # noqa: E402
from training.models.vits.synthesizer import SynthesizerTrn  # noqa: E402

TINY = Path(__file__).resolve().parents[2] / "training" / "configs" / "tiny.yaml"


@pytest.fixture(scope="module")
def bundle(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("export")
    cfg = load_config(TINY)
    torch.manual_seed(0)
    model = SynthesizerTrn(cfg.model)
    with torch.no_grad():
        model.dp.proj.bias.fill_(0.7)  # ~2 frames per phoneme
    ckpt = tmp / "ckpt_1.pt"
    torch.save({"net_g": model.state_dict(), "config": cfg.to_dict(), "step": 1}, ckpt)
    export(ckpt, tmp / "voice")
    return tmp / "voice"


def test_bundle_contents(bundle):
    cfg = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
    assert cfg["format"] == "kural-tts-onnx/1"
    assert cfg["sample_rate"] == 22050 and cfg["voice"] == "sahithi"
    assert (bundle / "model.onnx").stat().st_size > 10_000


def test_synthesize_with_local_model(bundle):
    tts = KuralTTS(model_path=bundle, device="cpu")
    audio, sr = tts.synthesize("வணக்கம்! Hello, naan Sahithi pesuren.")
    assert sr == 22050 and audio.dtype == np.float32 and audio.ndim == 1
    assert len(audio) > sr // 10 and np.isfinite(audio).all() and np.abs(audio).max() <= 1.0
    assert tts.providers == ["CPUExecutionProvider"]


def test_speed_changes_length(bundle):
    tts = KuralTTS(model_path=bundle, device="cpu")
    slow, _ = tts.synthesize("நாளைக்கு meeting இருக்கு.", speed=0.5, noise_scale=0.0)
    fast, _ = tts.synthesize("நாளைக்கு meeting இருக்கு.", speed=2.0, noise_scale=0.0)
    assert len(slow) > len(fast)


def test_stream_yields_one_chunk_per_sentence(bundle):
    tts = KuralTTS(model_path=bundle, device="cpu")
    chunks = list(tts.stream("வணக்கம். How are you? சரி!"))
    assert len(chunks) == 3


def test_save_and_cli(bundle, tmp_path):
    out = KuralTTS(model_path=bundle).save("சரி", tmp_path / "a.wav")
    data, sr = sf.read(out)
    assert sr == 22050 and len(data) > 0
    assert cli_main(["Hello", "-o", str(tmp_path / "b.wav"), "--model", str(bundle)]) == 0
    assert (tmp_path / "b.wav").exists()


def test_env_var_selects_model(bundle, monkeypatch):
    monkeypatch.setenv("KURAL_TTS_MODEL", str(bundle))
    audio, _ = KuralTTS().synthesize("Hi")
    assert len(audio) > 0


def test_incompatible_symbols_version_is_rejected(bundle, tmp_path):
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "model.onnx").write_bytes((bundle / "model.onnx").read_bytes())
    cfg = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
    (bad / "config.json").write_text(json.dumps({**cfg, "symbols_version": 999}), "utf-8")
    with pytest.raises(IncompatibleModelError):
        OnnxVoice(bad)


def test_render_eval_set_matches_baseline_format(bundle, tmp_path):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "eval"))
    from render_eval_set import main as render

    out = render(["--model", str(bundle), "--name", "kural_tiny", "--out-root", str(tmp_path)])
    assert out.name == "kural_tiny__cpu"
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["n_ok"] == 80 and meta["n_error"] == 0
    rows = [json.loads(x) for x in (out / "results.jsonl").read_text("utf-8").splitlines()]
    tanglish = next(r for r in rows if r["category"] == "tanglish")
    assert tanglish["input_text"].isascii()  # Kural was given the raw Tanglish
