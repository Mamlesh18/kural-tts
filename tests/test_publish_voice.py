import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "release"))

from publish_voice import prepare


def test_prepare_writes_card_and_checksums(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"fake-onnx")
    (tmp_path / "config.json").write_text(
        json.dumps({"sample_rate": 22050, "training_step": 123}), encoding="utf-8"
    )
    hashes = prepare(tmp_path)
    assert hashes["model.onnx"] == hashlib.sha256(b"fake-onnx").hexdigest()
    card = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "22050 Hz" in card and "| Training step | 123 |" in card and "{{" not in card
