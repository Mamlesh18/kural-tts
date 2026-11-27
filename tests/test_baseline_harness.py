import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "baseline"))

from common import (
    Adapter,
    cer,
    limit_per_category,
    load_sentences,
    model_input,
    normalize_for_cer,
    run_adapter,
    to_float32_mono,
    wer,
)


@pytest.fixture(scope="module")
def rows():
    return load_sentences()


def test_tanglish_rows_feed_tamil_script(rows):
    tg = next(r for r in rows if r["category"] == "tanglish")
    assert model_input(tg) == tg["reference_ta_script"]
    en = next(r for r in rows if r["category"] == "en")
    assert model_input(en) == en["text"]


def test_limit_per_category(rows):
    limited = limit_per_category(rows, 2)
    assert len(limited) == 8
    assert {r["category"] for r in limited} == {"en", "ta", "cs", "tanglish"}


def test_normalize_keeps_tamil_vowel_signs():
    # Vowel signs (Mc/Mn) and virama must survive; punctuation must not.
    assert normalize_for_cer("வணக்கம்!  சரியா?") == "வணக்கம் சரியா"
    assert normalize_for_cer("Hello, WORLD.") == "hello world"


def test_cer_wer():
    assert cer("வணக்கம்", "வணக்கம்") == 0.0
    assert cer("abcd", "abed") == pytest.approx(0.25)
    assert wer("the cat sat", "the cat sat down") == pytest.approx(1 / 3)


def test_to_float32_mono_int16_and_stereo():
    a = to_float32_mono(np.array([0, 16384, -32768], dtype=np.int16))
    assert a.dtype == np.float32 and a[1] == pytest.approx(0.5)
    stereo = np.ones((2, 100), dtype=np.float32)
    assert to_float32_mono(stereo).shape == (100,)


def test_run_adapter_with_fake_model(tmp_path, rows):
    def synth(text):
        if "fail" in text:
            raise ValueError("boom")
        return np.sin(np.linspace(0, 100, 16000)).astype(np.float32), 16000

    fake = Adapter(
        name="fake",
        hf_id="none",
        license="test",
        supports=("en", "ta"),
        load=lambda device: None,
        synth=synth,
    )
    subset = limit_per_category(rows, 2)
    out = run_adapter(fake, subset, tmp_path, "cpu")
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["n_ok"] == 4 and meta["n_skipped"] == 4 and meta["n_error"] == 0
    results = [json.loads(x) for x in (out / "results.jsonl").read_text("utf-8").splitlines()]
    ok = [r for r in results if r["status"] == "ok"]
    assert all((out / r["wav"]).exists() and r["audio_sec"] == pytest.approx(1.0) for r in ok)


def test_score_reference_uses_tamil_script_for_tanglish():
    from score import reference_text

    sentences = {r["id"]: r for r in load_sentences()}
    tg = next(r for r in sentences.values() if r["category"] == "tanglish")
    row = {"id": tg["id"], "input_text": tg["text"]}
    assert reference_text(row, sentences) == tg["reference_ta_script"]
    en = next(r for r in sentences.values() if r["category"] == "en")
    assert reference_text({"id": en["id"], "input_text": "x"}, sentences) == "x"
