import pytest

import kural_tts
from kural_tts.api import KuralTTS, ModelNotAvailableError
from kural_tts.cli import main
from kural_tts.text.segment import split_sentences


def test_version_exposed():
    assert isinstance(kural_tts.__version__, str)


def test_unknown_voice_rejected():
    with pytest.raises(ValueError):
        KuralTTS(voice="nobody")


@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_text_rejected(bad):
    with pytest.raises(ValueError):
        KuralTTS().synthesize(bad)


def test_synthesize_reports_missing_model():
    with pytest.raises(ModelNotAvailableError):
        KuralTTS().synthesize("வணக்கம்")


def test_cli_without_model_exits_nonzero(tmp_path, capsys):
    assert main(["hello", "-o", str(tmp_path / "x.wav")]) == 1
    assert "No Kural-TTS model" in capsys.readouterr().err


def test_split_sentences_mixed():
    text = "வணக்கம்! How are you? நான் Dr. Kumar பேசுறேன். Price is 10.5 rupees."
    assert split_sentences(text) == [
        "வணக்கம்!",
        "How are you?",
        "நான் Dr. Kumar பேசுறேன்.",
        "Price is 10.5 rupees.",
    ]


def test_normalize_works_without_model():
    spans = KuralTTS().normalize("நாளைக்கு meeting 10 மணிக்கு")
    assert [(s.text, s.lang) for s in spans] == [
        ("நாளைக்கு", "ta"),
        ("meeting", "en"),
        ("பத்து மணிக்கு", "ta"),
    ]


def test_cli_normalize_prints_spans(capsys):
    assert main(["konjam wait pannunga", "--normalize"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "[ta] கொஞ்சம்",
        "[en] wait",
        "[ta] பண்ணுங்க",
    ]
