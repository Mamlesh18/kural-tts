import json

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import soundfile as sf
from fastapi.testclient import TestClient

from kural_tts.server.app import create_app

SR = 22050


class FakeEngine:
    """Stands in for KuralTTS: 0.1 s of audio per sentence."""

    voice = "sahithi"
    sample_rate = SR
    providers = ("CPUExecutionProvider",)

    def __init__(self):
        self.calls = 0

    def _audio(self):
        return np.full(SR // 10, 0.1, dtype=np.float32)

    def synthesize(self, text, lang="auto", speed=1.0, noise_scale=None):
        self.calls += 1
        n = max(1, text.count(".") + text.count("?"))
        return np.concatenate([self._audio()] * n), SR

    def stream(self, text, lang="auto", speed=1.0, noise_scale=None):
        for _ in range(max(1, text.count("."))):
            yield self._audio()


@pytest.fixture
def client():
    with TestClient(create_app(FakeEngine())) as c:  # runs lifespan (warm-up)
        yield c


def test_health_and_info(client):
    assert client.get("/health").json() == {"status": "ok"}
    info = client.get("/model-info").json()
    assert (
        info["sample_rate"] == SR
        and info["providers"] == ["CPUExecutionProvider"]
        and info["streaming"] == "sentence-level"
    )


def test_post_tts_wav_and_pcm(client):
    r = client.post("/tts", json={"text": "வணக்கம். Hello."})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    import io

    audio, sr = sf.read(io.BytesIO(r.content))
    assert sr == SR and len(audio) == 2 * SR // 10
    r = client.post("/tts", json={"text": "Hi.", "format": "pcm16"})
    assert r.headers["x-sample-rate"] == str(SR) and len(r.content) == 2 * SR // 10


def test_validation(client):
    assert client.post("/tts", json={"text": ""}).status_code == 422
    assert client.post("/tts", json={"text": "hi", "speed": 0}).status_code == 422
    assert client.post("/tts", json={"text": "hi", "lang": "fr"}).status_code == 422


def test_websocket_streams_sentences(client):
    with client.websocket_connect("/tts/stream") as ws:
        ws.send_text(json.dumps({"text": "One. Two. Three."}))
        start = json.loads(ws.receive_text())
        assert start["event"] == "start" and start["sample_rate"] == SR
        chunks = [ws.receive_bytes() for _ in range(3)]
        assert all(len(c) == 2 * SR // 10 for c in chunks)
        end = json.loads(ws.receive_text())
        assert end["event"] == "end" and end["chunks"] == 3 and not end["cancelled"]
        ws.send_text("not json")
        assert json.loads(ws.receive_text())["event"] == "error"


class SlowEngine(FakeEngine):
    def stream(self, text, lang="auto", speed=1.0, noise_scale=None):
        import time

        for _ in range(10):
            time.sleep(0.05)
            yield self._audio()


def test_websocket_cancel_stops_early_and_keeps_connection():
    with TestClient(create_app(SlowEngine())) as c, c.websocket_connect("/tts/stream") as ws:
        ws.send_text(json.dumps({"text": "long answer"}))
        assert json.loads(ws.receive_text())["event"] == "start"
        ws.send_text(json.dumps({"event": "cancel"}))  # user barged in
        msg = ws.receive()
        chunks = 0
        while "bytes" in msg and msg["bytes"] is not None:
            chunks += 1
            msg = ws.receive()
        end = json.loads(msg["text"])
        assert end["cancelled"] and chunks < 10
        ws.send_text(json.dumps({"text": "Next."}))  # connection still usable
        assert json.loads(ws.receive_text())["event"] == "start"
