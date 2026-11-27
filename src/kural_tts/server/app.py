"""Optional local HTTP/WebSocket server (``pip install "kural-tts[server]"``).

    kural-tts serve --model exports/sahithi_v1 --port 8765

Endpoints
  POST /tts           JSON {text, lang?, speed?, noise_scale?, format?: "wav"|"pcm16"} -> audio
  WS   /tts/stream    send JSON {text, ...}; receive binary PCM16 chunks (one per sentence),
                      then a JSON {"event": "end", ...}. Send {"event": "cancel"} to stop early.
  GET  /health        readiness (model loaded and warmed up)
  GET  /model-info    voice, sample rate, providers, package version

Binds to 127.0.0.1 by default: it is meant to sit next to your voice agent, not on the internet.
Request text is never logged, only its length.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel, Field

from kural_tts import __version__
from kural_tts.api import MAX_CHARS, KuralTTS

log = logging.getLogger("kural_tts.server")


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS)
    lang: Literal["auto", "ta", "en"] = "auto"
    speed: float = Field(1.0, gt=0.25, le=4.0)
    noise_scale: float | None = Field(None, ge=0.0, le=2.0)
    format: Literal["wav", "pcm16"] = "wav"


def to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def to_wav(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def create_app(tts: KuralTTS | None = None, max_concurrency: int = 2) -> FastAPI:
    engine = tts or KuralTTS()
    gate = asyncio.Semaphore(max_concurrency)
    state = {"ready": False}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await asyncio.to_thread(engine.synthesize, "வணக்கம். Hello.")  # load + warm up
        state["ready"] = True
        log.info("kural-tts ready (sample_rate=%s)", engine.sample_rate)
        yield

    app = FastAPI(title="Kural-TTS", version=__version__, lifespan=lifespan)

    @app.get("/health")
    async def health():
        if not state["ready"]:
            raise HTTPException(503, "warming up")
        return {"status": "ok"}

    @app.get("/model-info")
    async def model_info():
        return {
            "voice": engine.voice,
            "sample_rate": engine.sample_rate,
            "providers": engine.providers,
            "version": __version__,
            "languages": ["ta", "en", "ta-en code-switching", "tanglish"],
            "streaming": "sentence-level",
        }

    @app.post("/tts")
    async def synthesize(req: TTSRequest):
        rid = uuid.uuid4().hex[:8]
        start = time.perf_counter()
        async with gate:
            audio, sr = await asyncio.to_thread(
                engine.synthesize, req.text, req.lang, req.speed, req.noise_scale
            )
        log.info("tts %s chars=%d audio=%.2fs took=%.3fs", rid, len(req.text), len(audio) / sr,
                 time.perf_counter() - start)  # fmt: skip
        headers = {"X-Request-Id": rid, "X-Sample-Rate": str(sr)}
        if req.format == "pcm16":
            return Response(to_pcm16(audio), media_type="audio/pcm", headers=headers)
        return Response(to_wav(audio, sr), media_type="audio/wav", headers=headers)

    @app.websocket("/tts/stream")
    async def stream(ws: WebSocket):
        await ws.accept()
        # A single reader task owns ws.receive_*. Requests and cancels arrive through `inbox`,
        # so checking for barge-in never cancels a pending receive.
        inbox: asyncio.Queue[str | None] = asyncio.Queue()

        async def reader() -> None:
            try:
                while True:
                    await inbox.put(await ws.receive_text())
            except WebSocketDisconnect:
                await inbox.put(None)

        reader_task = asyncio.create_task(reader())
        backlog: list[str] = []
        try:
            while True:
                msg = backlog.pop(0) if backlog else await inbox.get()
                if msg is None:
                    return
                try:
                    data = json.loads(msg)
                    if data.get("event") == "cancel":
                        continue  # nothing in progress
                    req = TTSRequest.model_validate(data)
                except ValueError as e:
                    await ws.send_text(json.dumps({"event": "error", "detail": str(e)[:300]}))
                    continue
                if not await _stream_one(ws, req, inbox, backlog):
                    return
        except WebSocketDisconnect:
            return
        finally:
            reader_task.cancel()

    async def _stream_one(ws, req: TTSRequest, inbox: asyncio.Queue, backlog: list) -> bool:
        """Stream one request. Returns False if the client disconnected."""
        rid = uuid.uuid4().hex[:8]
        start = time.perf_counter()
        first_audio = None
        chunks = 0
        cancelled = False
        sentences = iter(engine.stream(req.text, req.lang, req.speed, req.noise_scale))
        await ws.send_text(json.dumps({"event": "start", "request_id": rid,
                                       "sample_rate": engine.sample_rate}))  # fmt: skip
        while not cancelled:
            async with gate:
                audio = await asyncio.to_thread(next, sentences, None)
            if audio is None:
                break
            if first_audio is None:
                first_audio = time.perf_counter() - start
            await ws.send_bytes(to_pcm16(audio))
            chunks += 1
            while not inbox.empty():
                msg = inbox.get_nowait()
                if msg is None:
                    return False
                try:
                    is_cancel = json.loads(msg).get("event") == "cancel"
                except ValueError:
                    is_cancel = False
                if is_cancel:
                    cancelled = True
                else:
                    backlog.append(msg)  # next request, handled after this one
        await ws.send_text(json.dumps({"event": "end", "request_id": rid, "chunks": chunks,
                                       "cancelled": cancelled,
                                       "time_to_first_audio": first_audio}))  # fmt: skip
        return True

    return app


def serve(model_path: str | None = None, host: str = "127.0.0.1", port: int = 8765,
          device: str = "auto", max_concurrency: int = 2) -> None:  # fmt: skip
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    app = create_app(KuralTTS(model_path=model_path, device=device), max_concurrency)
    uvicorn.run(app, host=host, port=port)
