"""LiveKit Agents TTS plugin (``pip install "kural-tts[agents]"``), running Kural in-process.

    from livekit.agents import AgentSession
    from kural_tts.integrations.livekit import KuralLiveKitTTS

    session = AgentSession(stt=..., llm=..., tts=KuralLiveKitTTS(model_path="exports/sahithi_v1"))

``synthesize()`` pushes audio sentence by sentence, so playback starts after the first sentence.
For incremental LLM text, AgentSession wraps this non-streaming TTS in its sentence
``StreamAdapter`` automatically. Interruptions (barge-in) cancel the stream, and the remaining
sentences are not synthesized.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import numpy as np
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions, tts

from kural_tts.api import KuralTTS


class KuralLiveKitTTS(tts.TTS):
    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lang: str = "auto",
        speed: float = 1.0,
        noise_scale: float | None = None,
        engine: KuralTTS | None = None,
    ) -> None:
        self._engine = engine or KuralTTS(model_path=model_path, device=device)
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=self._engine.sample_rate,
            num_channels=1,
        )
        self._lang = lang
        self._speed = speed
        self._noise_scale = noise_scale

    @property
    def model(self) -> str:
        return f"kural-tts/{self._engine.voice}"

    @property
    def provider(self) -> str:
        return "kural-tts"

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> KuralChunkedStream:
        return KuralChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class KuralChunkedStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        owner: KuralLiveKitTTS = self._tts  # type: ignore[assignment]
        engine = owner._engine
        output_emitter.initialize(
            request_id=uuid.uuid4().hex,
            sample_rate=engine.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )
        sentences = iter(
            engine.stream(self.input_text, owner._lang, owner._speed, owner._noise_scale)
        )
        while True:
            # Synthesis runs off the event loop. Cancellation (barge-in) stops before the
            # next sentence.
            audio = await asyncio.to_thread(next, sentences, None)
            if audio is None:
                break
            output_emitter.push((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
        output_emitter.flush()
