# Using Kural-TTS in voice agents

There are three ways to use Kural-TTS, from simplest to most decoupled.

## 1. In-process Python

```python
from kural_tts import KuralTTS

tts = KuralTTS(model_path="exports/sahithi_v1")  # or the released voice: KuralTTS()
for chunk in tts.stream(llm_reply):  # one float32 array per sentence
    player.write(chunk)
```

`stream()` returns the first sentence as soon as it is synthesized, so time to first audio is the
time to synthesize **one sentence**, not the whole reply. Streaming is sentence-level: VITS
generates each sentence in one pass.

## 2. LiveKit Agents plugin

```bash
pip install "kural-tts[agents]"
```

```python
from livekit.agents import AgentSession
from kural_tts.integrations.livekit import KuralLiveKitTTS

session = AgentSession(
    stt=...,  # a Tamil-capable STT
    llm=...,
    tts=KuralLiveKitTTS(model_path="exports/sahithi_v1"),
)
```

The plugin implements LiveKit's `tts.TTS` (non-streaming `synthesize`). `AgentSession` wraps it
in LiveKit's sentence `StreamAdapter`, so LLM tokens are spoken sentence by sentence. Barge-in
cancels the stream, and the remaining sentences are never synthesized. Tested against
`livekit-agents` 1.9.

## 3. Local server (any language or framework)

```bash
pip install "kural-tts[server]"
kural-tts serve --model exports/sahithi_v1 --port 8765          # binds 127.0.0.1 by default
```

| Endpoint | Use |
|---|---|
| `POST /tts` | `{"text": "...", "lang": "auto", "speed": 1.0, "format": "wav" \| "pcm16"}` returns audio. Sample rate is in the `X-Sample-Rate` header |
| `WS /tts/stream` | Send `{"text": "..."}`. You receive `{"event":"start","sample_rate":22050}`, then binary PCM16 chunks (one per sentence), then `{"event":"end","time_to_first_audio":...}`. Send `{"event":"cancel"}` to stop (barge-in). The connection stays open for the next request |
| `GET /health` | 200 once the model is loaded and warmed up |
| `GET /model-info` | voice, sample rate, ONNX providers, version |

Concurrency is limited by `--max-concurrency` (default 2 simultaneous syntheses). Logs record text
**length only**, never content.

## Latency checklist

- Measure end of user speech → ASR final → LLM first token → **TTS first audio** → playback.
  `time_to_first_audio` in the WebSocket `end` event gives the TTS part.
- Ask the LLM for **short first sentences**, so the first audio chunk is ready sooner.
- Use the GPU build (`pip install "kural-tts[gpu]"`) when serving many concurrent calls. A single
  call runs comfortably on CPU.
- Let the LLM write Tamil, English or Tanglish freely. The frontend handles all three. Test with
  `kural-tts "..." --normalize` when a reading sounds wrong.
