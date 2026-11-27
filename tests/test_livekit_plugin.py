import asyncio

import numpy as np
import pytest

pytest.importorskip("livekit.agents")

from kural_tts.integrations.livekit import KuralLiveKitTTS

SR = 22050


class FakeEngine:
    voice = "sahithi"
    sample_rate = SR

    def stream(self, text, lang="auto", speed=1.0, noise_scale=None):
        for _ in text.split("."):
            yield np.full(SR // 5, 0.2, dtype=np.float32)  # 200 ms per sentence


def test_chunked_stream_emits_pcm_frames():
    async def run():
        plugin = KuralLiveKitTTS(engine=FakeEngine())
        assert plugin.sample_rate == SR and plugin.num_channels == 1
        assert not plugin.capabilities.streaming and plugin.provider == "kural-tts"
        frames = []
        async with plugin.synthesize("One. Two. Three") as stream:
            async for ev in stream:
                frames.append(ev.frame)
        await plugin.aclose()
        return frames

    frames = asyncio.run(run())
    samples = np.concatenate([np.frombuffer(f.data, dtype=np.int16) for f in frames])
    # LiveKit's emitter ramps frame sizes and appends a short silent frame at the end, so
    # check that all pushed audio arrived and anything extra is silence.
    assert np.count_nonzero(samples) == 3 * SR // 5
    assert not samples[3 * SR // 5 :].any()
    assert all(f.sample_rate == SR and f.num_channels == 1 for f in frames)
    first = np.frombuffer(frames[0].data, dtype=np.int16)
    assert abs(int(first[0]) - int(0.2 * 32767)) <= 1
