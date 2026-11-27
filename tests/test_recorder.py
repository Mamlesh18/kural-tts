import csv

import numpy as np
import soundfile as sf

from recording.recorder import Session, level_check, read_prompts, run


class FakeAudio:
    def __init__(self, amp=0.3):
        self.amp = amp
        self.played = 0

    def record_until_enter(self):
        t = np.arange(48_000) / 48_000
        return (self.amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    def record_seconds(self, seconds):
        return np.zeros(int(seconds * 48_000), dtype=np.float32)

    def play(self, wav):
        self.played += 1


def prompts_file(tmp_path):
    path = tmp_path / "prompts.tsv"
    path.write_text(
        "id\tcategory\ttext\tsay\test_seconds\n"
        "p1\tta\t10 மணிக்கு\tபத்து மணிக்கு\t1.0\n"
        "p2\ten\tHello there\tHello there\t1.0\n"
        "p3\tcs\tok சரி\tok சரி\t1.0\n",
        encoding="utf-8",
    )
    return path


def scripted(*answers):
    it = iter(answers)
    return lambda prompt="": next(it)


def test_level_check():
    assert level_check(np.zeros(0))[1] == "empty"
    assert level_check(np.full(10, 0.999))[1] == "clipped"
    assert level_check(np.full(10, 0.01))[1] == "quiet"
    assert level_check(np.full(10, 0.3))[1] == "ok"


def test_session_flow_retake_skip_and_resume(tmp_path):
    out = tmp_path / "session_01"
    session = Session(read_prompts(prompts_file(tmp_path)), out)
    audio = FakeAudio()
    # room tone, p1: record -> play -> retake -> record -> keep, p2: skip, p3: quit
    answers = scripted("", "", "p", "r", "", "", "s", "q")
    assert run(session, audio, ask=answers) == 1
    assert audio.played == 1
    assert (out / "room_tone.wav").exists() and (out / "p1.wav").exists()
    data, sr = sf.read(out / "p1.wav")
    assert sr == 48_000 and sf.info(out / "p1.wav").subtype == "PCM_24"

    resumed = Session(read_prompts(prompts_file(tmp_path)), out)
    assert [p["id"] for p in resumed.pending()] == ["p2", "p3"]
    assert resumed.take_path("p1").name == "p1_take2.wav"
    with (out / "session_log.csv").open(encoding="utf-8") as f:
        log = list(csv.DictReader(f))
    assert [r["id"] for r in log] == ["p1"] and log[0]["verdict"] == "ok"


def test_say_falls_back_to_text(tmp_path):
    path = tmp_path / "p.tsv"
    path.write_text("id\tcategory\ttext\nx\ten\tHi\n", encoding="utf-8")
    assert read_prompts(path)[0]["say"] == "Hi"
