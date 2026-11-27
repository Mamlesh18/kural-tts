"""Terminal recorder for Sahithi's sessions (one WAV per prompt, 48 kHz / 24-bit mono).

    uv run --group record python recording/recorder.py \\
        --prompts recording/prompts.tsv --out recordings/session_01
    uv run --group record python recording/recorder.py --list-devices

Per prompt: Enter = start, Enter = stop, then
    Enter = keep and go to the next prompt   r = retake   p = play back   s = skip   q = quit
Prompts that already have a file in the session folder are skipped, so you can stop and resume
at any time. Each session starts with 3 s of room tone (stay silent), which the data pipeline
uses for noise checks. The speaker reads the large SAY line, which is the exact spoken form the
model is trained on.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 48_000
ROOM_TONE_SEC = 3.0
LOG_FIELDS = ["time", "id", "file", "duration", "peak_dbfs", "verdict"]


def read_prompts(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    for r in rows:
        r.setdefault("say", r["text"])
        r["say"] = r["say"] or r["text"]
    return rows


def level_check(wav: np.ndarray) -> tuple[float, str]:
    """(peak dBFS, verdict) where verdict is ok | clipped | quiet | empty."""
    if wav.size == 0:
        return -120.0, "empty"
    peak = float(np.abs(wav).max())
    peak_db = 20 * np.log10(peak + 1e-12)
    if peak >= 0.99:
        return peak_db, "clipped"
    if peak_db < -24:
        return peak_db, "quiet"
    return peak_db, "ok"


@dataclass
class Session:
    prompts: list[dict]
    out_dir: Path
    done: set[str] = field(default_factory=set)

    def __post_init__(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.done = {p.stem.split("_take")[0] for p in self.out_dir.glob("*.wav")}

    def pending(self, start_at: str | None = None) -> list[dict]:
        rows = [p for p in self.prompts if p["id"] not in self.done]
        if start_at:
            ids = [p["id"] for p in rows]
            rows = rows[ids.index(start_at) :] if start_at in ids else rows
        return rows

    def take_path(self, prompt_id: str) -> Path:
        path = self.out_dir / f"{prompt_id}.wav"
        n = 2
        while path.exists():
            path = self.out_dir / f"{prompt_id}_take{n}.wav"
            n += 1
        return path

    def save(self, prompt_id: str, wav: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[Path, str]:
        path = self.take_path(prompt_id)
        sf.write(path, wav, sr, subtype="PCM_24")
        peak_db, verdict = level_check(wav)
        self._log(prompt_id, path, len(wav) / sr, peak_db, verdict)
        self.done.add(prompt_id)
        return path, verdict

    def _log(self, prompt_id, path, duration, peak_db, verdict) -> None:
        log = self.out_dir / "session_log.csv"
        new = not log.exists()
        with log.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if new:
                w.writeheader()
            w.writerow({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "id": prompt_id,
                        "file": path.name, "duration": f"{duration:.2f}",
                        "peak_dbfs": f"{peak_db:.1f}", "verdict": verdict})  # fmt: skip


class SoundDeviceAudio:
    """Microphone/speaker access through sounddevice (PortAudio)."""

    def __init__(self, device: int | None = None, sr: int = SAMPLE_RATE):
        import sounddevice as sd

        self.sd = sd
        self.device = device
        self.sr = sr

    def record_until_enter(self) -> np.ndarray:
        chunks: list[np.ndarray] = []

        def callback(indata, frames, t, status):
            chunks.append(indata[:, 0].copy())

        with self.sd.InputStream(samplerate=self.sr, channels=1, dtype="float32",
                                 device=self.device, callback=callback):  # fmt: skip
            input("  ● recording... press Enter to stop ")
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def record_seconds(self, seconds: float) -> np.ndarray:
        wav = self.sd.rec(int(seconds * self.sr), samplerate=self.sr, channels=1,
                          dtype="float32", device=self.device)  # fmt: skip
        self.sd.wait()
        return wav[:, 0]

    def play(self, wav: np.ndarray) -> None:
        self.sd.play(wav, self.sr)
        self.sd.wait()


def show(prompt: dict, index: int, total: int) -> None:
    print("\n" + "─" * 72)
    print(f"[{index}/{total}]  {prompt['id']}  ({prompt.get('category', '?')})")
    print(f"\n  SAY:  {prompt['say']}\n")
    if prompt["say"] != prompt["text"]:
        print(f"  (written: {prompt['text']})")


def run(session: Session, audio, start_at: str | None = None, ask=input) -> int:
    """Interactive loop. Returns the number of prompts recorded this run."""
    room = session.out_dir / "room_tone.wav"
    if not room.exists():
        ask("Room tone: stay completely silent for 3 seconds. Press Enter to start.")
        sf.write(room, audio.record_seconds(ROOM_TONE_SEC), SAMPLE_RATE, subtype="PCM_24")
    pending = session.pending(start_at)
    recorded = 0
    for i, prompt in enumerate(pending, 1):
        show(prompt, i, len(pending))
        while True:
            cmd = ask("  Enter = record, s = skip, q = quit > ").strip().lower()
            if cmd == "q":
                return recorded
            if cmd == "s":
                break
            wav = audio.record_until_enter()
            peak_db, verdict = level_check(wav)
            warning = "" if verdict == "ok" else f"  ⚠ {verdict.upper()}, consider a retake"
            print(f"  {len(wav) / SAMPLE_RATE:.1f}s, peak {peak_db:.1f} dBFS{warning}")
            choice = ask("  Enter = keep, r = retake, p = play > ").strip().lower()
            while choice == "p":
                audio.play(wav)
                choice = ask("  Enter = keep, r = retake, p = play > ").strip().lower()
            if choice == "r":
                continue
            path, _ = session.save(prompt["id"], wav)
            print(f"  saved {path.name}")
            recorded += 1
            break
    print(f"\nSession complete: {recorded} recorded this run.")
    return recorded


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--prompts", type=Path, help="prompts.tsv from make_script.py")
    p.add_argument("--out", type=Path, help="session folder, e.g. recordings/session_01")
    p.add_argument("--device", type=int, help="input device index (see --list-devices)")
    p.add_argument("--start-at", help="prompt id to start from")
    p.add_argument("--list-devices", action="store_true")
    args = p.parse_args(argv)
    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return
    if not args.prompts or not args.out:
        p.error("--prompts and --out are required")
    session = Session(read_prompts(args.prompts), args.out)
    print(f"{len(session.done)} already recorded, {len(session.pending())} to go.")
    run(session, SoundDeviceAudio(args.device), args.start_at)


if __name__ == "__main__":
    main()
