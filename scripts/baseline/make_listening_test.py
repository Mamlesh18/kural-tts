"""Build a blind, randomized listening test from baseline outputs.

Writes:
  outputs/listening_test/audio/<random>.wav     anonymized copies
  outputs/listening_test/ratings.csv            what listeners fill in (no model names)
  outputs/listening_test/KEY_DO_NOT_SHARE.csv   mapping back to model/run (keep away from raters)

Listeners rate each clip on:
  mos (1-5 naturalness), pron_errors (count of mispronounced words), notes.
For CS/Tanglish clips also: switch_naturalness (1-5): does the voice switch language smoothly?
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import secrets
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DEFAULT_OUT, REPO_ROOT, load_sentences


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--dest", type=Path, default=REPO_ROOT / "outputs" / "listening_test")
    p.add_argument("--runs", nargs="*", help="Run dir names to include (default: all *__cuda).")
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args()

    texts = {r["id"]: r["text"] for r in load_sentences()}
    runs = args.runs or [d.name for d in args.out.iterdir() if d.name.endswith("__cuda")]
    items = []
    for run in runs:
        for line in (args.out / run / "results.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            if r["status"] == "ok":
                items.append((run, r))

    random.Random(args.seed).shuffle(items)
    audio_dir = args.dest / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    with (
        (args.dest / "ratings.csv").open("w", encoding="utf-8", newline="") as rf,
        (args.dest / "KEY_DO_NOT_SHARE.csv").open("w", encoding="utf-8", newline="") as kf,
    ):
        rw, kw = csv.writer(rf), csv.writer(kf)
        rw.writerow(
            ["clip", "category", "text", "mos", "pron_errors", "switch_naturalness", "notes"]
        )
        kw.writerow(["clip", "run", "sentence_id"])
        for run, r in items:
            clip = f"{secrets.token_hex(4)}.wav"
            shutil.copy(args.out / run / r["wav"], audio_dir / clip)
            rw.writerow([clip, r["category"], texts.get(r["id"], r["input_text"]), "", "", "", ""])
            kw.writerow([clip, run, r["id"]])
    print(f"{len(items)} clips -> {args.dest}")


if __name__ == "__main__":
    main()
