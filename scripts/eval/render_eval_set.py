"""Render an eval sentence set with an exported Kural-TTS voice, in the baseline output format.

    python scripts/eval/render_eval_set.py --model exports/sahithi_v1 \\
        --out outputs/baseline/kural_sahithi_v1__cuda

The output directory has the same layout as scripts/baseline/run_baseline.py (wavs,
results.jsonl, meta.json), so scripts/baseline/score.py and make_listening_test.py score
Kural-TTS side by side with the Phase 1 baselines. Unlike the baselines, Kural gets the
**raw** text, including Tanglish, because its frontend handles it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))

from common import DEFAULT_OUT, DEFAULT_SENTENCES, Adapter, load_sentences, run_adapter

from kural_tts import KuralTTS


def kural_adapter(model_dir: Path, name: str) -> Adapter:
    state: dict = {}

    def load(device: str) -> None:
        state["tts"] = KuralTTS(model_path=model_dir, device=device)
        state["tts"].sample_rate  # noqa: B018 - force model load before timing

    def synth(text: str):
        return state["tts"].synthesize(text)

    return Adapter(
        name=name,
        hf_id=str(model_dir),
        license="Kural-TTS (project voice)",
        supports=("en", "ta", "cs", "tanglish"),
        load=load,
        synth=synth,
        notes="Kural frontend receives raw text (Tanglish included).",
    )


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", type=Path, required=True, help="dir with model.onnx + config.json")
    p.add_argument("--name", default="kural", help="run name in the comparison table")
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--sentences", type=Path, default=DEFAULT_SENTENCES)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args(argv)

    rows = load_sentences(args.sentences)
    for r in rows:
        # The shared harness feeds Tanglish rows their Tamil-script reference (baselines cannot
        # read Tanglish). Kural's frontend can, so point the reference back at the raw text.
        if r["category"] == "tanglish":
            r["reference_ta_script"] = r["text"]
    out = run_adapter(kural_adapter(args.model, args.name), rows, args.out_root, args.device)
    print(f"done -> {out}")
    return out


if __name__ == "__main__":
    main()
