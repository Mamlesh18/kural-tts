"""Score baseline outputs with ASR (CER/WER) and optional UTMOS, then write a summary table.

Usage (in the `scoring` venv, see envs/scoring.txt):
    python scripts/baseline/score.py                     # score every outputs/baseline/*/ dir
    python scripts/baseline/score.py --utmos             # + UTMOS naturalness for English
    python scripts/baseline/score.py --summary-only      # rebuild the table from existing scores

ASR models:
  en -> openai/whisper-large-v3 (language forced to English)
  ta, cs, tanglish -> ai4bharat/indic-conformer-600m-multilingual (CTC), falling back to
                      Whisper with language=ta if IndicConformer fails to load.

ASR-based CER on code-switched speech is only a rough signal, because English words come back
transliterated. The same bias applies to every model, so the numbers compare models against each
other rather than measure absolute quality. Human listening (make_listening_test.py) is the primary
metric for CS.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DEFAULT_OUT, REPO_ROOT, cer, wer


class ASR:
    def __init__(self, device: str):
        self.device = device
        self._whisper = None
        self._indic = None
        self.indic_name = None

    def _load_audio_16k(self, path: Path):
        import soundfile as sf
        import torch
        import torchaudio

        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        wav = torch.from_numpy(np.asarray(audio)).unsqueeze(0)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        return wav

    def whisper(self, path: Path, language: str) -> str:
        if self._whisper is None:
            import torch
            from transformers import pipeline

            self._whisper = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-large-v3",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device=0 if self.device == "cuda" else -1,
            )
        wav = self._load_audio_16k(path).squeeze(0).numpy()
        out = self._whisper(
            {"raw": wav, "sampling_rate": 16000},
            generate_kwargs={"language": language, "task": "transcribe"},
        )
        return out["text"].strip()

    def tamil(self, path: Path) -> str:
        if self._indic is None:
            try:
                from transformers import AutoModel

                self._indic = AutoModel.from_pretrained(
                    "ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True
                )
                self.indic_name = "indic-conformer-600m (ctc)"
            except Exception as e:
                print(f"IndicConformer unavailable ({e}); falling back to Whisper ta")
                self._indic = False
                self.indic_name = "whisper-large-v3 (ta)"
        if self._indic is False:
            return self.whisper(path, "ta")
        return str(self._indic(self._load_audio_16k(path), "ta", "ctc")).strip()


def reference_text(row: dict) -> str:
    return row["input_text"]  # exactly what the model was asked to say


def score_dir(run_dir: Path, asr: ASR, utmos_model=None) -> list[dict]:
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").open(encoding="utf-8")]
    scored = []
    for r in rows:
        if r["status"] != "ok":
            scored.append(r)
            continue
        wav = run_dir / r["wav"]
        if r["category"] == "en":
            r["asr_model"] = "whisper-large-v3 (en)"
            r["asr_text"] = asr.whisper(wav, "en")
        else:
            r["asr_text"] = asr.tamil(wav)
            r["asr_model"] = asr.indic_name
        ref = reference_text(r)
        r["cer"] = round(cer(ref, r["asr_text"]), 4)
        r["wer"] = round(wer(ref, r["asr_text"]), 4)
        if utmos_model is not None and r["category"] == "en":
            r["utmos"] = round(utmos_model(wav), 3)
        scored.append(r)
        print(f"{run_dir.name} {r['id']} cer={r['cer']:.3f}", flush=True)
    with (run_dir / "scores.jsonl").open("w", encoding="utf-8") as f:
        for r in scored:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return scored


def load_utmos(device: str):
    import soundfile as sf
    import torch
    import torchaudio

    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    predictor = predictor.to(device).eval()

    def score(path: Path) -> float:
        audio, sr = sf.read(path, dtype="float32")
        wav = torch.from_numpy(audio).unsqueeze(0)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        with torch.no_grad():
            return float(predictor(wav.to(device), 16000).item())

    return score


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return f"{np.mean(xs):.3f}" if xs else "–"


def summarize(out_root: Path, report: Path) -> None:
    lines = [
        "# Feasibility baseline: automatic scores",
        "",
        "Generated by `scripts/baseline/score.py`. CER is ASR-based, so lower is better and the",
        "numbers are only comparable between models. Human MOS lives in `reports/feasibility.md`.",
        "",
        "| Run | License (as stated) | OK/Err/Skip | Mean RTF | p50 synth s | Peak GB | "
        "CER en | CER ta | CER cs | CER tanglish | UTMOS en |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run_dir in sorted(p for p in out_root.iterdir() if (p / "meta.json").exists()):
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        scores_path = run_dir / "scores.jsonl"
        by_cat: dict[str, list] = defaultdict(list)
        utmos = []
        if scores_path.exists():
            for line in scores_path.open(encoding="utf-8"):
                r = json.loads(line)
                if "cer" in r:
                    by_cat[r["category"]].append(r["cer"])
                if "utmos" in r:
                    utmos.append(r["utmos"])
        lines.append(
            f"| {run_dir.name} | {meta['license_as_stated']} | "
            f"{meta['n_ok']}/{meta['n_error']}/{meta['n_skipped']} | {meta['mean_rtf']} | "
            f"{meta['p50_synth_sec']} | {meta['peak_mem_gb'] or '–'} | "
            + " | ".join(_mean(by_cat.get(c, [])) for c in ("en", "ta", "cs", "tanglish"))
            + f" | {_mean(utmos)} |"
        )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {report}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--utmos", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--report", type=Path, default=REPO_ROOT / "reports" / "feasibility_scores.md")
    args = p.parse_args()

    if not args.summary_only:
        asr = ASR(args.device)
        utmos = load_utmos(args.device) if args.utmos else None
        for run_dir in sorted(p for p in args.out.iterdir() if (p / "results.jsonl").exists()):
            score_dir(run_dir, asr, utmos)
    summarize(args.out, args.report)


if __name__ == "__main__":
    main()
