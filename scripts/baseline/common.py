"""Shared harness for Phase 1 baseline runs: sentence loading, timing, output layout, CER/WER.

Kept free of torch/model imports so it can be unit-tested locally without a GPU.
"""

from __future__ import annotations

import csv
import json
import re
import time
import unicodedata
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SENTENCES = REPO_ROOT / "eval" / "sentences" / "feasibility_v1.tsv"
DEFAULT_OUT = REPO_ROOT / "outputs" / "baseline"
CATEGORIES = ("en", "ta", "cs", "tanglish")


# ---------------------------------------------------------------------------- sentences
def load_sentences(path: Path = DEFAULT_SENTENCES) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    for r in rows:
        r["reference_ta_script"] = r.get("reference_ta_script") or ""
    return rows


def limit_per_category(rows: list[dict], n: int) -> list[dict]:
    counts: dict[str, int] = {}
    kept = []
    for r in rows:
        if counts.get(r["category"], 0) < n:
            counts[r["category"]] = counts.get(r["category"], 0) + 1
            kept.append(r)
    return kept


def model_input(row: dict) -> str:
    """Text given to a baseline model.

    No baseline model reads Tanglish natively, so Tanglish rows are fed their reviewed Tamil-script
    reference. This measures the TTS, not transliteration (which Kural's frontend handles).
    """
    if row["category"] == "tanglish":
        return row["reference_ta_script"]
    return row["text"]


# ---------------------------------------------------------------------------- adapters
@dataclass
class Adapter:
    """A baseline model wrapper. ``synth`` returns (float32 mono waveform, sample_rate)."""

    name: str
    hf_id: str
    license: str
    supports: tuple[str, ...]
    load: Callable[[str], None]
    synth: Callable[[str], tuple[np.ndarray, int]]
    notes: str = ""


@dataclass
class Result:
    id: str
    category: str
    input_text: str
    status: str  # ok | skipped | error
    wav: str = ""
    sample_rate: int = 0
    audio_sec: float = 0.0
    synth_sec: float = 0.0
    rtf: float = 0.0
    error: str = ""
    extra: dict = field(default_factory=dict)


def to_float32_mono(audio) -> np.ndarray:
    a = np.asarray(audio)
    if a.dtype == np.int16:
        a = a.astype(np.float32) / 32768.0
    a = a.astype(np.float32).squeeze()
    if a.ndim > 1:  # (channels, samples) or (samples, channels)
        a = a.mean(axis=0 if a.shape[0] < a.shape[-1] else -1)
    return a


def run_adapter(
    adapter: Adapter,
    rows: list[dict],
    out_root: Path,
    device: str,
    warmup: int = 1,
    peak_mem_fn: Callable[[], float] | None = None,
) -> Path:
    """Synthesize every row, write wavs + results.jsonl + meta.json. Returns the output dir."""
    import soundfile as sf

    out_dir = Path(out_root) / f"{adapter.name}__{device}"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    adapter.load(device)
    load_sec = time.perf_counter() - t0

    supported = [r for r in rows if r["category"] in adapter.supports]
    for r in supported[:warmup]:  # warm-up excluded from timing
        adapter.synth(model_input(r))

    results: list[Result] = []
    for r in rows:
        text = model_input(r)
        res = Result(id=r["id"], category=r["category"], input_text=text, status="skipped")
        if r["category"] in adapter.supports:
            try:
                t = time.perf_counter()
                audio, sr = adapter.synth(text)
                res.synth_sec = time.perf_counter() - t
                audio = to_float32_mono(audio)
                wav = out_dir / f"{r['id']}.wav"
                sf.write(wav, audio, sr, subtype="PCM_16")
                res.status, res.wav, res.sample_rate = "ok", wav.name, sr
                res.audio_sec = len(audio) / sr
                res.rtf = res.synth_sec / res.audio_sec if res.audio_sec else 0.0
            except Exception as e:  # keep going; record failure
                res.status, res.error = "error", f"{type(e).__name__}: {e}"
        results.append(res)
        print(f"[{adapter.name}] {res.id} {res.status} rtf={res.rtf:.3f}", flush=True)

    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(asdict(res), ensure_ascii=False) + "\n")

    ok = [r for r in results if r.status == "ok"]
    meta = {
        "model": adapter.name,
        "hf_id": adapter.hf_id,
        "license_as_stated": adapter.license,
        "supports": list(adapter.supports),
        "notes": adapter.notes,
        "device": device,
        "load_sec": round(load_sec, 2),
        "n_ok": len(ok),
        "n_error": sum(r.status == "error" for r in results),
        "n_skipped": sum(r.status == "skipped" for r in results),
        "mean_rtf": round(float(np.mean([r.rtf for r in ok])), 4) if ok else None,
        "p50_synth_sec": round(float(np.median([r.synth_sec for r in ok])), 3) if ok else None,
        "peak_mem_gb": round(peak_mem_fn(), 2) if peak_mem_fn else None,
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


# ---------------------------------------------------------------------------- metrics
_SPACE_RE = re.compile(r"\s+")


def normalize_for_cer(text: str) -> str:
    """NFC, lowercase, drop punctuation/symbols (keep Tamil combining marks), collapse spaces."""
    text = unicodedata.normalize("NFC", text).lower()
    kept = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat[0] in "PS":
            kept.append(" ")
        elif cat == "Cf":  # zero-width joiners etc.
            continue
        else:
            kept.append(ch)
    return _SPACE_RE.sub(" ", "".join(kept)).strip()


def edit_distance(a: list | str, b: list | str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    r, h = normalize_for_cer(ref).replace(" ", ""), normalize_for_cer(hyp).replace(" ", "")
    return edit_distance(r, h) / max(len(r), 1)


def wer(ref: str, hyp: str) -> float:
    r, h = normalize_for_cer(ref).split(), normalize_for_cer(hyp).split()
    return edit_distance(r, h) / max(len(r), 1)
