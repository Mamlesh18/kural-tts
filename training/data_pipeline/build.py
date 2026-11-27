"""Build the training dataset from raw recordings.

    python -m training.data_pipeline.build \\
        --raw recordings/ --prompts recording/prompts.tsv --out data/sahithi_v1 \\
        [--corrections data/corrections.tsv]

Inputs
  raw/**/<prompt_id>.wav       one file per prompt (48 kHz/24-bit recommended); extra takes may be
                               named <prompt_id>_take2.wav, and the best take by SNR is kept.
                               The parent directory name is recorded as the session.
  prompts.tsv                  id <TAB> category <TAB> text (as written by recording/make_script.py)
  corrections.tsv (optional)   id <TAB> text: what was actually said, from transcript review

Outputs (under --out)
  wavs/<id>.wav                trimmed, loudness-normalized, 22.05 kHz 16-bit mono
  manifests/all.jsonl          every utterance with QC fields and status pass/review/reject
  manifests/{train,val,test}.jsonl   status == pass only
  manifests/test_ids.lock      test set membership, pinned forever once written
  qc_report.md                 summary per category and reasons
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from training.data_pipeline.audio_qc import load_mono, measure, normalize_loudness, trim_silence

REPO = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO / "eval" / "sentences"
_TAKE_RE = re.compile(r"^(?P<id>.+?)(?:_take(?P<take>\d+))?$")


@dataclass
class QCRules:
    sample_rate: int = 22050
    min_duration: float = 0.8
    max_duration: float = 15.0
    reject_snr_db: float = 25.0
    review_snr_db: float = 35.0
    review_pause_sec: float = 1.5
    cps_z_review: float = 3.0  # characters-per-second outlier threshold (per category)
    test_frac: float = 0.03
    val_frac: float = 0.02


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def norm_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return " ".join(text.split())


def read_tsv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def eval_texts(eval_dir: Path = EVAL_DIR) -> set[str]:
    texts: set[str] = set()
    for tsv in eval_dir.glob("*.tsv"):
        for row in read_tsv(tsv):
            for col in ("text", "reference_ta_script"):
                if row.get(col):
                    texts.add(norm_text(row[col]))
    return texts


def split_for(utt_id: str, locked_test: set[str], rules: QCRules) -> str:
    if utt_id in locked_test:
        return "test"
    frac = int(hashlib.sha1(utt_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if frac < rules.test_frac:
        return "test"
    if frac < rules.test_frac + rules.val_frac:
        return "val"
    return "train"


def find_takes(raw_dir: Path) -> dict[str, list[Path]]:
    takes: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(raw_dir.rglob("*")):
        if path.suffix.lower() in (".wav", ".flac"):
            takes[_TAKE_RE.match(path.stem).group("id")].append(path)
    return takes


def process_take(path: Path, rules: QCRules) -> tuple[np.ndarray, dict]:
    wav, sr = load_mono(path, target_sr=rules.sample_rate)
    raw_stats = measure(wav, sr)  # clipping is judged on the untouched signal
    wav = normalize_loudness(trim_silence(wav, sr), sr)
    stats = measure(wav, sr).to_dict()
    stats["clip_runs"] = raw_stats.clip_runs
    return wav, stats


def judge(row: dict, rules: QCRules) -> tuple[str, list[str]]:
    reasons_reject, reasons_review = [], []
    if row["clip_runs"] > 0:
        reasons_reject.append("clipping")
    if row["duration"] < rules.min_duration:
        reasons_reject.append("too_short")
    if row["duration"] > rules.max_duration:
        reasons_reject.append("too_long")
    if row["snr_db"] < rules.reject_snr_db:
        reasons_reject.append("low_snr")
    elif row["snr_db"] < rules.review_snr_db:
        reasons_review.append("snr")
    if row["longest_pause"] > rules.review_pause_sec:
        reasons_review.append("long_pause")
    if reasons_reject:
        return "reject", reasons_reject + reasons_review
    return ("review", reasons_review) if reasons_review else ("pass", [])


def flag_rate_outliers(rows: list[dict], rules: QCRules) -> None:
    """Mark utterances with unusual characters/second for review (misread or wrong prompt)."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["status"] == "pass" and r["duration"] > 0:
            r["cps"] = round(len(r["text"]) / r["duration"], 2)
            by_cat[r["category"]].append(r)
    for group in by_cat.values():
        if len(group) < 10:
            continue
        cps = np.array([r["cps"] for r in group])
        mu, sd = cps.mean(), cps.std() + 1e-6
        for r in group:
            if abs(r["cps"] - mu) / sd > rules.cps_z_review:
                r["status"], r["reasons"] = "review", [*r["reasons"], "speaking_rate"]


def build(
    raw_dir: Path,
    prompts_path: Path,
    out_dir: Path,
    corrections_path: Path | None = None,
    rules: QCRules | None = None,
    eval_dir: Path = EVAL_DIR,
) -> list[dict]:
    rules = rules or QCRules()
    prompts = {r["id"]: r for r in read_tsv(prompts_path)}
    corrections = (
        {r["id"]: r["text"] for r in read_tsv(corrections_path)} if corrections_path else {}
    )
    banned = eval_texts(eval_dir)
    (out_dir / "wavs").mkdir(parents=True, exist_ok=True)
    manifests = out_dir / "manifests"
    manifests.mkdir(exist_ok=True)
    lock_path = manifests / "test_ids.lock"
    locked = set(lock_path.read_text("utf-8").split()) if lock_path.exists() else set()

    rows: list[dict] = []
    for utt_id, paths in find_takes(raw_dir).items():
        prompt = prompts.get(utt_id)
        if prompt is None:
            print(f"warning: {utt_id} has audio but no prompt; skipped")
            continue
        best = None
        for path in paths:
            wav, stats = process_take(path, rules)
            if best is None or stats["snr_db"] > best[2]["snr_db"]:
                best = (path, wav, stats)
        path, wav, stats = best
        text = corrections.get(utt_id, prompt["text"])
        row = {
            "id": utt_id,
            "audio": f"wavs/{utt_id}.wav",
            "text": text,
            "category": prompt["category"],
            "speaker": "sahithi",
            "sr": rules.sample_rate,
            "source": path.parent.name,
            "raw_file": str(path.relative_to(raw_dir)),
            "raw_sha256": sha256(path),
            "n_takes": len(paths),
            "corrected": utt_id in corrections,
            **stats,
        }
        row["status"], row["reasons"] = judge(row, rules)
        if norm_text(text) in banned:
            row["status"], row["reasons"] = "reject", [*row["reasons"], "eval_overlap"]
        sf.write(out_dir / row["audio"], wav, rules.sample_rate, subtype="PCM_16")
        rows.append(row)

    flag_rate_outliers(rows, rules)
    _dedupe_texts(rows)
    for r in rows:
        r["split"] = split_for(r["id"], locked, rules)
        r["quality"] = r["status"]
    _write_outputs(rows, manifests, lock_path, locked)
    (out_dir / "qc_report.md").write_text(qc_report(rows), encoding="utf-8")
    return rows


def _dedupe_texts(rows: list[dict]) -> None:
    """Same sentence recorded under different ids: keep the best SNR, mark others duplicate."""
    by_text: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["status"] != "reject":
            by_text[norm_text(r["text"])].append(r)
    for group in by_text.values():
        group.sort(key=lambda r: -r["snr_db"])
        for r in group[1:]:
            r["status"], r["reasons"] = "reject", [*r["reasons"], "duplicate_text"]


def _write_outputs(rows: list[dict], manifests: Path, lock_path: Path, locked: set[str]) -> None:
    def dump(name: str, items: list[dict]) -> None:
        with (manifests / name).open("w", encoding="utf-8") as f:
            for r in sorted(items, key=lambda r: r["id"]):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump("all.jsonl", rows)
    for split in ("train", "val", "test"):
        dump(f"{split}.jsonl", [r for r in rows if r["status"] == "pass" and r["split"] == split])
    test_ids = locked | {r["id"] for r in rows if r["split"] == "test"}
    lock_path.write_text("\n".join(sorted(test_ids)) + "\n", encoding="utf-8")


def qc_report(rows: list[dict]) -> str:
    status = Counter(r["status"] for r in rows)
    reasons = Counter(reason for r in rows for reason in r["reasons"])
    lines = ["# Data QC report", "", f"Utterances: {len(rows)}", ""]
    lines += ["| Status | Count |", "|---|---|"] + [f"| {k} | {v} |" for k, v in status.items()]
    lines += ["", "## Hours by category (pass only)", "", "| Category | Utterances | Hours |",
              "|---|---|---|"]  # fmt: skip
    by_cat: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["status"] == "pass":
            by_cat[r["category"]].append(r["duration"])
    for cat, durs in sorted(by_cat.items()):
        lines.append(f"| {cat} | {len(durs)} | {sum(durs) / 3600:.2f} |")
    lines += ["", "## Split (pass only)", ""]
    for split in ("train", "val", "test"):
        n = sum(1 for r in rows if r["status"] == "pass" and r["split"] == split)
        lines.append(f"- {split}: {n}")
    lines += ["", "## Reasons", "", "| Reason | Count |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in reasons.most_common()]
    review = [r for r in rows if r["status"] == "review"]
    if review:
        lines += ["", "## Needs listening review", ""]
        lines += [f"- `{r['id']}` ({', '.join(r['reasons'])}): {r['text']}" for r in review[:200]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--corrections", type=Path)
    p.add_argument("--sample-rate", type=int, default=22050)
    args = p.parse_args(argv)
    rows = build(args.raw, args.prompts, args.out, args.corrections,
                 QCRules(sample_rate=args.sample_rate))  # fmt: skip
    status = Counter(r["status"] for r in rows)
    print(f"{len(rows)} utterances: {dict(status)} -> {args.out}")


if __name__ == "__main__":
    main()
