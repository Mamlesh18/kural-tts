"""Run one baseline model over the feasibility sentence set.

Usage (inside that model's venv, see scripts/baseline/envs/):
    python scripts/baseline/run_baseline.py --model mms_tam --device cuda
    python scripts/baseline/run_baseline.py --model mms_tam --device cpu --limit 10   # CPU RTF
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapters import ADAPTERS
from common import (
    DEFAULT_OUT,
    DEFAULT_SENTENCES,
    limit_per_category,
    load_sentences,
    run_adapter,
)


def _peak_mem_fn(device: str):
    if device != "cuda":
        return None
    import torch

    torch.cuda.reset_peak_memory_stats()
    return lambda: torch.cuda.max_memory_allocated() / 1e9


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=sorted(ADAPTERS))
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--sentences", type=Path, default=DEFAULT_SENTENCES)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--limit", type=int, default=0, help="Per-category limit (0 = all).")
    args = p.parse_args()

    rows = load_sentences(args.sentences)
    if args.limit:
        rows = limit_per_category(rows, args.limit)

    out = run_adapter(
        ADAPTERS[args.model], rows, args.out, args.device, peak_mem_fn=_peak_mem_fn(args.device)
    )
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
