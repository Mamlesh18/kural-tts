"""Checkpointing built for Colab: atomic writes, rotation, auto-resume, partial warm start."""

from __future__ import annotations

import os
import re
from pathlib import Path

import torch

_CKPT_RE = re.compile(r"^ckpt_(\d+)\.pt$")


def list_checkpoints(run_dir: str | Path) -> list[Path]:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return []
    found = [(int(m.group(1)), p) for p in run_dir.iterdir() if (m := _CKPT_RE.match(p.name))]
    return [p for _, p in sorted(found)]


def latest_checkpoint(run_dir: str | Path) -> Path | None:
    ckpts = list_checkpoints(run_dir)
    return ckpts[-1] if ckpts else None


def save_checkpoint(run_dir: str | Path, step: int, state: dict, keep: int = 3) -> Path:
    """Write ckpt_<step>.pt atomically (tmp + rename) and keep only the newest ``keep``."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"ckpt_{step}.pt"
    tmp = path.with_suffix(".pt.tmp")
    torch.save({**state, "step": step}, tmp)
    os.replace(tmp, path)
    for old in list_checkpoints(run_dir)[:-keep]:
        old.unlink(missing_ok=True)
    return path


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def load_matching(module: torch.nn.Module, state_dict: dict) -> tuple[list[str], list[str]]:
    """Copy tensors whose name and shape match (warm start). Returns (loaded, skipped)."""
    own = module.state_dict()
    loaded, skipped = [], []
    for name, tensor in state_dict.items():
        if name in own and own[name].shape == tensor.shape:
            own[name] = tensor
            loaded.append(name)
        else:
            skipped.append(name)
    module.load_state_dict(own)
    return loaded, skipped
