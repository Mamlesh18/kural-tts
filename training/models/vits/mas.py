"""Monotonic Alignment Search (MAS), vectorized over the batch with NumPy.

Finds, for each utterance, the monotonic, non-skipping alignment of text tokens to latent frames
that maximizes the total log-likelihood. It runs under no_grad on CPU. The per-frame loop is
O(T_y) NumPy ops over [B, T_x] arrays, which is fast enough without Cython or numba.
"""

from __future__ import annotations

import numpy as np
import torch

_NEG = -1e9


def maximum_path(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """neg_cent, mask: [B, T_y, T_x] -> hard path [B, T_y, T_x] (0/1), same device/dtype."""
    device, dtype = neg_cent.device, neg_cent.dtype
    value = neg_cent.detach().to(torch.float64).cpu().numpy()
    valid = mask.detach().cpu().numpy() > 0
    b, t_y_max, _t_x_max = value.shape
    t_ys = valid[:, :, 0].sum(1)
    t_xs = valid[:, 0, :].sum(1)

    value = np.where(valid, value, _NEG)
    q = np.full_like(value, _NEG)
    q[:, 0, 0] = value[:, 0, 0]
    for y in range(1, t_y_max):
        prev = q[:, y - 1]
        shifted = np.concatenate([np.full((b, 1), _NEG), prev[:, :-1]], axis=1)
        q[:, y] = np.maximum(prev, shifted) + value[:, y]

    path = np.zeros_like(value, dtype=np.float32)
    x = (t_xs - 1).astype(np.int64)
    rows = np.arange(b)
    for y in range(t_y_max - 1, -1, -1):
        active = y < t_ys
        path[rows[active], y, x[active]] = 1.0
        if y == 0:
            break
        stay = q[rows, y - 1, x]
        move = q[rows, y - 1, np.maximum(x - 1, 0)]
        step = active & (x > 0) & ((x == y) | (stay < move))
        x = x - step.astype(np.int64)
    return torch.from_numpy(path).to(device=device, dtype=dtype)
