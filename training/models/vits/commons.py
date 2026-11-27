"""Shared helpers for Kural-VITS.

Architecture follows VITS (Kim et al., 2021) and its MIT-licensed reference implementation
https://github.com/jaywalnut310/vits. See NOTICE.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

__all__ = ["remove_weight_norm", "weight_norm"]


def remove_weight_norm(module: torch.nn.Module) -> None:
    """Fold the weight-norm parametrization into a plain weight (before export/inference)."""
    remove_parametrizations(module, "weight")


def init_weights(m: torch.nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    """Normal init for conv weights; handles weight-norm parametrized convs (g = ||v||)."""
    if "Conv" not in m.__class__.__name__:
        return
    params = getattr(m, "parametrizations", None)
    if params is not None and "weight" in params:
        v = params.weight.original1
        v.data.normal_(mean, std)
        g = v.data.flatten(1).norm(dim=1).view(-1, *[1] * (v.dim() - 1))
        params.weight.original0.data.copy_(g)
    else:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


def sequence_mask(lengths: torch.Tensor, max_length=None) -> torch.Tensor:
    # max_length stays a tensor when omitted, so ONNX export keeps output length dynamic.
    if max_length is None:
        max_length = lengths.max()
    x = torch.arange(max_length, dtype=lengths.dtype, device=lengths.device)
    return x.unsqueeze(0) < lengths.unsqueeze(1)


def generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Hard monotonic alignment from integer durations.

    duration: [B, 1, T_x], mask: [B, 1, T_y, T_x] -> path [B, 1, T_y, T_x]
    """
    b, _, t_y, t_x = mask.shape
    cum = torch.cumsum(duration, -1).view(b * t_x)
    path = sequence_mask(cum, t_y).to(mask.dtype).view(b, t_x, t_y)
    path = path - F.pad(path, (0, 0, 1, 0, 0, 0))[:, :-1]
    return path.unsqueeze(1).transpose(2, 3) * mask


def slice_segments(x: torch.Tensor, ids_str: torch.Tensor, segment_size: int) -> torch.Tensor:
    out = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        start = int(ids_str[i])
        out[i] = x[i, :, start : start + segment_size]
    return out


def rand_slice_segments(
    x: torch.Tensor, x_lengths: torch.Tensor, segment_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    b = x.size(0)
    ids_str_max = torch.clamp(x_lengths - segment_size + 1, min=1)
    ids_str = (torch.rand(b, device=x.device) * ids_str_max).long()
    return slice_segments(x, ids_str, segment_size), ids_str


def fused_add_tanh_sigmoid_multiply(
    input_a: torch.Tensor, input_b: torch.Tensor, n_channels: int
) -> torch.Tensor:
    n = n_channels
    acts = input_a + input_b
    return torch.tanh(acts[:, :n, :]) * torch.sigmoid(acts[:, n:, :])
