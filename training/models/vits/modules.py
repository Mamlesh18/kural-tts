"""Core layers: LayerNorm, WaveNet-style WN, HiFi-GAN ResBlock, coupling layer, Flip."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from training.models.vits.commons import (
    fused_add_tanh_sigmoid_multiply,
    get_padding,
    init_weights,
    remove_weight_norm,
    weight_norm,
)

LRELU_SLOPE = 0.1


class LayerNorm(nn.Module):
    """LayerNorm over the channel axis of [B, C, T]."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (x.size(-1),), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class WN(nn.Module):
    """Non-causal WaveNet stack with gated activations and optional global conditioning."""

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)
        self.cond_layer = (
            weight_norm(nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1))
            if gin_channels
            else None
        )
        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = (kernel_size * dilation - dilation) // 2
            self.in_layers.append(
                weight_norm(
                    nn.Conv1d(
                        hidden_channels,
                        2 * hidden_channels,
                        kernel_size,
                        dilation=dilation,
                        padding=padding,
                    )
                )
            )
            res_skip = 2 * hidden_channels if i < n_layers - 1 else hidden_channels
            self.res_skip_layers.append(weight_norm(nn.Conv1d(hidden_channels, res_skip, 1)))

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None
    ) -> torch.Tensor:
        output = torch.zeros_like(x)
        n_channels = self.hidden_channels
        if g is not None and self.cond_layer is not None:
            g = self.cond_layer(g)
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None and self.cond_layer is not None:
                offset = i * 2 * self.hidden_channels
                g_l = g[:, offset : offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)
            acts = self.drop(fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels))
            res_skip = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = (x + res_skip[:, : self.hidden_channels, :]) * x_mask
                output = output + res_skip[:, self.hidden_channels :, :]
            else:
                output = output + res_skip
        return output * x_mask

    def remove_weight_norm(self) -> None:
        if self.cond_layer is not None:
            remove_weight_norm(self.cond_layer)
        for layer in [*self.in_layers, *self.res_skip_layers]:
            remove_weight_norm(layer)


class ResBlock1(nn.Module):
    """HiFi-GAN V1 residual block (two convs per dilation)."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: tuple = (1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            weight_norm(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=d,
                    padding=get_padding(kernel_size, d),
                )
            )
            for d in dilation
        )
        self.convs2 = nn.ModuleList(
            weight_norm(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                )
            )
            for _ in dilation
        )
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None = None) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2, strict=True):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            x = c2(xt) + x
        return x if x_mask is None else x * x_mask

    def remove_weight_norm(self) -> None:
        for layer in [*self.convs1, *self.convs2]:
            remove_weight_norm(layer)


class ResidualCouplingLayer(nn.Module):
    """Affine (mean-only by default) coupling layer used in the VITS flow."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        mean_only: bool = True,
    ):
        super().__init__()
        assert channels % 2 == 0, "channels should be divisible by 2"
        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels)
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse: bool = False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.enc(self.pre(x0) * x_mask, x_mask, g=g)
        stats = self.post(h) * x_mask
        if self.mean_only:
            m, logs = stats, torch.zeros_like(stats)
        else:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            return torch.cat([x0, x1], 1), torch.sum(logs, [1, 2])
        x1 = (x1 - m) * torch.exp(-logs) * x_mask
        return torch.cat([x0, x1], 1)


class Flip(nn.Module):
    def forward(self, x, *args, reverse: bool = False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            return x, torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
        return x
