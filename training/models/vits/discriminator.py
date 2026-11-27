"""Multi-period + scale discriminators (HiFi-GAN), used only during training."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from training.models.vits.commons import get_padding, weight_norm
from training.models.vits.modules import LRELU_SLOPE


class DiscriminatorP(nn.Module):
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3):
        super().__init__()
        self.period = period
        pad = (get_padding(kernel_size, 1), 0)
        chans = [1, 32, 128, 512, 1024]
        self.convs = nn.ModuleList(
            weight_norm(nn.Conv2d(chans[i], chans[i + 1], (kernel_size, 1), (stride, 1), pad))
            for i in range(4)
        )
        self.convs.append(weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, pad)))
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class DiscriminatorS(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv1d(1, 16, 15, 1, padding=7)),
                weight_norm(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                weight_norm(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                weight_norm(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                weight_norm(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                weight_norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor):
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: tuple[int, ...] = (2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorS(), *(DiscriminatorP(p) for p in periods)]
        )

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
