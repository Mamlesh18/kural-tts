"""Text encoder (per-phoneme language embeddings), posterior encoder, flow, duration predictor."""

from __future__ import annotations

import math

import torch
from torch import nn

from training.models.vits import attentions
from training.models.vits.commons import sequence_mask
from training.models.vits.modules import WN, Flip, LayerNorm, ResidualCouplingLayer


class TextEncoder(nn.Module):
    """Phoneme ids + language ids -> hidden states and prior statistics (m_p, logs_p).

    The language embedding is added **per phoneme**, not per sentence. That is what lets one
    utterance switch between Tamil and English pronunciation mid-sentence.
    """

    def __init__(
        self,
        n_vocab: int,
        n_langs: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        self.lang_emb = nn.Embedding(n_langs, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)
        nn.init.normal_(self.lang_emb.weight, 0.0, hidden_channels**-0.5)
        self.encoder = attentions.Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor, lang: torch.Tensor):
        h = (self.emb(x) + self.lang_emb(lang)) * math.sqrt(self.hidden_channels)  # [b, t, h]
        h = h.transpose(1, -1)  # [b, h, t]
        x_mask = sequence_mask(x_lengths, h.size(2)).unsqueeze(1).to(h.dtype)
        h = self.encoder(h * x_mask, x_mask)
        stats = self.proj(h) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return h, m, logs, x_mask


class PosteriorEncoder(nn.Module):
    """Linear spectrogram -> latent z ~ q(z|x) (training only)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)
        h = self.enc(self.pre(x) * x_mask, x_mask, g=g)
        stats = self.proj(h) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
    """Normalizing flow between the posterior latent z and the prior space z_p."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels
                )
            )
            self.flows.append(Flip())

    def forward(self, x, x_mask, g=None, reverse: bool = False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=False)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=True)
        return x


class DurationPredictor(nn.Module):
    """Deterministic log-duration predictor (ONNX-friendly; stochastic DP is a later option)."""

    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        gin_channels: int = 0,
    ):
        super().__init__()
        pad = kernel_size // 2
        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=pad)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=pad)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)
        self.cond = nn.Conv1d(gin_channels, in_channels, 1) if gin_channels else None

    def forward(self, x, x_mask, g=None):
        x = torch.detach(x)
        if g is not None and self.cond is not None:
            x = x + self.cond(torch.detach(g))
        x = self.drop(self.norm_1(torch.relu(self.conv_1(x * x_mask))))
        x = self.drop(self.norm_2(torch.relu(self.conv_2(x * x_mask))))
        return self.proj(x * x_mask) * x_mask
