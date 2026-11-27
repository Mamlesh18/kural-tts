"""Kural-VITS synthesizer: training forward pass and inference."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
from torch import nn

from training.models.vits.commons import generate_path, rand_slice_segments, sequence_mask
from training.models.vits.encoders import (
    DurationPredictor,
    PosteriorEncoder,
    ResidualCouplingBlock,
    TextEncoder,
)
from training.models.vits.generator import Generator
from training.models.vits.mas import maximum_path


@dataclass
class ModelConfig:
    n_vocab: int
    n_langs: int
    spec_channels: int
    segment_frames: int = 32  # decoder training window, in latent frames
    inter_channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: float = 0.1
    resblock_kernel_sizes: list[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: list[list[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    upsample_rates: list[int] = field(default_factory=lambda: [8, 8, 2, 2])
    upsample_initial_channel: int = 512
    upsample_kernel_sizes: list[int] = field(default_factory=lambda: [16, 16, 4, 4])
    posterior_layers: int = 16
    flow_layers: int = 4
    n_flows: int = 4
    dp_filter_channels: int = 256
    n_speakers: int = 1
    gin_channels: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class SynthesizerTrn(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        gin = cfg.gin_channels if cfg.n_speakers > 1 else 0
        self.enc_p = TextEncoder(
            cfg.n_vocab,
            cfg.n_langs,
            cfg.inter_channels,
            cfg.hidden_channels,
            cfg.filter_channels,
            cfg.n_heads,
            cfg.n_layers,
            cfg.kernel_size,
            cfg.p_dropout,
        )
        self.dec = Generator(
            cfg.inter_channels,
            cfg.resblock_kernel_sizes,
            cfg.resblock_dilation_sizes,
            cfg.upsample_rates,
            cfg.upsample_initial_channel,
            cfg.upsample_kernel_sizes,
            gin_channels=gin,
        )
        self.enc_q = PosteriorEncoder(
            cfg.spec_channels,
            cfg.inter_channels,
            cfg.hidden_channels,
            5,
            1,
            cfg.posterior_layers,
            gin_channels=gin,
        )
        self.flow = ResidualCouplingBlock(
            cfg.inter_channels,
            cfg.hidden_channels,
            5,
            1,
            cfg.flow_layers,
            cfg.n_flows,
            gin_channels=gin,
        )
        self.dp = DurationPredictor(
            cfg.hidden_channels, cfg.dp_filter_channels, 3, 0.5, gin_channels=gin
        )
        self.emb_g = nn.Embedding(cfg.n_speakers, gin) if gin else None

    @property
    def hop_length(self) -> int:
        return math.prod(self.cfg.upsample_rates)

    def _speaker(self, sid: torch.Tensor | None) -> torch.Tensor | None:
        if self.emb_g is None or sid is None:
            return None
        return self.emb_g(sid).unsqueeze(-1)

    def forward(self, x, x_lengths, lang, y, y_lengths, sid=None):
        """Training pass. x/lang: [B, T_x] ids; y: [B, spec_channels, T_y] linear spectrogram."""
        g = self._speaker(sid)
        h, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, lang)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        with torch.no_grad():
            # log N(z_p; m_p, exp(logs_p)) for every (frame, token) pair, summed over channels
            s_p_sq_r = torch.exp(-2 * logs_p.float())
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p.float(), [1], keepdim=True)
            neg_cent2 = torch.matmul(-0.5 * (z_p.float() ** 2).transpose(1, 2), s_p_sq_r)
            neg_cent3 = torch.matmul(z_p.float().transpose(1, 2), m_p.float() * s_p_sq_r)
            neg_cent4 = torch.sum(-0.5 * (m_p.float() ** 2) * s_p_sq_r, [1], keepdim=True)
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4  # [B, T_y, T_x]
            attn_mask = x_mask.unsqueeze(2) * y_mask.unsqueeze(-1)  # [B, 1, T_y, T_x]
            attn = maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).to(m_p.dtype)

        w = attn.sum(2)  # frames per token, [B, 1, T_x]
        logw_ = torch.log(w + 1e-6) * x_mask
        logw = self.dp(h, x_mask, g=g)
        l_length = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(x_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = rand_slice_segments(z, y_lengths, self.cfg.segment_frames)
        o = self.dec(z_slice, g=g)
        return o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(
        self,
        x,
        x_lengths,
        lang,
        sid=None,
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
        max_frames: int | None = None,
    ):
        g = self._speaker(sid)
        h, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, lang)
        logw = self.dp(h, x_mask, g=g)
        w_ceil = torch.ceil(torch.exp(logw) * x_mask * length_scale)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(2) * y_mask.unsqueeze(-1)
        attn = generate_path(w_ceil, attn_mask)
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        z = z * y_mask
        if max_frames is not None:
            z = z[:, :, :max_frames]
        o = self.dec(z, g=g)
        return o, attn, y_mask

    def remove_weight_norm(self) -> None:
        """Fold weight norm for export/inference (training-only parts are left untouched)."""
        self.dec.remove_weight_norm()
        for flow in self.flow.flows:
            if hasattr(flow, "enc"):
                flow.enc.remove_weight_norm()
