"""Transformer encoder with windowed relative-position attention (as in VITS).

Relative embeddings are always padded by the sequence length, so there are no Python ``max()``
calls on tensor sizes. That keeps the ONNX graph dynamic in the input length.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from training.models.vits.modules import LayerNorm


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        window_size: int = 4,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for _ in range(n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels, hidden_channels, n_heads, p_dropout, window_size
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout)
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.drop(self.attn_layers[i](x, x, attn_mask))
            x = self.norm_layers_1[i](x + y)
            y = self.drop(self.ffn_layers[i](x, x_mask))
            x = self.norm_layers_2[i](x + y)
        return x * x_mask


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        window_size: int = 4,
    ):
        super().__init__()
        assert channels % n_heads == 0
        self.n_heads = n_heads
        self.window_size = window_size
        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)
        rel_stddev = self.k_channels**-0.5
        n_rel = window_size * 2 + 1
        self.emb_rel_k = nn.Parameter(torch.randn(1, n_rel, self.k_channels) * rel_stddev)
        self.emb_rel_v = nn.Parameter(torch.randn(1, n_rel, self.k_channels) * rel_stddev)
        for conv in (self.conv_q, self.conv_k, self.conv_v):
            nn.init.xavier_uniform_(conv.weight)

    def forward(self, x: torch.Tensor, c: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        q, k, v = self.conv_q(x), self.conv_k(c), self.conv_v(c)
        return self.conv_o(self._attention(q, k, v, attn_mask))

    def _attention(self, query, key, value, mask):
        b, d, t_s = key.shape
        t_t = query.size(2)
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        query = query / math.sqrt(self.k_channels)
        scores = torch.matmul(query, key.transpose(-2, -1))
        key_rel = self._relative_embeddings(self.emb_rel_k, t_s)
        rel_logits = torch.matmul(query, key_rel.unsqueeze(0).transpose(-2, -1))
        scores = scores + self._rel_to_abs(rel_logits)
        scores = scores.masked_fill(mask == 0, -1e4)
        p_attn = self.drop(F.softmax(scores, dim=-1))
        output = torch.matmul(p_attn, value)
        value_rel = self._relative_embeddings(self.emb_rel_v, t_s)
        output = output + torch.matmul(self._abs_to_rel(p_attn), value_rel.unsqueeze(0))
        return output.transpose(2, 3).contiguous().view(b, d, t_t)

    def _relative_embeddings(self, rel_emb: torch.Tensor, length) -> torch.Tensor:
        """Embeddings for relative positions -(L-1)..(L-1): [1, 2L-1, k]."""
        padded = F.pad(rel_emb, (0, 0, length, length, 0, 0))
        start = self.window_size + 1
        return padded[:, start : start + 2 * length - 1]

    @staticmethod
    def _rel_to_abs(x: torch.Tensor) -> torch.Tensor:
        """[b, h, l, 2l-1] -> [b, h, l, l]"""
        b, h, length, _ = x.shape
        x = F.pad(x, (0, 1))
        x_flat = F.pad(x.reshape(b, h, length * 2 * length), (0, length - 1))
        return x_flat.view(b, h, length + 1, 2 * length - 1)[:, :, :length, length - 1 :]

    @staticmethod
    def _abs_to_rel(x: torch.Tensor) -> torch.Tensor:
        """[b, h, l, l] -> [b, h, l, 2l-1]"""
        b, h, length, _ = x.shape
        x = F.pad(x, (0, length - 1))
        x_flat = F.pad(x.reshape(b, h, length * length + length * (length - 1)), (length, 0))
        return x_flat.view(b, h, length, 2 * length)[:, :, :, 1:]


class FFN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        pad = kernel_size // 2
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=pad)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=pad)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = self.drop(torch.relu(self.conv_1(x * x_mask)))
        return self.conv_2(x * x_mask) * x_mask
