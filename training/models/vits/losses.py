"""VITS training losses."""

from __future__ import annotations

import torch


def feature_loss(fmap_r, fmap_g) -> torch.Tensor:
    loss = 0.0
    for dr, dg in zip(fmap_r, fmap_g, strict=True):
        for rl, gl in zip(dr, dg, strict=True):
            loss = loss + torch.mean(torch.abs(rl.float().detach() - gl.float()))
    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0.0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs, strict=True):
        loss = loss + torch.mean((1 - dr.float()) ** 2) + torch.mean(dg.float() ** 2)
    return loss


def generator_loss(disc_outputs):
    loss = 0.0
    for dg in disc_outputs:
        loss = loss + torch.mean((1 - dg.float()) ** 2)
    return loss


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask) -> torch.Tensor:
    """KL(q(z|x) || p(z|c)) evaluated on the flow-transformed posterior sample z_p."""
    z_p, logs_q, m_p, logs_p = (t.float() for t in (z_p, logs_q, m_p, logs_p))
    kl = logs_p - logs_q - 0.5
    kl = kl + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    return torch.sum(kl * z_mask) / torch.sum(z_mask)
