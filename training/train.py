"""Train Kural-VITS.

    python -m training.train --config training/configs/base.yaml --run-dir runs/sahithi_v1

Re-running the same command resumes from the newest checkpoint in --run-dir, so Colab
disconnects only cost the steps since the last checkpoint. Use --init-from to warm-start from
another run's checkpoint (tensors with matching names and shapes are copied).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from kural_tts.text.g2p import text_to_ids
from training.audio import mel_spectrogram, spec_to_mel
from training.checkpoint import latest_checkpoint, load_checkpoint, load_matching, save_checkpoint
from training.config import Config, load_config
from training.dataset import BucketBatchSampler, TextAudioDataset, collate
from training.models.vits.commons import slice_segments
from training.models.vits.discriminator import MultiPeriodDiscriminator
from training.models.vits.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from training.models.vits.synthesizer import SynthesizerTrn


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def pick_precision(requested: str, device: torch.device) -> str:
    if device.type != "cuda":
        return "fp32"
    if requested != "auto":
        return requested
    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Trainer:
    def __init__(self, cfg: Config, run_dir: Path, device: torch.device):
        self.cfg = cfg
        self.run_dir = run_dir
        self.device = device
        self.precision = pick_precision(cfg.train.precision, device)
        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(self.precision)
        self.net_g = SynthesizerTrn(cfg.model).to(device)
        self.net_d = MultiPeriodDiscriminator().to(device)
        t = cfg.train
        self.opt_g = torch.optim.AdamW(self.net_g.parameters(), t.learning_rate, t.betas, t.eps)
        self.opt_d = torch.optim.AdamW(self.net_d.parameters(), t.learning_rate, t.betas, t.eps)
        self.sched_g = torch.optim.lr_scheduler.ExponentialLR(self.opt_g, t.lr_decay)
        self.sched_d = torch.optim.lr_scheduler.ExponentialLR(self.opt_d, t.lr_decay)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.precision == "fp16")
        self.step = 0
        self.epoch = 0
        self.writer = None

    # ------------------------------------------------------------------ state
    def state(self) -> dict:
        return {
            "net_g": self.net_g.state_dict(),
            "net_d": self.net_d.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "sched_g": self.sched_g.state_dict(),
            "sched_d": self.sched_d.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.epoch,
            "config": self.cfg.to_dict(),
        }

    def resume(self, path: Path) -> None:
        ckpt = load_checkpoint(path, map_location=self.device)
        self.net_g.load_state_dict(ckpt["net_g"])
        self.net_d.load_state_dict(ckpt["net_d"])
        self.opt_g.load_state_dict(ckpt["opt_g"])
        self.opt_d.load_state_dict(ckpt["opt_d"])
        self.sched_g.load_state_dict(ckpt["sched_g"])
        self.sched_d.load_state_dict(ckpt["sched_d"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.step, self.epoch = ckpt["step"], ckpt["epoch"]
        print(f"resumed from {path} (step {self.step}, epoch {self.epoch})")

    def warm_start(self, path: Path) -> None:
        ckpt = load_checkpoint(path, map_location=self.device)
        loaded, skipped = load_matching(self.net_g, ckpt["net_g"])
        if "net_d" in ckpt:
            load_matching(self.net_d, ckpt["net_d"])
        print(f"warm start from {path}: {len(loaded)} tensors loaded, {len(skipped)} skipped")

    def save(self) -> Path:
        return save_checkpoint(
            self.run_dir, self.step, self.state(), self.cfg.train.keep_checkpoints
        )

    # ------------------------------------------------------------------ training
    def _autocast(self):
        if self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(self.device.type, dtype=self.amp_dtype)

    def train_step(self, batch) -> dict[str, float]:
        cfg = self.cfg
        x, x_len, lang, spec, spec_len, wav, _ = (t.to(self.device) for t in batch)
        hop = cfg.audio.hop_length
        seg_frames = cfg.model.segment_frames

        with self._autocast():
            (
                y_hat,
                l_length,
                _attn,
                ids_slice,
                _x_mask,
                z_mask,
                (_z, z_p, m_p, logs_p, _m_q, logs_q),
            ) = self.net_g(x, x_len, lang, spec, spec_len)
            y = slice_segments(wav, ids_slice * hop, cfg.train.segment_size)
            y_d_r, y_d_g, _, _ = self.net_d(y, y_hat.detach())
            loss_disc = discriminator_loss(y_d_r, y_d_g)

        self.opt_d.zero_grad(set_to_none=True)
        self.scaler.scale(loss_disc).backward()
        self.scaler.unscale_(self.opt_d)
        grad_d = torch.nn.utils.clip_grad_norm_(self.net_d.parameters(), 1e6)
        self.scaler.step(self.opt_d)

        with self._autocast():
            y_mel = slice_segments(spec_to_mel(spec.float(), cfg.audio), ids_slice, seg_frames)
            y_hat_mel = mel_spectrogram(y_hat.squeeze(1).float(), cfg.audio)
            _, y_d_g, fmap_r, fmap_g = self.net_d(y, y_hat)
            loss_dur = torch.sum(l_length.float())
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * cfg.train.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * cfg.train.c_kl
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen = generator_loss(y_d_g)
            loss_g = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

        self.opt_g.zero_grad(set_to_none=True)
        self.scaler.scale(loss_g).backward()
        self.scaler.unscale_(self.opt_g)
        grad_g = torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 1e6)
        self.scaler.step(self.opt_g)
        self.scaler.update()

        return {
            "loss/d": loss_disc.item(),
            "loss/g_total": loss_g.item(),
            "loss/g_adv": loss_gen.item(),
            "loss/fm": loss_fm.item(),
            "loss/mel": loss_mel.item(),
            "loss/dur": loss_dur.item(),
            "loss/kl": loss_kl.item(),
            "grad/g": float(grad_g),
            "grad/d": float(grad_d),
        }

    # ------------------------------------------------------------------ eval
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader | None) -> dict[str, float]:
        self.net_g.eval()
        metrics: dict[str, float] = {}
        if val_loader is not None:
            losses = []
            for batch in val_loader:
                x, x_len, lang, spec, spec_len, _, _ = (t.to(self.device) for t in batch)
                _, l_length, *_ = self.net_g(x, x_len, lang, spec, spec_len)
                losses.append(l_length.mean().item())
            metrics["val/dur"] = float(np.mean(losses)) if losses else math.nan
        for i, text in enumerate(self.cfg.train.eval_texts):
            ids, langs = text_to_ids(text)
            x = torch.tensor([ids], device=self.device)
            lang = torch.tensor([langs], device=self.device)
            audio, _, _ = self.net_g.infer(x, torch.tensor([len(ids)], device=self.device), lang)
            if self.writer is not None:
                self.writer.add_audio(
                    f"eval/text_{i}", audio[0].float().cpu().clamp(-1, 1), self.step,
                    sample_rate=self.cfg.audio.sample_rate,
                )  # fmt: skip
        self.net_g.train()
        return metrics

    def log(self, metrics: dict[str, float]) -> None:
        if self.writer is not None:
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, self.step)

    def fit(self, train_loader: DataLoader, sampler, val_loader: DataLoader | None) -> None:
        t = self.cfg.train
        self.net_g.train()
        self.net_d.train()
        start = time.time()
        while self.epoch < t.epochs:
            sampler.set_epoch(self.epoch)
            for batch in train_loader:
                metrics = self.train_step(batch)
                self.step += 1
                if self.step % t.log_interval == 0:
                    metrics["lr"] = self.opt_g.param_groups[0]["lr"]
                    metrics["steps_per_sec"] = t.log_interval / max(time.time() - start, 1e-6)
                    start = time.time()
                    self.log(metrics)
                    print(
                        f"step {self.step} " + " ".join(f"{k}={v:.3f}" for k, v in metrics.items())
                    )
                if self.step % t.eval_interval == 0:
                    self.log(self.evaluate(val_loader))
                if self.step % t.checkpoint_interval == 0:
                    print(f"saved {self.save()}")
                if t.max_steps is not None and self.step >= t.max_steps:
                    self.save()
                    return
            self.epoch += 1
            self.sched_g.step()
            self.sched_d.step()


def build_loaders(cfg: Config):
    train_ds = TextAudioDataset(cfg.data.train_manifest, cfg.data, cfg.audio)
    if len(train_ds) == 0:
        raise SystemExit(f"no usable training items (skipped: {train_ds.skipped})")
    print(f"train items: {len(train_ds)} (skipped: {train_ds.skipped})")
    sampler = BucketBatchSampler(train_ds.frame_lengths(), cfg.train.batch_size, cfg.train.seed)
    loader_kw = dict(
        collate_fn=collate, num_workers=cfg.train.num_workers, pin_memory=torch.cuda.is_available()
    )
    train_loader = DataLoader(train_ds, batch_sampler=sampler, **loader_kw)
    val_loader = None
    if cfg.data.val_manifest and Path(cfg.data.val_manifest).exists():
        val_ds = TextAudioDataset(cfg.data.val_manifest, cfg.data, cfg.audio)
        if len(val_ds):
            val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, **loader_kw)
    return train_loader, sampler, val_loader


def main(argv: list[str] | None = None) -> Trainer:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--init-from", help="Warm-start weights from a checkpoint (new run only).")
    p.add_argument("--max-steps", type=int, help="Override train.max_steps.")
    p.add_argument("--train-manifest")
    p.add_argument("--val-manifest")
    p.add_argument("--audio-root")
    p.add_argument("--no-tensorboard", action="store_true")
    args = p.parse_args(argv)

    data_over = {k: v for k, v in {
        "train_manifest": args.train_manifest,
        "val_manifest": args.val_manifest,
        "audio_root": args.audio_root,
    }.items() if v is not None}  # fmt: skip
    train_over = {"max_steps": args.max_steps} if args.max_steps is not None else {}
    cfg = load_config(args.config, {"data": data_over, "train": train_over})

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    seed_everything(cfg.train.seed)

    device = pick_device(args.device)
    trainer = Trainer(cfg, run_dir, device)
    print(f"device={device} precision={trainer.precision} "
          f"params_g={sum(p.numel() for p in trainer.net_g.parameters()) / 1e6:.1f}M")  # fmt: skip

    latest = latest_checkpoint(run_dir)
    if latest is not None:
        trainer.resume(latest)
    elif args.init_from:
        trainer.warm_start(Path(args.init_from))

    if not args.no_tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        trainer.writer = SummaryWriter(str(run_dir / "logs"))

    train_loader, sampler, val_loader = build_loaders(cfg)
    trainer.fit(train_loader, sampler, val_loader)
    if trainer.writer is not None:
        trainer.writer.close()
    return trainer


if __name__ == "__main__":
    main()
