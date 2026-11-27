"""Export a trained Kural-VITS checkpoint to the ONNX bundle the pip package loads.

    python -m training.export_onnx runs/sahithi_v1 exports/sahithi_v1
    python -m training.export_onnx runs/sahithi_v1/ckpt_200000.pt exports/sahithi_v1

Produces ``model.onnx`` and ``config.json``. The graph contains only the inference path
(text encoder, duration predictor, flow, decoder). It is checked against PyTorch with
deterministic sampling (noise_scale = 0) before the script returns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from kural_tts.text.g2p import text_to_ids
from kural_tts.text.symbols import LANGS, SYMBOLS, SYMBOLS_VERSION
from training.checkpoint import latest_checkpoint, load_checkpoint
from training.models.vits.synthesizer import ModelConfig, SynthesizerTrn

ONNX_FILE = "model.onnx"
CONFIG_FILE = "config.json"
OPSET = 17


class InferenceWrapper(torch.nn.Module):
    """ONNX signature: (input, input_lengths, lang, scales[noise, length]) -> audio [B, samples]."""

    def __init__(self, model: SynthesizerTrn):
        super().__init__()
        self.model = model

    def forward(self, input, input_lengths, lang, scales):
        audio, _, _ = self.model.infer(
            input, input_lengths, lang, noise_scale=scales[0], length_scale=scales[1]
        )
        return audio.squeeze(1)


def load_generator(ckpt_path: Path) -> tuple[SynthesizerTrn, dict]:
    ckpt = load_checkpoint(ckpt_path)
    cfg = ckpt["config"]
    if cfg.get("symbols_version") != SYMBOLS_VERSION:
        raise SystemExit(
            f"checkpoint symbols_version {cfg.get('symbols_version')} != package {SYMBOLS_VERSION}"
        )
    model = SynthesizerTrn(ModelConfig(**cfg["model"]))
    model.load_state_dict(ckpt["net_g"])
    model.eval()
    model.remove_weight_norm()
    return model, {**cfg, "step": ckpt.get("step")}


def export(
    ckpt_path: Path,
    out_dir: Path,
    voice: str = "sahithi",
    noise_scale: float = 0.667,
    length_scale: float = 1.0,
) -> Path:
    model, cfg = load_generator(ckpt_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    # eval() on the wrapper matters: torch.onnx.export restores the module's previous mode
    # afterwards, and a wrapper left in train mode would switch dropout back on in `model`.
    wrapper = InferenceWrapper(model).eval()

    ids, langs = text_to_ids("வணக்கம், hello Sahithi.")
    x = torch.tensor([ids], dtype=torch.long)
    lang = torch.tensor([langs], dtype=torch.long)
    x_len = torch.tensor([len(ids)], dtype=torch.long)
    scales = torch.tensor([noise_scale, length_scale], dtype=torch.float32)

    onnx_path = out_dir / ONNX_FILE
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (x, x_len, lang, scales),
            str(onnx_path),
            input_names=["input", "input_lengths", "lang", "scales"],
            output_names=["audio"],
            dynamic_axes={
                "input": {0: "batch", 1: "phonemes"},
                "input_lengths": {0: "batch"},
                "lang": {0: "batch", 1: "phonemes"},
                "audio": {0: "batch", 1: "samples"},
            },
            opset_version=OPSET,
            dynamo=False,
        )

    bundle_cfg = {
        "format": "kural-tts-onnx/1",
        "voice": voice,
        "sample_rate": cfg["audio"]["sample_rate"],
        "hop_length": cfg["audio"]["hop_length"],
        "symbols_version": SYMBOLS_VERSION,
        "n_symbols": len(SYMBOLS),
        "langs": LANGS,
        "inference": {"noise_scale": noise_scale, "length_scale": length_scale},
        "training_step": cfg.get("step"),
    }
    (out_dir / CONFIG_FILE).write_text(json.dumps(bundle_cfg, indent=2), encoding="utf-8")
    verify(model, onnx_path)
    return onnx_path


def verify(
    model: SynthesizerTrn, onnx_path: Path, atol: float = 1e-3, max_frame_diff: int = 1
) -> float:
    """Compare ONNX Runtime and PyTorch outputs on texts of different lengths (noise_scale=0)."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    model.eval()
    worst = 0.0
    for text in ["சரி.", "Hello, this is a longer English sentence for the check.",
                 "நாளைக்கு meeting பத்து மணிக்கு இருக்கு."]:  # fmt: skip
        ids, langs = text_to_ids(text)
        x = np.array([ids], dtype=np.int64)
        lang = np.array([langs], dtype=np.int64)
        x_len = np.array([len(ids)], dtype=np.int64)
        scales = np.array([0.0, 1.0], dtype=np.float32)
        (onnx_audio,) = session.run(
            None, {"input": x, "input_lengths": x_len, "lang": lang, "scales": scales}
        )
        with torch.no_grad():
            ref, _, _ = model.infer(
                torch.from_numpy(x), torch.from_numpy(x_len), torch.from_numpy(lang),
                noise_scale=0.0, length_scale=1.0,
            )  # fmt: skip
        ref = ref.squeeze(1).numpy()
        if onnx_audio.shape != ref.shape:
            # ceil() of a duration that sits on an integer boundary can round differently in
            # ONNX Runtime and PyTorch float paths. A frame or two is harmless. More is a bug.
            frames = abs(onnx_audio.shape[-1] - ref.shape[-1]) // model.hop_length
            if frames > max_frame_diff:
                raise RuntimeError(
                    f"length mismatch for {text!r}: {onnx_audio.shape} vs {ref.shape}"
                )
            print(f"note: {text!r} differs by {frames} frame(s) from duration rounding")
            continue
        worst = max(worst, float(np.abs(onnx_audio - ref).max()))
    if worst > atol:
        raise RuntimeError(f"ONNX/PyTorch mismatch: max abs diff {worst:.2e} > {atol}")
    print(f"verified {onnx_path} (max abs diff {worst:.2e})")
    return worst


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("checkpoint", help="ckpt_*.pt file or a run directory (newest checkpoint)")
    p.add_argument("out_dir")
    p.add_argument("--voice", default="sahithi")
    p.add_argument("--noise-scale", type=float, default=0.667)
    p.add_argument("--length-scale", type=float, default=1.0)
    args = p.parse_args(argv)
    ckpt = Path(args.checkpoint)
    if ckpt.is_dir():
        found = latest_checkpoint(ckpt)
        if found is None:
            raise SystemExit(f"no checkpoints in {ckpt}")
        ckpt = found
    return export(ckpt, Path(args.out_dir), args.voice, args.noise_scale, args.length_scale)


if __name__ == "__main__":
    main()
