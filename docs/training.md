# Runbook: from recordings to a released voice

```
1 make_script ─> 2 record ─> 3 build dataset ─> 4 train (Colab) ─> 5 export ONNX ─> 6 evaluate ─> 7 publish
```

| Step | Command | Output |
|---|---|---|
| 1. Script | `python recording/make_script.py --pool pools/*.txt --hours 1 -o recording/pilot_prompts.tsv` | prompts with a spoken-form `say` column |
| 2. Record | `python recording/recorder.py --prompts ... --out recordings/session_01` | 48 kHz/24-bit WAV per prompt |
| 3. Dataset | `python -m training.data_pipeline.build --raw recordings --prompts ... --out data/sahithi_v1` | wavs, manifests, `qc_report.md` |
| 4. Train | `notebooks/02_train_kural_vits.ipynb` (or `python -m training.train ...`) | `runs/<name>/ckpt_*.pt`, TensorBoard |
| 5. Export | `python -m training.export_onnx runs/<name> exports/<name>` | `model.onnx` + `config.json` (verified against PyTorch) |
| 6. Evaluate | `python scripts/eval/render_eval_set.py --model exports/<name> --name kural_<name>`, then `scripts/baseline/score.py` | CER/WER/UTMOS beside the baselines, plus a listening test |
| 7. Publish | `python scripts/release/publish_voice.py exports/<name> --repo <org>/kural-tts-sahithi --upload` | Hugging Face release, revision pinned in `hub.py` |

## Pilot first

Record about **1 hour** (`--hours 1`) and run steps 3–6 before booking the remaining sessions. On 1 h
the voice will be rough, but it proves that:
- the frontend, recordings and alignment agree (the alignment plot in TensorBoard is diagonal, and
  eval audio becomes intelligible within roughly 20–40k steps);
- the QC thresholds suit your studio (check `qc_report.md`);
- Tamil, English and code-switched eval sentences all sound like the same person.

## Training on Colab

- **L4 / A100** (Colab Pro): `BATCH_SIZE=None` (16) or 32 on an A100. On a **T4**, use 8.
- Checkpoints are written to Drive every 2,000 steps (`checkpoint_interval`). After a disconnect,
  re-run the Train cell and it continues from the latest checkpoint, with optimizer, scheduler and
  step all restored.
- What to watch in TensorBoard:
  - `loss/mel` falls steadily. Around 15–20 is usable and lower is better.
  - `loss/dur` stabilizes early.
  - `loss/kl` should stay small.
  - Above all, **listen to `eval/text_*`**. The four eval sentences cover Tamil, English,
    code-switched and Tanglish input.
- Rough budget: 150–300k steps for a 15 h dataset. Export and listen every ~50k steps. Stop when
  the eval audio stops improving.

### Warm start

```bash
python -m training.train --config training/configs/base.yaml --run-dir runs/v2 \
    --init-from runs/v1/ckpt_300000.pt
```

This copies every tensor whose name and shape match. Use it to continue from the pilot model on
the full dataset, or after adding sessions. Adding new symbols to the inventory changes the
embedding shapes. The embeddings are then skipped (re-initialized) and everything else carries over.

## Local CPU smoke run (no GPU)

```bash
uv sync --group dev --group data --group train
uv run pytest tests/training     # model, MAS, training+resume, ONNX export, data pipeline
```

## Configs

| File | Use |
|---|---|
| `training/configs/base.yaml` | Full model, about 36M generator params, 22.05 kHz |
| `training/configs/tiny.yaml` | CPU tests and debugging only |

Key knobs are listed below. Change them one at a time and log each run in `docs/experiments.md`.

| Knob | Default | Notes |
|---|---|---|
| `train.segment_size` | 8192 | Decoder training window. Larger costs more memory |
| `train.c_mel` | 45 | Weight of the mel reconstruction loss |
| `train.learning_rate` | 2e-4 | AdamW, decayed 0.999875 per epoch |
| `model.n_layers` | 6 | Text encoder depth |
| inference `noise_scale` | 0.667 | Lower is flatter but more stable. Set at export (`--noise-scale`) |
