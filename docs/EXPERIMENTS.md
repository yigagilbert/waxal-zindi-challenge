# Experiment Runbook

All commands use `uv run` through the Makefile. On RTX 5090 after manual cu128 torch repair, prefix commands with `WAXAL_NO_SYNC=1`.

## 1. GPU Environment Check

```bash
make restart-check
```

Use the direct command when debugging:

```bash
uv run scripts/check_gpu_env.py \
  --raw-dir "$WAXAL_RAW_DIR" \
  --require-gpu
```

## 2. Audit and Metadata

```bash
make audit
make prepare-metadata
```

## 3. Tiny Smoke Cache

```bash
make prepare-tiny
```

Run tiny Sunbird Luganda first:

```bash
make sunbird-lug-tiny
make eval-sunbird-lug-tiny
```

Run tiny general Whisper:

```bash
make whisper-tiny
make eval-tiny
```

Tiny predictions must be evaluated against `data/processed_smoke/validation.csv`.

## 4. Prepare Validation First

Do not prepare full train/test before baseline validation. Prepare only validation audio:

```bash
make prepare-validation
```

This writes or updates `data/processed/hf_dataset` while preserving any existing cached splits.

## 5. Sunbird Luganda Expert Baseline

```bash
make sunbird-lug-validation
make eval-sunbird-lug
```

This is first-class because Luganda is one of the WAXAL languages and `Sunbird/asr-whisper-large-v3-salt` is especially relevant for Luganda.

## 6. General Whisper Turbo Baseline

```bash
make whisper-turbo-validation
make eval-whisper-turbo
```

Use this for all-language comparison across Lingala, Shona, and Luganda.

## 7. Optional Stronger General Whisper

```bash
make whisper-large-validation
uv run scripts/evaluate_predictions.py \
  --predictions outputs/predictions/whisper_large_v3_validation.csv \
  --references data/processed/validation.csv \
  --normalization all \
  --output outputs/experiments/whisper_large_v3_validation_all_norms.json
```

Run this only if turbo results justify the extra inference time.

## 8. Compare Normalization Policies

For every validation prediction file, save all normalization metrics:

```bash
uv run scripts/evaluate_predictions.py \
  --predictions outputs/predictions/<RUN_NAME>_validation.csv \
  --references data/processed/validation.csv \
  --normalization all \
  --output outputs/experiments/<RUN_NAME>_all_norms.json
```

Track overall weighted score, macro-by-language score, and each language separately.

## 9. Prepare Train Cache

Only after validation baselines are working:

```bash
make prepare-train
```

## 10. Train XLS-R 300M Smoke Run

```bash
make xlsr-smoke
```

Proceed to a real XLS-R 300M run only if the smoke run creates a checkpoint and reports metrics.

## 11. Train XLS-R 300M Real Run

```bash
uv run scripts/train_xlsr_ctc.py \
  --config configs/xlsr_300m.yaml \
  --dataset-dir data/processed \
  --output-dir checkpoints/xlsr_300m_ctc_all
```

Resume:

```bash
uv run scripts/train_xlsr_ctc.py \
  --config configs/xlsr_300m.yaml \
  --dataset-dir data/processed \
  --resume-from-checkpoint checkpoints/xlsr_300m_ctc_all/checkpoint-XXXX
```

## 12. Train Whisper LoRA Smoke Run

```bash
make whisper-smoke
```

## 13. Train Whisper LoRA Real Runs

All-language OpenAI Whisper:

```bash
uv run scripts/train_whisper.py \
  --config configs/whisper_large_v3_lora.yaml \
  --dataset-dir data/processed \
  --output-dir checkpoints/whisper_large_v3_lora_all
```

Luganda-specific Sunbird branch:

```bash
uv run scripts/train_whisper.py \
  --config configs/sunbird_whisper_lug_lora.yaml \
  --dataset-dir data/processed \
  --output-dir checkpoints/sunbird_whisper_lug_lora
```

## 14. Prepare Test Cache and Submit

Prepare test only after validation is stable:

```bash
make prepare-test
```

Run test inference:

```bash
uv run scripts/run_whisper_inference.py \
  --model-name <MODEL_OR_CHECKPOINT> \
  --dataset-dir data/processed \
  --split test \
  --output outputs/predictions/<RUN_NAME>_test.csv
```

Create submission:

```bash
uv run scripts/make_submission.py \
  --predictions outputs/predictions/<RUN_NAME>_test.csv \
  --raw-dir "$WAXAL_RAW_DIR" \
  --model-name <RUN_NAME>
```

Do not submit until validation reports are saved under `outputs/experiments/`.
