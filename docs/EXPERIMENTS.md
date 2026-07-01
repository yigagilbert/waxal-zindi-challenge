# Experiment Runbook

All commands use `uv run`. Do not run full training until the smoke sequence passes.

## 1. Audit Data

```bash
uv run scripts/audit_data.py \
  --raw-dir "$WAXAL_RAW_DIR" \
  --output outputs/data_audit.json
```

## 2. Prepare Metadata Only

```bash
uv run scripts/prepare_dataset.py \
  --raw-dir "$WAXAL_RAW_DIR" \
  --output-dir data/processed \
  --metadata-only
```

## 3. Prepare Tiny Audio Cache

```bash
uv run scripts/prepare_dataset.py \
  --raw-dir "$WAXAL_RAW_DIR" \
  --output-dir data/processed_smoke \
  --streaming \
  --max-per-language-split 3
```

## 4. Run GPU Environment Check

```bash
uv run scripts/check_gpu_env.py \
  --raw-dir "$WAXAL_RAW_DIR" \
  --require-gpu
```

## 5. Run Whisper Turbo on Tiny Validation Cache

```bash
uv run scripts/run_whisper_inference.py \
  --model-name openai/whisper-large-v3-turbo \
  --dataset-dir data/processed_smoke \
  --split validation \
  --max-samples 3 \
  --output outputs/predictions/whisper_turbo_tiny_validation.csv
```

## 6. Run Whisper Turbo on Full Validation

```bash
uv run scripts/run_whisper_inference.py \
  --model-name openai/whisper-large-v3-turbo \
  --dataset-dir data/processed \
  --split validation \
  --output outputs/predictions/whisper_turbo_validation.csv
```

## 7. Run Sunbird Whisper on Luganda Validation

```bash
uv run scripts/run_whisper_inference.py \
  --model-name Sunbird/asr-whisper-large-v3-salt \
  --dataset-dir data/processed \
  --split validation \
  --languages lug \
  --output outputs/predictions/sunbird_whisper_lug_validation.csv
```

## 8. Evaluate Normalization Policies

```bash
uv run scripts/evaluate_predictions.py \
  --predictions outputs/predictions/whisper_turbo_validation.csv \
  --references data/processed/validation.csv \
  --normalization all \
  --output outputs/experiments/whisper_turbo_validation_all_norms.json
```

## 9. Train XLS-R 300M Smoke Run

```bash
uv run scripts/train_xlsr_ctc.py \
  --config configs/xlsr_300m.yaml \
  --dataset-dir data/processed_smoke \
  --max-train-samples 6 \
  --max-eval-samples 3 \
  --max-steps 2 \
  --output-dir checkpoints/xlsr_300m_smoke
```

## 10. Train XLS-R 300M Real Run

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

## 11. Train Whisper LoRA Smoke Run

```bash
uv run scripts/train_whisper.py \
  --config configs/whisper_medium_lora.yaml \
  --dataset-dir data/processed_smoke \
  --max-train-samples 3 \
  --max-eval-samples 3 \
  --max-steps 2 \
  --output-dir checkpoints/whisper_medium_lora_smoke
```

## 12. Train Whisper LoRA Real Run

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

## 13. Create Validation Comparison Table

Evaluate each prediction file with all normalization policies:

```bash
uv run scripts/evaluate_predictions.py \
  --predictions outputs/predictions/<PREDICTION_FILE>.csv \
  --references data/processed/validation.csv \
  --normalization all \
  --output outputs/experiments/<RUN_NAME>_all_norms.json
```

Track at minimum:

- model name
- checkpoint
- normalization policy
- overall WER/CER/combined
- per-language WER/CER/combined
- public leaderboard score only after local validation is stable

## 14. Generate First Safe Test Submission

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

