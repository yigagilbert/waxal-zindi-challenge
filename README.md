# WAXAL ASR Competition Pipeline

This repository is a local-first, cloud-ready training and inference setup for the Google WAXAL ASR Challenge on Zindi. It supports robust Zindi CSV loading, Hugging Face WAXAL audio matching, local WER/CER scoring, Whisper inference and fine-tuning, XLS-R CTC fine-tuning, submission generation, and local experiment logs.

The target languages are Lingala `lin`, Shona `sna`, and Luganda `lug`.

## Setup

Python 3.11 is pinned in `.python-version`. Use `uv` as the package manager:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

Install dependencies:

```bash
UV_TORCH_BACKEND=auto uv sync --locked --extra training
```

For CPU-only local development:

```bash
UV_TORCH_BACKEND=cpu uv sync --locked --extra dev
```

The scripts default to the downloaded Zindi files at:

```text
/Users/sunbird/Downloads/google-waxal-asr-challenge20260630-10570-elxebu
```

For portability, set:

```bash
export WAXAL_RAW_DIR=/path/to/google-waxal-asr-challenge20260630-10570-elxebu
```

Copy `.env.example` to `.env` on cloud machines and fill only local paths or optional tokens:

```bash
cp .env.example .env
```

Do not commit `.env`, Hugging Face tokens, downloaded audio caches, checkpoints, or model outputs.

## GPU Setup

See `docs/GPU_SETUP.md` for Vast.ai, Azure GPU VM, CUDA-enabled PyTorch, and local CPU/macOS setup.

Quick CUDA verification:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Project environment check:

```bash
uv run scripts/check_gpu_env.py --raw-dir "$WAXAL_RAW_DIR"
```

Require a GPU on cloud:

```bash
uv run scripts/check_gpu_env.py --raw-dir "$WAXAL_RAW_DIR" --require-gpu --min-free-gb 100
```

## Guardrails

Do not use public Hugging Face Phase 1 test labels. `prepare_dataset.py` may load Hugging Face test audio by ID, but it deliberately drops any test transcription field and uses only official Zindi IDs.

Do not rely on CC-BY-NC models such as MMS for a final prize solution until Zindi confirms the license is acceptable. Do not hardcode Hugging Face tokens. Do not use the public leaderboard as your main validation signal.

## Data Audit

```bash
uv run scripts/audit_data.py \
  --raw-dir "$WAXAL_RAW_DIR" \
  --output outputs/data_audit.json
```

This verifies row counts, required columns, missing values, duplicate IDs, train/test overlap, sample-submission alignment, language/split distributions, length statistics, character vocabulary, and suspicious text examples.

## Prepare Data

Metadata-only smoke preparation:

```bash
uv run scripts/prepare_dataset.py \
  --raw-dir "$WAXAL_RAW_DIR" \
  --output-dir data/processed \
  --metadata-only
```

Small audio smoke cache:

```bash
uv run scripts/prepare_dataset.py \
  --raw-dir "$WAXAL_RAW_DIR" \
  --output-dir data/processed_smoke \
  --streaming \
  --max-per-language-split 3
```

Full local Hugging Face cache:

```bash
uv run scripts/prepare_dataset.py \
  --raw-dir "$WAXAL_RAW_DIR" \
  --output-dir data/processed
```

Outputs include:

```text
data/processed/train.csv
data/processed/validation.csv
data/processed/test.csv
data/processed/hf_dataset/
data/processed/prepare_report.json
```

## Whisper Baseline Inference

Run validation inference after preparing `hf_dataset`:

```bash
uv run scripts/run_whisper_inference.py \
  --model-name openai/whisper-large-v3-turbo \
  --dataset-dir data/processed \
  --split validation \
  --output outputs/predictions/whisper_large_v3_turbo_validation.csv
```

General Whisper large-v3:

```bash
uv run scripts/run_whisper_inference.py \
  --model-name openai/whisper-large-v3 \
  --dataset-dir data/processed \
  --split validation \
  --output outputs/predictions/whisper_large_v3_validation.csv
```

Luganda Sunbird Whisper:

```bash
uv run scripts/run_whisper_inference.py \
  --model-name Sunbird/asr-whisper-large-v3-salt \
  --dataset-dir data/processed \
  --split validation \
  --languages lug \
  --output outputs/predictions/sunbird_whisper_lug_validation.csv
```

## Evaluate Predictions

```bash
uv run scripts/evaluate_predictions.py \
  --predictions outputs/predictions/whisper_large_v3_turbo_validation.csv \
  --references data/processed/validation.csv \
  --normalization starter_lower \
  --output outputs/experiments/whisper_large_v3_turbo_validation_metrics.json
```

Evaluate all normalization policies:

```bash
uv run scripts/evaluate_predictions.py \
  --predictions outputs/predictions/whisper_large_v3_turbo_validation.csv \
  --references data/processed/validation.csv \
  --normalization all
```

## Whisper LoRA Training

Smoke run only:

```bash
uv run scripts/train_whisper.py \
  --config configs/whisper_large_v3_lora.yaml \
  --dataset-dir data/processed_smoke \
  --max-train-samples 4 \
  --max-eval-samples 4 \
  --max-steps 2 \
  --output-dir checkpoints/whisper_large_v3_lora_smoke
```

Luganda-specialized Sunbird smoke run:

```bash
uv run scripts/train_whisper.py \
  --config configs/sunbird_whisper_lug_lora.yaml \
  --dataset-dir data/processed_smoke \
  --max-train-samples 4 \
  --max-eval-samples 4 \
  --max-steps 2 \
  --output-dir checkpoints/sunbird_whisper_lug_lora_smoke
```

## XLS-R CTC Training

Smoke run only:

```bash
uv run scripts/train_xlsr_ctc.py \
  --config configs/xlsr_300m.yaml \
  --dataset-dir data/processed_smoke \
  --max-train-samples 8 \
  --max-eval-samples 8 \
  --max-steps 2 \
  --output-dir checkpoints/xlsr_300m_smoke
```

The XLS-R branch builds a character vocabulary from training transcripts and writes it under the configured checkpoint directory.

## Make a Submission

First produce test predictions:

```bash
uv run scripts/run_whisper_inference.py \
  --model-name openai/whisper-large-v3-turbo \
  --dataset-dir data/processed \
  --split test \
  --output outputs/predictions/whisper_large_v3_turbo_test.csv
```

Then align exactly to `SampleSubmission.csv`:

```bash
uv run scripts/make_submission.py \
  --predictions outputs/predictions/whisper_large_v3_turbo_test.csv \
  --raw-dir "$WAXAL_RAW_DIR" \
  --model-name whisper_large_v3_turbo
```

The output is written to `outputs/submissions/submission_<model>_<timestamp>.csv`.

## Outputs

```text
outputs/predictions/    validation and test prediction CSVs
outputs/submissions/    Zindi-ready submission CSVs
outputs/experiments/    local JSON metrics and experiment metadata
outputs/logs/           trainer and cloud run logs
data/processed/         prepared metadata and optional Hugging Face dataset cache
checkpoints/            model checkpoints, tokenizer files, processors, adapters
```

Large generated artifacts are ignored by git. Keep important cloud checkpoints on persistent disk and sync the best runs elsewhere before deleting instances.

## Recommended First Experiments

Follow `docs/EXPERIMENTS.md`. The short version:

1. Audit data.
2. Prepare metadata and a tiny audio smoke cache.
3. Run the GPU environment check.
4. Run `openai/whisper-large-v3-turbo` on tiny validation, then full validation.
5. Run `Sunbird/asr-whisper-large-v3-salt` on Luganda validation.
6. Compare per-language WER/CER under `raw`, `starter_lower`, and `language_safe`.
7. Run XLS-R 300M smoke, then a real run if validation is sane.
8. Run Whisper LoRA smoke, then a real run.
9. Generate a test submission only after validation behavior is stable.

## Makefile Shortcuts

```bash
make audit
make prepare-metadata
make check-gpu
make prepare-tiny
make whisper-tiny
make eval
make xlsr-smoke
make whisper-smoke
```

All shortcuts call `uv run`.

## Project Layout

```text
configs/                  experiment configs
data/processed/            local prepared metadata and optional HF cache
docs/                      GPU setup and experiment runbook
notebooks/exploration.ipynb scratch exploration notebook
outputs/                   predictions, submissions, experiment logs
scripts/                   command-line entry points
src/waxal/                 shared WAXAL library code
```
