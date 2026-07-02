# Restart Runbook

This runbook is for recreating the WAXAL GPU environment after an interrupted Vast.ai run and getting useful validation results before spending time on full training.

## What Happened

The previous Vast.ai instance was an RTX 5090. The locked environment installed `torch==2.6.0+cu124`, which does not support the RTX 5090 `sm_120` compute capability. CUDA import could work, but real kernels failed until PyTorch was replaced with a CUDA 12.8 build.

The instance was lost before full data preparation completed. Before failure, the environment was repaired, CUDA matmul passed, tiny WAXAL inference ran, and tiny predictions were produced.

## Completed Before Failure

- Repaired PyTorch inside `.venv` to `torch==2.11.0+cu128`.
- Verified CUDA on `NVIDIA GeForce RTX 5090`, capability `(12, 0)`.
- Fixed Whisper inference padding to 30-second max-length features.
- Fixed Whisper input dtype casting to match float16 CUDA model weights.
- Ran tiny Whisper validation inference successfully.
- Confirmed tiny evaluation must use `data/processed_smoke/validation.csv`, not the full validation CSV.

## Files Changed

- `scripts/run_whisper_inference.py`: Whisper max-length padding/truncation and model dtype casting.
- `scripts/prepare_dataset.py`: `--splits` support and safe merging with existing cached splits.
- `scripts/evaluate_predictions.py`: optional `--languages` filtering for per-language evaluation.
- `Makefile`: restart, staged preparation, Sunbird-first validation, and artifact backup targets.
- `docs/RESTART_RUNBOOK.md` and `docs/CHANGES_LOG.md`: restart knowledge captured in-repo.

## New GPU Instance Setup

```bash
apt-get update
apt-get install -y git curl ffmpeg rsync
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
git clone <YOUR_REPO_URL> WAXAL-ZINDI-CHALLENGE
cd WAXAL-ZINDI-CHALLENGE
export WAXAL_RAW_DIR=/workspace/data/google-waxal-asr-challenge20260630-10570-elxebu
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
UV_TORCH_BACKEND=auto uv sync --locked --extra training
```

Run the environment check:

```bash
uv run scripts/check_gpu_env.py --raw-dir "$WAXAL_RAW_DIR" --require-gpu --min-free-gb 100
```

## RTX 5090 Torch Repair

Use this only on RTX 5090 or another GPU where the locked PyTorch build fails real CUDA kernels.

```bash
uv pip uninstall --python .venv/bin/python torch torchaudio torchvision -y
uv pip install --python .venv/bin/python --upgrade torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
```

After this repair, do not let `uv run` resync the environment back to the locked torch build. Use:

```bash
export WAXAL_NO_SYNC=1
uv run --no-sync python -c "import torch; x=torch.randn(1024,1024,device='cuda'); y=x@x; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), y.mean().item())"
```

All Makefile commands respect `WAXAL_NO_SYNC=1`.

## Stage 1: Cheap Checks

```bash
WAXAL_NO_SYNC=1 make restart-check
WAXAL_NO_SYNC=1 make audit
WAXAL_NO_SYNC=1 make prepare-metadata
WAXAL_NO_SYNC=1 make prepare-tiny
```

Tiny inference:

```bash
WAXAL_NO_SYNC=1 make sunbird-lug-tiny
WAXAL_NO_SYNC=1 make eval-sunbird-lug-tiny
WAXAL_NO_SYNC=1 make whisper-tiny
WAXAL_NO_SYNC=1 make eval-tiny
```

The tiny evaluation reference is `data/processed_smoke/validation.csv`.

## Stage 2: Validation First

Prepare only validation audio first:

```bash
WAXAL_NO_SYNC=1 make prepare-validation
```

Run first-class validation baselines:

```bash
WAXAL_NO_SYNC=1 make sunbird-lug-validation
WAXAL_NO_SYNC=1 make eval-sunbird-lug
WAXAL_NO_SYNC=1 make whisper-turbo-validation
WAXAL_NO_SYNC=1 make eval-whisper-turbo
```

Optional stronger baseline:

```bash
WAXAL_NO_SYNC=1 make whisper-large-validation
```

Compare per-language WER, CER, and combined score before training.

## Stage 3: Train Cache and Smoke Training

Only after validation baselines work:

```bash
WAXAL_NO_SYNC=1 make prepare-train
WAXAL_NO_SYNC=1 make xlsr-smoke
WAXAL_NO_SYNC=1 make whisper-smoke
```

Start real XLS-R 300M training only after the smoke runs produce metrics and checkpoints.

## Stage 4: Test Cache and Submission

Prepare test audio only after validation is stable:

```bash
WAXAL_NO_SYNC=1 make prepare-test
```

Then run test inference from the best validation model and create a submission with `scripts/make_submission.py`.

## Artifact Backup

Back up before stopping or destroying an instance:

```bash
WAXAL_NO_SYNC=1 make backup-artifacts BACKUP_DIR=/workspace/persistent/waxal_backup
```

Copy to a local machine:

```bash
rsync -avP /workspace/persistent/waxal_backup/ user@LOCAL_HOST:/path/to/waxal_backup/
```

Or copy directly from the instance:

```bash
rsync -avP outputs/predictions outputs/experiments outputs/submissions checkpoints \
  data/processed/prepare_report.json scripts configs docs \
  user@LOCAL_HOST:/path/to/waxal_backup/
```

Critical artifacts:

- `outputs/predictions/`
- `outputs/experiments/`
- `outputs/submissions/`
- `checkpoints/`
- `data/processed/prepare_report.json`
- modified `scripts/`, `configs/`, `docs/`, `README.md`, `pyproject.toml`, `uv.lock`, `Makefile`
