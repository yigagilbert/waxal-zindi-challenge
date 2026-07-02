# GPU Setup

This project uses `uv` for reproducible Python environments. Python 3.11 is pinned in `.python-version` because it is a stable choice for current PyTorch, Transformers, datasets audio decoding, and training utilities.

## Common Setup

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

Clone the repository:

```bash
git clone <YOUR_REPO_URL> waxal-asr
cd waxal-asr
```

Copy environment template:

```bash
cp .env.example .env
```

Set the raw Zindi file directory:

```bash
export WAXAL_RAW_DIR=/workspace/data/google-waxal-asr-challenge20260630-10570-elxebu
```

Install dependencies with automatic PyTorch backend detection:

```bash
UV_TORCH_BACKEND=auto uv sync --locked --extra training
```

Use the unlocked form only when intentionally updating dependencies and regenerating `uv.lock`:

```bash
UV_TORCH_BACKEND=auto uv sync --extra training
```

If automatic backend detection does not choose the expected CUDA wheel, pin the backend explicitly. Use the CUDA version that matches your driver/image:

```bash
UV_TORCH_BACKEND=cu124 uv sync --locked --extra training
```

For CPU-only local development:

```bash
UV_TORCH_BACKEND=cpu uv sync --locked
```

Verify PyTorch/CUDA:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Run the project GPU environment check:

```bash
uv run scripts/check_gpu_env.py --raw-dir "$WAXAL_RAW_DIR" --require-gpu
```

## Vast.ai

Recommended GPUs:

- RTX 4090 24GB: cheap smoke tests, Whisper medium LoRA, XLS-R 300M experiments with careful batch sizes.
- RTX 5090: viable but requires a CUDA 12.8 PyTorch build for `sm_120`; see `docs/RESTART_RUNBOOK.md` and use `uv run --no-sync` after manual torch repair.
- A100 40GB/80GB: serious WAXAL training, Whisper large LoRA, XLS-R 1B, pseudo-label sweeps.
- H100: final fast sweeps only if budget allows.

Recommended image:

- Prefer a recent PyTorch CUDA image with working NVIDIA drivers, CUDA runtime, Python 3.11, git, ffmpeg, and enough disk.
- Ubuntu plus CUDA drivers also works, but you must verify `nvidia-smi`, `ffmpeg`, and Python yourself.

Storage:

- Minimum 150GB disk.
- Prefer 250GB+ because Hugging Face audio cache, model downloads, checkpoints, and logs grow quickly.
- Put Hugging Face caches and checkpoints on persistent instance storage.

Interruptible instance rule:

- Use frequent checkpointing. Current configs save every 200-250 steps for the first serious runs.
- Resume with `--resume-from-checkpoint checkpoints/<run>/checkpoint-XXXX`.
- Sync important checkpoints off-machine before terminating the instance.

Vast.ai setup:

```bash
apt-get update
apt-get install -y git curl ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
git clone <YOUR_REPO_URL> waxal-asr
cd waxal-asr
export WAXAL_RAW_DIR=/workspace/data/google-waxal-asr-challenge20260630-10570-elxebu
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
UV_TORCH_BACKEND=auto uv sync --locked --extra training
uv run scripts/check_gpu_env.py --raw-dir "$WAXAL_RAW_DIR" --require-gpu
```

Tiny data cache:

```bash
uv run scripts/prepare_dataset.py \
  --raw-dir "$WAXAL_RAW_DIR" \
  --output-dir data/processed_smoke \
  --streaming \
  --max-per-language-split 3
```

Tiny training smoke test:

```bash
uv run scripts/train_xlsr_ctc.py \
  --config configs/xlsr_300m.yaml \
  --dataset-dir data/processed_smoke \
  --max-train-samples 6 \
  --max-eval-samples 3 \
  --max-steps 2 \
  --output-dir checkpoints/xlsr_300m_smoke
```

## Azure GPU VM

Recommended VM:

- A100 80GB if available: best balance for serious training.
- H100 only if budget allows and quota is approved.
- Avoid T4 as the main training GPU. It is fine for smoke tests, not for serious Whisper large or XLS-R 1B work.

Quota:

- Request GPU quota before the final competition window. A100/H100 quotas can take time.

Disk:

- Minimum 256GB SSD.
- Prefer 512GB for full cache, multiple model branches, pseudo-labels, and checkpoint retention.

Azure setup:

```bash
sudo apt-get update
sudo apt-get install -y git curl ffmpeg
nvidia-smi
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
git clone <YOUR_REPO_URL> waxal-asr
cd waxal-asr
export WAXAL_RAW_DIR=/mnt/data/google-waxal-asr-challenge20260630-10570-elxebu
export HF_HOME=/mnt/data/.cache/huggingface
export HF_DATASETS_CACHE=/mnt/data/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/mnt/data/.cache/huggingface/transformers
UV_TORCH_BACKEND=auto uv sync --locked --extra training
uv run scripts/check_gpu_env.py --raw-dir "$WAXAL_RAW_DIR" --require-gpu --min-free-gb 150
```

If PyTorch installs CPU-only by mistake:

```bash
UV_TORCH_BACKEND=cu124 uv sync --locked --extra training --reinstall-package torch --reinstall-package torchaudio
```

## Local CPU/macOS Development

Local development is for audit, metadata prep, scoring, submission alignment, config checks, and tiny CPU smoke tests.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
UV_TORCH_BACKEND=cpu uv sync --locked --extra dev
uv run scripts/check_gpu_env.py
uv run scripts/audit_data.py
uv run scripts/prepare_dataset.py --metadata-only
```

On Apple Silicon, use CPU/MPS only for lightweight checks. Do not plan serious final training there.
