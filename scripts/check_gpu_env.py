#!/usr/bin/env python3
"""Check GPU/cloud readiness for WAXAL ASR training."""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.data import default_raw_dir  # noqa: E402


def module_version(name: str) -> tuple[bool, str]:
    """Return whether a module imports and its version if available."""
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        return False, f"MISSING ({type(exc).__name__}: {exc})"
    version = getattr(mod, "__version__", "unknown")
    return True, str(version)


def run_command(args: list[str]) -> tuple[bool, str]:
    """Run a diagnostic command and return status plus compact output."""
    try:
        result = subprocess.run(
            args,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    output = result.stdout.strip()
    return result.returncode == 0, output[:2000]


def disk_free(path: Path) -> str:
    """Return free/total disk space string."""
    usage = shutil.disk_usage(path)
    gb = 1024**3
    return f"{usage.free / gb:.1f} GB free / {usage.total / gb:.1f} GB total"


def ram_available() -> str:
    """Return RAM availability string."""
    try:
        import psutil
    except Exception:
        return "unknown (psutil unavailable)"
    gb = 1024**3
    mem = psutil.virtual_memory()
    return f"{mem.available / gb:.1f} GB available / {mem.total / gb:.1f} GB total"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=default_raw_dir())
    parser.add_argument("--require-gpu", action="store_true", help="Fail if CUDA GPU is unavailable.")
    parser.add_argument("--min-free-gb", type=float, default=50.0)
    args = parser.parse_args()

    failures: list[str] = []
    warnings: list[str] = []

    print("== System ==")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"Working dir: {Path.cwd()}")
    print(f"Disk: {disk_free(Path.cwd())}")
    print(f"RAM: {ram_available()}")

    free_gb = shutil.disk_usage(Path.cwd()).free / (1024**3)
    if free_gb < args.min_free_gb:
        failures.append(f"Only {free_gb:.1f} GB free; recommended at least {args.min_free_gb:.1f} GB.")

    print("\n== Python Packages ==")
    for module in ["torch", "torchaudio", "transformers", "datasets", "accelerate"]:
        ok, version = module_version(module)
        print(f"{module}: {version}")
        if not ok:
            failures.append(f"Missing required package: {module}")

    torch_ok, _ = module_version("torch")
    cuda_available = False
    if torch_ok:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        print("\n== PyTorch CUDA ==")
        print(f"torch version: {torch.__version__}")
        print(f"cuda available: {cuda_available}")
        print(f"torch cuda version: {torch.version.cuda}")
        if cuda_available:
            print(f"gpu count: {torch.cuda.device_count()}")
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                print(
                    f"gpu[{idx}]: {torch.cuda.get_device_name(idx)} "
                    f"({props.total_memory / (1024**3):.1f} GB)"
                )
        elif args.require_gpu:
            failures.append("CUDA GPU is required but torch.cuda.is_available() is False.")

    print("\n== NVIDIA ==")
    nvidia_smi = shutil.which("nvidia-smi")
    print(f"nvidia-smi: {nvidia_smi or 'MISSING'}")
    if nvidia_smi:
        ok, output = run_command([nvidia_smi])
        print(output)
        if not ok and args.require_gpu:
            failures.append("nvidia-smi exists but failed.")
    elif args.require_gpu:
        failures.append("nvidia-smi is required for GPU cloud checks.")

    print("\n== FFmpeg ==")
    ffmpeg = shutil.which("ffmpeg")
    print(f"ffmpeg: {ffmpeg or 'MISSING'}")
    if ffmpeg:
        ok, output = run_command([ffmpeg, "-version"])
        print(output.splitlines()[0] if output else "ffmpeg found")
    else:
        failures.append("ffmpeg is missing; datasets[audio] decoding commonly needs it.")

    print("\n== Hugging Face ==")
    token_present = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    print(f"HF token available in environment: {token_present}")
    if not token_present:
        warnings.append("HF_TOKEN/HUGGING_FACE_HUB_TOKEN not set. Public data may work, gated models will not.")

    print("\n== Zindi Data Files ==")
    expected = ["Train.csv", "Test.csv", "SampleSubmission.csv"]
    print(f"raw dir: {args.raw_dir}")
    for filename in expected:
        path = args.raw_dir / filename
        exists = path.exists()
        print(f"{filename}: {'OK' if exists else 'MISSING'}")
        if not exists:
            failures.append(f"Missing expected Zindi file: {path}")

    if warnings:
        print("\n== Warnings ==")
        for warning in warnings:
            print(f"WARNING: {warning}")

    if failures:
        print("\n== Failures ==")
        for failure in failures:
            print(f"ERROR: {failure}")
        raise SystemExit(1)

    print("\nEnvironment check passed.")


if __name__ == "__main__":
    main()

