"""Audio utilities for WAXAL ASR preparation and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TARGET_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class AudioStats:
    """Basic audio validation stats."""

    duration: float
    sample_rate: int
    num_samples: int
    peak_abs: float
    rms: float
    is_silent: bool


def _np():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required for audio utilities.") from exc
    return np


def audio_array_from_example(audio: Any) -> tuple[Any, int]:
    """Extract (array, sampling_rate) from a Hugging Face Audio value."""
    if not isinstance(audio, dict):
        raise TypeError(f"Expected Hugging Face audio dict, got {type(audio)}")
    if "array" not in audio or "sampling_rate" not in audio:
        raise ValueError(f"Audio dict missing array/sampling_rate keys: {list(audio)}")
    return audio["array"], int(audio["sampling_rate"])


def to_mono(array: Any) -> Any:
    """Convert an audio array to mono if it has channels."""
    np = _np()
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[1]:
            return arr.mean(axis=0)
        return arr.mean(axis=1)
    raise ValueError(f"Unsupported audio array shape: {arr.shape}")


def duration_seconds(array: Any, sample_rate: int) -> float:
    """Return duration in seconds."""
    arr = to_mono(array)
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate: {sample_rate}")
    return float(len(arr) / sample_rate)


def peak_normalize(array: Any, target_peak: float = 0.95) -> Any:
    """Peak-normalize audio without changing silence."""
    np = _np()
    arr = to_mono(array)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak <= 0:
        return arr
    return (arr / peak * target_peak).astype(np.float32)


def trim_silence(array: Any, threshold: float = 1e-4, pad_samples: int = 1600) -> Any:
    """Lightly trim leading/trailing silence from a mono waveform."""
    np = _np()
    arr = to_mono(array)
    if arr.size == 0:
        return arr
    mask = np.abs(arr) > threshold
    if not mask.any():
        return arr
    indices = np.where(mask)[0]
    start = max(int(indices[0]) - pad_samples, 0)
    end = min(int(indices[-1]) + pad_samples + 1, len(arr))
    return arr[start:end].astype(np.float32)


def resample_if_needed(array: Any, sample_rate: int, target_rate: int = TARGET_SAMPLE_RATE) -> tuple[Any, int]:
    """Resample audio with librosa only when the input rate differs."""
    arr = to_mono(array)
    if sample_rate == target_rate:
        return arr, sample_rate
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError(
            f"Audio is {sample_rate} Hz and librosa is unavailable for resampling."
        ) from exc
    return librosa.resample(arr, orig_sr=sample_rate, target_sr=target_rate), target_rate


def audio_stats(array: Any, sample_rate: int, silence_threshold: float = 1e-5) -> AudioStats:
    """Compute validation stats for a waveform."""
    np = _np()
    arr = to_mono(array)
    if arr.size == 0:
        return AudioStats(0.0, sample_rate, 0, 0.0, 0.0, True)
    peak = float(np.max(np.abs(arr)))
    rms = float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))
    return AudioStats(
        duration=duration_seconds(arr, sample_rate),
        sample_rate=sample_rate,
        num_samples=int(arr.size),
        peak_abs=peak,
        rms=rms,
        is_silent=bool(rms < silence_threshold),
    )


def is_audio_usable(array: Any, sample_rate: int, min_duration: float = 0.05) -> tuple[bool, str]:
    """Return whether audio looks usable and a short reason."""
    stats = audio_stats(array, sample_rate)
    if stats.num_samples == 0:
        return False, "empty"
    if stats.duration < min_duration:
        return False, "too_short"
    if stats.is_silent:
        return False, "silent"
    return True, "ok"

