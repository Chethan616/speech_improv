"""Minimal PCM WAV I/O used by the Review 1 experiments.

The project intentionally keeps the audio boundary small and explicit so the
streaming algorithm can be tested without a heavyweight audio framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import wave

import numpy as np


@dataclass(frozen=True)
class AudioData:
    samples: np.ndarray
    sample_rate: int


def _decode_pcm(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = (
            packed[:, 0].astype(np.int32)
            | (packed[:, 1].astype(np.int32) << 8)
            | (packed[:, 2].astype(np.int32) << 16)
        )
        negative = (values & 0x800000) != 0
        values[negative] -= 1 << 24
        return values.astype(np.float32) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    raise ValueError(f"Unsupported PCM sample width: {sample_width} bytes")


def read_wav(path: str | Path, mono: bool = True) -> AudioData:
    """Read an uncompressed PCM WAV file as float32 samples in [-1, 1]."""

    path = Path(path)
    with wave.open(str(path), "rb") as handle:
        if handle.getcomptype() != "NONE":
            raise ValueError("Only uncompressed PCM WAV files are supported")
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())

    decoded = _decode_pcm(frames, sample_width)
    if channels > 1:
        decoded = decoded.reshape(-1, channels)
        if mono:
            decoded = decoded.mean(axis=1)
    return AudioData(np.asarray(decoded, dtype=np.float32), sample_rate)


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write mono float samples as 16-bit PCM WAV."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    pcm = np.clip(mono, -1.0, 1.0)
    pcm = np.round(pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())


def resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Small dependency-free resampler for setup and smoke-test data."""

    if source_rate == target_rate:
        return np.asarray(samples, dtype=np.float32).copy()
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive")
    source = np.asarray(samples, dtype=np.float32).reshape(-1)
    if source.size == 0:
        return source.copy()
    output_length = max(1, round(source.size * target_rate / source_rate))
    old_positions = np.arange(source.size, dtype=np.float64)
    new_positions = np.linspace(0, source.size - 1, output_length)
    return np.interp(new_positions, old_positions, source).astype(np.float32)

