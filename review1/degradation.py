"""Controlled noise and room-reverberation generation for Review 1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DegradedPair:
    clean: np.ndarray
    reverberant: np.ndarray
    noisy_reverberant: np.ndarray
    room_impulse_response: np.ndarray
    snr_db: float
    rt60_seconds: float


def _rms(signal: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=np.float64)
    return float(np.sqrt(np.mean(signal * signal) + 1e-12))


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Add noise at a target RMS SNR, repeating or trimming the noise as needed."""

    clean = np.asarray(clean, dtype=np.float32).reshape(-1)
    noise = np.asarray(noise, dtype=np.float32).reshape(-1)
    if clean.size == 0:
        return clean.copy()
    if noise.size == 0:
        return clean.copy()
    repeated = np.resize(noise, clean.size)
    clean_rms = _rms(clean)
    noise_rms = _rms(repeated)
    if noise_rms < 1e-10:
        return clean.copy()
    desired_noise_rms = clean_rms / (10.0 ** (snr_db / 20.0))
    return (clean + repeated * (desired_noise_rms / noise_rms)).astype(np.float32)


def synthetic_room_impulse_response(
    sample_rate: int,
    rt60_seconds: float = 0.35,
    seed: int = 7,
    reflection_count: int | None = None,
) -> np.ndarray:
    """Create a deterministic, causal synthetic room impulse response.

    This is a controlled data-generation fixture, not a measurement of a real
    room. The decay reaches roughly -60 dB by rt60_seconds.
    """

    if sample_rate <= 0 or rt60_seconds <= 0:
        raise ValueError("sample_rate and rt60_seconds must be positive")
    length = max(2, int(round(sample_rate * rt60_seconds)))
    rng = np.random.default_rng(seed)
    time = np.arange(length, dtype=np.float64) / sample_rate
    decay = 10.0 ** (-3.0 * time / rt60_seconds)
    ir = np.zeros(length, dtype=np.float64)
    ir[0] = 1.0
    count = reflection_count if reflection_count is not None else max(8, min(64, length // 160))
    for _ in range(count):
        index = int(rng.integers(1, length))
        ir[index] += rng.normal(0.0, 0.22) * decay[index]
    ir *= np.hanning(length) * 0.8 + 0.2
    ir[0] = 1.0
    norm = np.sqrt(np.sum(ir * ir))
    return (ir / max(norm, 1e-12)).astype(np.float32)


def degrade_audio(
    clean: np.ndarray,
    noise: np.ndarray,
    sample_rate: int,
    snr_db: float = 5.0,
    rt60_seconds: float = 0.35,
    seed: int = 7,
) -> DegradedPair:
    """Generate a clean/reverberant/noisy-reverberant supervised pair."""

    clean = np.asarray(clean, dtype=np.float32).reshape(-1)
    ir = synthetic_room_impulse_response(sample_rate, rt60_seconds, seed=seed)
    reverberant = np.convolve(clean, ir, mode="full")[: clean.size].astype(np.float32)
    noisy_reverberant = mix_at_snr(reverberant, noise, snr_db)

    peak = max(1.0, float(np.max(np.abs(noisy_reverberant))) if noisy_reverberant.size else 1.0)
    return DegradedPair(
        clean=(clean / peak).astype(np.float32),
        reverberant=(reverberant / peak).astype(np.float32),
        noisy_reverberant=(noisy_reverberant / peak).astype(np.float32),
        room_impulse_response=ir,
        snr_db=float(snr_db),
        rt60_seconds=float(rt60_seconds),
    )

