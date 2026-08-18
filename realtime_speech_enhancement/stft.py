"""Offline STFT helpers used for diagnostics and round-trip tests."""

from __future__ import annotations

import numpy as np


def analysis_window(frame_size: int) -> np.ndarray:
    if frame_size < 2:
        raise ValueError("frame_size must be at least 2")
    return np.hanning(frame_size).astype(np.float32)


def stft(
    signal: np.ndarray,
    frame_size: int = 512,
    hop_size: int = 128,
) -> tuple[np.ndarray, int]:
    """Return (one-sided spectra, original_length) with zero-padding at the end."""

    if frame_size <= 0 or hop_size <= 0 or hop_size > frame_size:
        raise ValueError("Require 0 < hop_size <= frame_size")
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    original_length = signal.size
    if original_length == 0:
        return np.empty((0, frame_size // 2 + 1), dtype=np.complex64), 0
    frame_count = max(1, int(np.ceil(max(0, original_length - frame_size) / hop_size)) + 1)
    padded_length = (frame_count - 1) * hop_size + frame_size
    padded = np.pad(signal, (0, padded_length - original_length))
    window = analysis_window(frame_size)
    spectra = np.stack(
        [np.fft.rfft(padded[start : start + frame_size] * window) for start in range(0, padded_length - frame_size + 1, hop_size)]
    )
    return spectra.astype(np.complex64), original_length


def istft(
    spectra: np.ndarray,
    frame_size: int = 512,
    hop_size: int = 128,
    length: int | None = None,
) -> np.ndarray:
    """Reconstruct a real signal from one-sided spectra using overlap-add."""

    spectra = np.asarray(spectra)
    if spectra.ndim != 2:
        raise ValueError("spectra must have shape [frames, frequency_bins]")
    if spectra.shape[1] != frame_size // 2 + 1:
        raise ValueError("frequency-bin count does not match frame_size")
    if spectra.shape[0] == 0:
        return np.zeros(0 if length is None else length, dtype=np.float32)
    window = analysis_window(frame_size)
    total_length = (spectra.shape[0] - 1) * hop_size + frame_size
    output = np.zeros(total_length, dtype=np.float64)
    normalization = np.zeros(total_length, dtype=np.float64)
    for index, spectrum in enumerate(spectra):
        start = index * hop_size
        frame = np.fft.irfft(spectrum, n=frame_size)
        output[start : start + frame_size] += frame * window
        normalization[start : start + frame_size] += window.astype(np.float64) ** 2
    valid = normalization > 1e-8
    output[valid] /= normalization[valid]
    output[~valid] = 0.0
    if length is not None:
        if length < 0:
            raise ValueError("length must be non-negative")
        output = output[:length]
        if output.size < length:
            output = np.pad(output, (0, length - output.size))
    return output.astype(np.float32)

