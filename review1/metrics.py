"""Objective and real-time measurements for the Review 1 report."""

from __future__ import annotations

from typing import Any
import time

import numpy as np


def _aligned(reference: np.ndarray, estimate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    estimate = np.asarray(estimate, dtype=np.float64).reshape(-1)
    length = min(reference.size, estimate.size)
    return reference[:length], estimate[:length]


def rms(signal: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean(signal * signal) + 1e-12)) if signal.size else 0.0


def snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference, estimate = _aligned(reference, estimate)
    if reference.size == 0:
        return float("nan")
    error = reference - estimate
    return float(10.0 * np.log10((np.sum(reference * reference) + 1e-12) / (np.sum(error * error) + 1e-12)))


def si_sdr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference, estimate = _aligned(reference, estimate)
    if reference.size == 0:
        return float("nan")
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    reference_energy = np.sum(reference * reference) + 1e-12
    scale = np.sum(estimate * reference) / reference_energy
    target = scale * reference
    residual = estimate - target
    return float(10.0 * np.log10((np.sum(target * target) + 1e-12) / (np.sum(residual * residual) + 1e-12)))


def real_time_factor(processing_seconds: float, audio_seconds: float) -> float:
    if audio_seconds <= 0:
        return float("nan")
    return float(processing_seconds / audio_seconds)


def evaluate_quality(clean: np.ndarray, degraded: np.ndarray, enhanced: np.ndarray) -> dict[str, Any]:
    """Return before/after metrics; no PESQ/STOI dependency is assumed."""

    input_snr = snr_db(clean, degraded)
    output_snr = snr_db(clean, enhanced)
    input_si_sdr = si_sdr_db(clean, degraded)
    output_si_sdr = si_sdr_db(clean, enhanced)
    return {
        "snr_input_db": input_snr,
        "snr_output_db": output_snr,
        "snr_improvement_db": output_snr - input_snr,
        "si_sdr_input_db": input_si_sdr,
        "si_sdr_output_db": output_si_sdr,
        "si_sdr_improvement_db": output_si_sdr - input_si_sdr,
        "optional_metrics": {
            "pesq": "not computed; install an explicitly selected PESQ implementation",
            "stoi": "not computed; install an explicitly selected STOI implementation",
            "wer": "not computed; run the enhanced WAV through a chosen ASR model",
        },
    }


def timed(function, *args, **kwargs):
    started = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - started

