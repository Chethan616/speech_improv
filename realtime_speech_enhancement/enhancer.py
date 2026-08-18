"""Causal low-latency enhancement baseline for Review 1.

The implementation combines a conservative spectral noise suppressor with a
single-delay adaptive complex predictor. The predictor is deliberately called
WPE-inspired rather than WPE: it is a bounded one-tap experiment that keeps the
baseline dependency-free and makes the dereverberation trade-off measurable.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EnhancerConfig:
    frame_size: int = 512
    hop_size: int = 128
    noise_frames: int = 8
    noise_bootstrap_percentile: float = 35.0
    suppression_strength: float = 0.85
    gain_floor: float = 0.12
    noise_update_rate: float = 0.02
    dereverb_strength: float = 0.20
    dereverb_delay_frames: int = 2
    predictor_learning_rate: float = 0.03

    def validate(self) -> None:
        if self.frame_size < 4 or self.hop_size <= 0 or self.hop_size > self.frame_size:
            raise ValueError("Require frame_size >= 4 and 0 < hop_size <= frame_size")
        if self.noise_frames < 1:
            raise ValueError("noise_frames must be positive")
        if not 0.0 <= self.noise_bootstrap_percentile <= 100.0:
            raise ValueError("noise_bootstrap_percentile must be in [0, 100]")
        if not 0.0 <= self.suppression_strength <= 1.0:
            raise ValueError("suppression_strength must be in [0, 1]")
        if not 0.0 < self.gain_floor <= 1.0:
            raise ValueError("gain_floor must be in (0, 1]")
        if not 0.0 <= self.noise_update_rate <= 1.0:
            raise ValueError("noise_update_rate must be in [0, 1]")
        if self.dereverb_strength < 0.0:
            raise ValueError("dereverb_strength must be non-negative")
        if self.dereverb_delay_frames < 1:
            raise ValueError("dereverb_delay_frames must be positive")


class StreamingEnhancer:
    """Stateful framed processor with bounded look-ahead."""

    def __init__(self, config: EnhancerConfig | None = None):
        self.config = config or EnhancerConfig()
        self.config.validate()
        size = self.config.frame_size
        self.window = np.hanning(size).astype(np.float32)
        self.window_power = self.window.astype(np.float64) ** 2
        self.pending = np.zeros(0, dtype=np.float32)
        self.ola = np.zeros(size, dtype=np.float64)
        self.normalization = np.zeros(size, dtype=np.float64)
        self.noise_power: np.ndarray | None = None
        self.noise_bootstrap: list[np.ndarray] = []
        self.history: deque[np.ndarray] = deque(maxlen=self.config.dereverb_delay_frames + 1)
        self.predictor = np.zeros(size // 2 + 1, dtype=np.complex128)
        self.processed_frames = 0

    @property
    def algorithmic_latency_frames(self) -> int:
        return self.config.frame_size

    def _noise_and_vad(self, power: np.ndarray) -> bool:
        if self.noise_power is None:
            self.noise_bootstrap.append(power.copy())
            if len(self.noise_bootstrap) >= self.config.noise_frames:
                self.noise_power = np.percentile(
                    np.stack(self.noise_bootstrap, axis=0),
                    self.config.noise_bootstrap_percentile,
                    axis=0,
                )
            return False

        frame_power = float(np.mean(power))
        noise_level = float(np.mean(self.noise_power)) + 1e-12
        speech_present = frame_power > 2.0 * noise_level
        if not speech_present:
            rate = self.config.noise_update_rate
            self.noise_power = (1.0 - rate) * self.noise_power + rate * power
        return speech_present

    def _enhance_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        epsilon = 1e-10
        power = np.abs(spectrum) ** 2
        speech_present = self._noise_and_vad(power)
        if self.noise_power is None:
            # Do not treat the first speech frame as the complete noise PSD.
            # During bootstrap, pass the frame through and wait for the
            # configured robust estimate to be ready.
            self.history.append(spectrum.astype(np.complex128).copy())
            self.processed_frames += 1
            return spectrum.astype(np.complex128)

        noise = np.maximum(self.noise_power, epsilon)
        gain = 1.0 - self.config.suppression_strength * noise / (power + epsilon)
        gain = np.clip(gain, self.config.gain_floor, 1.0)
        # Mild frequency smoothing reduces isolated musical-noise-like peaks.
        gain = (0.25 * np.roll(gain, 1) + 0.5 * gain + 0.25 * np.roll(gain, -1)).astype(np.float64)
        gain[0] = gain[1] if gain.size > 1 else gain[0]
        gain[-1] = gain[-2] if gain.size > 1 else gain[-1]

        dereverberated = spectrum.astype(np.complex128)
        delay = self.config.dereverb_delay_frames
        if self.config.dereverb_strength > 0.0 and len(self.history) >= delay:
            delayed = self.history[-delay]
            prediction = self.predictor * delayed
            prediction_magnitude = np.abs(prediction)
            cap = 0.8 * np.abs(spectrum)
            prediction *= np.minimum(1.0, cap / (prediction_magnitude + epsilon))
            dereverberated = spectrum - self.config.dereverb_strength * prediction

            # Normalized complex LMS update. This is the small causal
            # dereverberation experiment, not a claim of full batch WPE.
            update = dereverberated * np.conj(delayed) / (np.abs(delayed) ** 2 + epsilon)
            self.predictor += self.config.predictor_learning_rate * update
            predictor_magnitude = np.abs(self.predictor)
            self.predictor *= np.minimum(1.0, 0.95 / (predictor_magnitude + epsilon))

        self.history.append(spectrum.astype(np.complex128).copy())
        output = dereverberated * gain
        if not speech_present:
            output *= 0.98
        self.processed_frames += 1
        return output

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        if frame.size != self.config.frame_size:
            raise ValueError("process_frame requires exactly frame_size samples")
        spectrum = np.fft.rfft(frame * self.window)
        enhanced_spectrum = self._enhance_spectrum(spectrum)
        enhanced_frame = np.fft.irfft(enhanced_spectrum, n=self.config.frame_size)
        enhanced_frame *= self.window

        self.ola += enhanced_frame
        self.normalization += self.window_power
        hop = self.config.hop_size
        output = self.ola[:hop] / np.maximum(self.normalization[:hop], 1e-8)
        self.ola[:-hop] = self.ola[hop:]
        self.ola[-hop:] = 0.0
        self.normalization[:-hop] = self.normalization[hop:]
        self.normalization[-hop:] = 0.0
        return output.astype(np.float32)

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if chunk.size:
            self.pending = np.concatenate((self.pending, chunk))
        output: list[np.ndarray] = []
        while self.pending.size >= self.config.frame_size:
            frame = self.pending[: self.config.frame_size]
            self.pending = self.pending[self.config.hop_size :]
            output.append(self.process_frame(frame))
        if not output:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(output)

    def flush(self) -> np.ndarray:
        output: list[np.ndarray] = []
        if self.pending.size:
            padded = np.pad(self.pending, (0, self.config.frame_size - self.pending.size))
            self.pending = np.zeros(0, dtype=np.float32)
            output.append(self.process_frame(padded))
        tail_size = self.config.frame_size - self.config.hop_size
        if tail_size > 0 and np.any(self.normalization[:tail_size] > 1e-8):
            tail = self.ola[:tail_size] / np.maximum(self.normalization[:tail_size], 1e-8)
            output.append(tail.astype(np.float32))
        self.ola.fill(0.0)
        self.normalization.fill(0.0)
        return np.concatenate(output) if output else np.zeros(0, dtype=np.float32)


def enhance_audio(
    samples: np.ndarray,
    config: EnhancerConfig | None = None,
    chunk_size: int = 256,
) -> np.ndarray:
    """Enhance samples through the same chunked path used by the CLI."""

    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    enhancer = StreamingEnhancer(config)
    output: list[np.ndarray] = []
    for start in range(0, samples.size, chunk_size):
        chunk_output = enhancer.process_chunk(samples[start : start + chunk_size])
        if chunk_output.size:
            output.append(chunk_output)
    flushed = enhancer.flush()
    if flushed.size:
        output.append(flushed)
    if not output:
        return np.zeros_like(samples)
    result = np.concatenate(output)
    if result.size < samples.size:
        result = np.pad(result, (0, samples.size - result.size))
    return result[: samples.size].astype(np.float32)
