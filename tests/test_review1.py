from __future__ import annotations

import unittest

import numpy as np

from review1.degradation import degrade_audio
from review1.enhancer import EnhancerConfig, enhance_audio
from review1.metrics import evaluate_quality
from review1.stft import istft, stft


class Review1Tests(unittest.TestCase):
    def test_stft_round_trip(self) -> None:
        signal = np.sin(np.linspace(0.0, 40.0, 4096, endpoint=False)).astype(np.float32)
        spectra, length = stft(signal, frame_size=256, hop_size=64)
        reconstructed = istft(spectra, frame_size=256, hop_size=64, length=length)
        self.assertLess(float(np.max(np.abs(signal[128:-128] - reconstructed[128:-128]))), 1e-4)

    def test_degradation_has_requested_shape(self) -> None:
        clean = np.ones(1600, dtype=np.float32) * 0.1
        noise = np.random.default_rng(2).standard_normal(1600).astype(np.float32)
        pair = degrade_audio(clean, noise, 16000, snr_db=5.0, rt60_seconds=0.2)
        self.assertEqual(pair.clean.shape, clean.shape)
        self.assertEqual(pair.reverberant.shape, clean.shape)
        self.assertEqual(pair.noisy_reverberant.shape, clean.shape)
        self.assertGreater(pair.room_impulse_response.size, 1)

    def test_streaming_output_is_finite_and_aligned(self) -> None:
        samples = np.random.default_rng(3).standard_normal(4000).astype(np.float32) * 0.05
        result = enhance_audio(samples, EnhancerConfig(frame_size=256, hop_size=64), chunk_size=73)
        self.assertEqual(result.shape, samples.shape)
        self.assertTrue(np.isfinite(result).all())

    def test_metrics_have_before_after_fields(self) -> None:
        clean = np.zeros(1000, dtype=np.float32)
        clean[100:900] = 0.2
        degraded = clean + 0.05 * np.random.default_rng(4).standard_normal(1000)
        enhanced = clean + 0.02 * np.random.default_rng(5).standard_normal(1000)
        report = evaluate_quality(clean, degraded, enhanced)
        self.assertIn("snr_improvement_db", report)
        self.assertGreater(report["snr_improvement_db"], 0.0)


if __name__ == "__main__":
    unittest.main()

