# Review 1: Real-Time Speech Dereverberation / Denoising

This repository contains only the Review 1 implementation requested from the attached technical brief. It is a dependency-light, single-channel WAV baseline for poor-quality recordings:

`WAV input -> causal STFT frames -> noise suppression -> causal dereverberation experiment -> ISTFT overlap-add -> WAV output`

The implementation is intentionally small enough to profile and explain. It does not claim to be RNNoise, DeepFilterNet, DCCRN, or full WPE. It provides a reproducible internal baseline and clear measurement hooks before a larger neural model is selected.

## Included

- `review1/`: audio I/O, STFT helpers, controlled degradation generation, streaming enhancer, metrics, and CLI.
- `algorithm.md`: the exact Review 1 algorithm and its assumptions.
- `workflow.md`: the end-to-end experiment and demonstration workflow.
- `differences.md`: a faculty-facing comparison of the five papers and this project.
- `tests/`: focused round-trip, degradation, streaming, and metric checks.

## Run it

The only required runtime dependency is NumPy.

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Generate a controlled smoke-test fixture, enhance it, and write before/after metrics:

```powershell
python -m review1 demo --out-dir artifacts/review1_demo
```

The output directory contains `clean_fixture.wav`, `noisy_reverberant_fixture.wav`, `enhanced_fixture.wav`, `room_impulse_response.wav`, `generation.json`, and `evaluation.json`.

For a real recording:

```powershell
python -m review1 enhance input.wav --output artifacts/enhanced.wav --report artifacts/enhancement.json
python -m review1 evaluate --clean clean.wav --degraded input.wav --enhanced artifacts/enhanced.wav --report artifacts/evaluation.json
```

The evaluation WAVs must be aligned and use the same sample rate. The CLI measures SNR, SI-SDR, processing time, real-time factor, and the configured algorithmic latency. PESQ, STOI, and WER are explicitly left as optional follow-up measurements rather than invented values.

## Scope boundary

The attached brief also mentions future product integrations, multiple open-source baselines, downstream Whisper testing, and final deployment choices. Those are not silently represented as completed work here. This folder completes the Review 1 core: controlled data preparation, a causal enhancement baseline, a dereverberation experiment hook, before/after measurement, and explanation material for the five-paper comparison.

The synthetic demo is a deterministic voiced-like signal for testing the pipeline. It is not a human speech result and should not be presented to faculty as a real speech-quality claim. Use aligned clean/noisy/reverberant recordings for the academic result.

