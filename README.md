# Real-Time Speech Dereverberation and Denoising

This repository contains a dependency-light, single-channel WAV baseline for real-time enhancement of poor-quality recordings:

`WAV input -> causal STFT frames -> noise suppression -> causal dereverberation experiment -> ISTFT overlap-add -> WAV output`

The implementation is intentionally small enough to profile and explain. It does not claim to be RNNoise, DeepFilterNet, DCCRN, or full WPE. It provides a reproducible internal baseline and clear measurement hooks before a larger neural model is selected.

## Included

- `realtime_speech_enhancement/`: audio I/O, STFT helpers, controlled degradation generation, streaming enhancer, dataset manifests, metrics, and CLI.
- `problem_and_literature.md`: problem definition, objectives, five-paper survey, and research gap.
- `system_design.md`: system architecture and workflow diagram.
- `data/README.md`: dataset collection plan and justification.
- `evaluation_evidence.md`: requirement-to-evidence mapping.
- `initial_results.md`: reproducible initial measurements and demonstration sequence.
- `algorithm.md`: the implemented algorithm and its assumptions.
- `workflow.md`: the end-to-end experiment and demonstration workflow.
- `differences.md`: a technical comparison of five papers and this project.
- `tests/`: focused round-trip, degradation, streaming, and metric checks.

## Run it

The only required runtime dependency is NumPy.

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Generate a controlled smoke-test fixture, enhance it, and write before/after metrics:

```powershell
python -m realtime_speech_enhancement demo --out-dir artifacts/realtime_speech_enhancement_demo
```

The output directory contains `clean_fixture.wav`, `noisy_reverberant_fixture.wav`, `enhanced_fixture.wav`, `room_impulse_response.wav`, `generation.json`, and `evaluation.json`.

For a real recording:

```powershell
python -m realtime_speech_enhancement enhance input.wav --output artifacts/enhanced.wav --report artifacts/enhancement.json
python -m realtime_speech_enhancement evaluate --clean clean.wav --degraded input.wav --enhanced artifacts/enhanced.wav --report artifacts/evaluation.json
```

The evaluation WAVs must be aligned and use the same sample rate. The CLI measures SNR, SI-SDR, processing time, real-time factor, and the configured algorithmic latency. PESQ, STOI, and WER are explicitly left as optional follow-up measurements rather than invented values.

For a noisy 48 kHz speech test set, use the stronger dataset profile:

```powershell
python -m realtime_speech_enhancement enhance-batch input_folder `
  --output-root artifacts/noisy_testset_enhanced `
  --report-dir artifacts/noisy_testset_reports `
  --profile strong-denoise
```

The batch command preserves the input folder structure and writes one JSON timing report per WAV. It does not calculate quality metrics by itself; use `evaluate` when aligned clean files are available.

For matching clean, degraded, and enhanced folders, calculate dataset-level means with:

```powershell
python -m realtime_speech_enhancement evaluate-batch `
  --clean-root clean_folder `
  --degraded-root input_folder `
  --enhanced-root artifacts/noisy_testset_enhanced `
  --report artifacts/noisy_testset_reports/dataset_evaluation_summary.json
```

## Scope boundary

The current implementation covers controlled data preparation, a causal enhancement baseline, a dereverberation experiment hook, before/after measurement, and a technical comparison with five representative papers. Product integrations, external model benchmarking, downstream Whisper testing, and final deployment choices remain explicit extension points rather than undocumented assumptions.

The synthetic demo is a deterministic voiced-like signal for testing the pipeline. It is not a human-speech result. Use aligned clean/noisy/reverberant recordings for quantitative speech-quality claims.
