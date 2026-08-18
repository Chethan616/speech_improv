# Problem Definition and Literature Survey

## Problem definition

Speech captured by a low-cost or distant microphone can contain additive background noise, competing talkers, microphone noise, and room reverberation. Reverberation is especially damaging because reflected copies of the direct speech overlap later phonemes and smear temporal boundaries. The same distortions reduce perceived quality and can reduce automatic speech-recognition accuracy.

The project models the observed signal as a degraded version of clean speech:

```text
observed audio = room response * clean speech + additive noise
```

The engineering problem is to estimate a cleaner speech stream while the input is still arriving. The system must therefore balance enhancement quality, speech preservation, algorithmic latency, CPU cost, memory use, and downstream usefulness.

## Objectives

- Suppress stationary and non-stationary environmental noise.
- Reduce late reverberation without removing the direct speech signal.
- Preserve intelligibility, spectral detail, and speaker characteristics.
- Operate causally or with a clearly bounded look-ahead.
- Measure both acoustic improvement and real-time behavior.
- Provide an audio front end that can feed ASR, recording, voice typing, meetings, or conversational applications.

## Five-paper literature survey

| Paper | Main idea | Contribution to project design | Limitation relevant to this project |
|---|---|---|---|
| [Valin, 2018 — A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement](https://arxiv.org/abs/1709.08243) | Combines signal-processing structure, critical-band gains, a compact recurrent model, and pitch filtering. | Supports the decision to begin with a small, profileable real-time baseline instead of a large offline network. | Primarily a noise-suppression design; strong room reverberation is not its central target. |
| [Li et al., 2017 — Integrated Speech Enhancement Method Based on WPE and DNN for Dereverberation and Denoising](https://arxiv.org/abs/1708.08251) | Combines WPE dereverberation with DNN-based noise suppression and speech-variance estimation. | Motivates treating denoising and dereverberation as a joint problem and retaining a classical dereverberation comparison. | Multiple coupled components increase implementation and latency complexity. |
| [Hu et al., 2020 — DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement](https://www.isca-archive.org/interspeech_2020/hu20g_interspeech.html) | Uses complex-domain convolution and recurrence to model phase-aware spectral information. | Shows why a magnitude-only approach may leave phase-related distortion and provides a compact neural candidate. | Complex-valued models are harder to implement, tune, and profile than a simple spectral baseline. |
| [Schröter et al., 2022 — DeepFilterNet2](https://arxiv.org/abs/2205.05474) | Uses efficient deep filtering and harmonic structure for full-band embedded real-time enhancement. | Establishes a strong quality-versus-compute reference for later baseline comparisons. | Efficient denoising does not automatically solve dereverberation; that requirement needs explicit testing. |
| [Rosenbaum et al., 2025 — Deep-Learning Framework for Efficient Real-Time Speech Enhancement and Dereverberation](https://www.mdpi.com/1424-8220/25/3/630) | Extends efficient deep-filtering ideas toward explicit dereverberation. | Most closely matches the joint target of low-latency denoising and dereverberation. | Reproduction requires careful training, configuration, and hardware-specific validation. |

## Research gap

The papers provide strong individual solutions, but the project is concerned with the complete streaming path: controlled degradation, causal chunking, enhancement, speech preservation, measured latency, and downstream evaluation. The project baseline therefore serves as an interpretable reference against which stronger neural or WPE-based systems can be compared.

## Design decision derived from the survey

The first implementation uses a causal STFT pipeline with conservative spectral suppression and a bounded one-delay complex predictor. This choice is not claimed as a new neural architecture. It is a reproducible reference that exposes the quality/latency trade-off and leaves a clear interface for RNNoise, DeepFilterNet, DCCRN, WPE+DNN, or a custom trained model.

