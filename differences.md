# Five Papers and Project Positioning

## 1. Project position

The five papers are research systems that each emphasize a particular modeling idea: hybrid DSP plus a small recurrent network, WPE plus DNN, complex phase-aware recurrence, efficient deep filtering, and efficient deep filtering extended toward dereverberation. This project is a reproducible streaming pipeline built around the application constraint: process poor-quality audio continuously, measure latency, preserve speech, and make the output useful to downstream ASR. The current code is a lightweight causal baseline, not a claim that it already matches any paper's final neural model.

## 2. Selected base paper

**Decision: yes.** The selected base paper is [Rosenbaum et al., 2025 — Deep-Learning Framework for Efficient Real-Time Speech Enhancement and Dereverberation](https://www.mdpi.com/1424-8220/25/3/630).

This is the most appropriate architectural base for the project because the primary goals are fastness and low processing power. The paper starts from efficient deep filtering, identifies why a short input context limits dereverberation, and adds delayed information to model late reflections. It also evaluates the trade-off between simultaneous and two-step processing using computational-efficiency measures. Its reported comparison of parameter count, MACs per second, and real-time factor gives the project a concrete way to discuss quality versus compute.

The project is not a reproduction of the paper. The paper supplies the design direction; this project adds the surrounding reproducible system and deployment evidence:

- causal microphone/WAV chunking with explicit bounded latency;
- controlled noisy and reverberant data generation plus dataset manifests;
- denoising-only versus joint denoising/dereverberation ablations;
- CPU, memory, model-size, real-time-factor, and end-to-end-latency measurements;
- downstream ASR WER comparison when aligned speech and an ASR model are available; and
- failure-case analysis under noise, reverberation, silence, and speech-preservation stress.

The current dependency-light spectral baseline validates the streaming and measurement path first. The next enhancement-engine replacement should be an efficient DeepFilterNet-family model aligned with the selected paper's delayed dereverberation principle.

## 3. Comparison table

| Paper | Main representation / algorithm | Noise handling | Dereverberation | Real-time emphasis | Main strength | Difference from this project |
|---|---|---|---|---|---|---|
| [Valin, 2018 — A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement](https://arxiv.org/abs/1709.08243) | Critical-band gains from a small recurrent model plus a pitch filter | Strong | Limited as a primary target | Very strong | Excellent low-compute design lesson | This project keeps the low-compute causal idea but makes room reverberation and downstream evaluation explicit; the current baseline is spectral rather than RNNoise's learned hybrid model. |
| [Li et al., 2017 — Integrated Speech Enhancement Method Based on WPE and DNN for Dereverberation and Denoising](https://arxiv.org/abs/1708.08251) | WPE with a DNN that supports noise suppression and speech-variance estimation | Jointly addressed | Strong | Moderate | Directly combines denoising and dereverberation | This project uses a bounded one-delay causal predictor for the first experiment; it is simpler and easier to profile, but it is not full WPE and should not be presented as equivalent. |
| [Hu et al., 2020 — DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement](https://www.isca-archive.org/interspeech_2020/hu20g_interspeech.html) | Complex convolution and recurrent modeling of complex spectra | Strong | Can help through phase-aware mapping | Strong | Shows why phase and compact complex models matter | This project keeps complex spectra at the signal-processing boundary but does not yet learn a complex mask or complex mapping. |
| [Schröter et al., 2022 — DeepFilterNet2](https://arxiv.org/abs/2205.05474) | Efficient deep filtering with harmonic structure and embedded-friendly components | Strong | Limited-to-moderate in the standard denoising focus | Very strong | Practical quality/compute trade-off for full-band audio | This project is a transparent dependency-light baseline; DeepFilterNet2 is a learned full-band reference that should be measured separately when the target runtime is selected. |
| [Rosenbaum et al., 2025 — Deep-Learning Framework for Efficient Real-Time Speech Enhancement and Dereverberation](https://www.mdpi.com/1424-8220/25/3/630) | Efficient deep filtering extended with explicit dereverberation analysis | Strong | Strong and central | Strong | Closest paper to the joint target | Selected as the base paper. We add causal I/O, data generation, low-power profiling, downstream ASR evaluation, and a replaceable implementation path rather than reproducing the research architecture verbatim. |

## 4. Dataset choices in the papers and this project

| Paper | Dataset(s) used by the researchers | Why that choice fits the paper | What this project does with the same or a different choice |
|---|---|---|---|
| [Valin, 2018 — RNNoise](https://arxiv.org/abs/1709.08243) | Artificial noisy speech built from clean speech sources, including McGill TSP and NTT Multi-Lingual Speech Database. | Clean targets and artificial mixing make a compact low-latency denoiser easier to train and compare. | We use controlled mixtures for the same reproducibility benefit, but add measured room responses and a separate dereverberation objective. |
| [Li et al., 2017 — WPE + DNN](https://arxiv.org/abs/1708.08251) | 2014 REVERB Challenge data: WSJCAM0 speech, measured RIRs, additive noise, and simulated/real test conditions. | REVERB is designed to test distant speech and dereverberation, which is the paper's central problem. | We plan to use REVERB for formal dereverberation testing. We start with controlled data so streaming, latency, and failure cases can be verified before the larger benchmark. |
| [Hu et al., 2020 — DCCRN](https://www.isca-archive.org/interspeech_2020/hu20g_interspeech.html) | Simulated WSJ0 + MUSAN mixtures, then DNS Challenge data with large speech/noise collections and simulated RIRs. | WSJ0/MUSAN supports controlled SNR experiments; DNS supplies a common, diverse real-time challenge testbed. | We keep the controlled-mixture idea and plan DNS for broader comparison, but use a smaller staged subset because the project's first constraint is CPU and streaming behavior. |
| [Schröter et al., 2022 — DeepFilterNet2](https://arxiv.org/abs/2205.05474) | English DNS4 for training; VoiceBank+DEMAND and DNS4 blind test data for evaluation. | DNS4 provides full-band diversity; VoiceBank+DEMAND provides a paired, speaker-exclusive test; the blind set checks generalization. | We take VoiceBank-DEMAND as the first external paired benchmark and DNS as the later broad source. The current fixture remains separate for quick reproducible checks. |
| [Rosenbaum et al., 2025 — selected base paper](https://www.mdpi.com/1424-8220/25/3/630) | DNS Challenge training data with speech, noise, and room responses; VCTK/DEMAND-style evaluation data. | The data match the joint denoising/dereverberation target and let the authors compare quality with real-time cost. | We retain the comparable DNS/DEMAND direction, then add controlled data, manifests, explicit denoising-only versus joint ablations, CPU/memory/RTF/latency profiling, and downstream ASR checks. |

### Why the datasets are not identical

The project shares DNS, VoiceBank-DEMAND, and REVERB with the literature where this improves comparability. It does not claim to reproduce every paper's exact training split. Full challenge data are large, access and licences differ, and a large training run would make it harder to isolate a causal streaming bug. Controlled mixtures are therefore used first, followed by a small paired external benchmark, then larger DNS and REVERB evaluations.

The evaluation will keep these groups separate: controlled mixtures for repeatability, VoiceBank-DEMAND for paired denoising, DNS for varied noise and reverberation, and REVERB for focused dereverberation. This prevents a synthetic demonstration from being reported as a general human-speech result.

## 5. Difference along the important dimensions

### Objective

The papers mainly optimize an enhancement model or algorithm. This project combines controlled input generation, streaming processing, quality metrics, real-time factor, nominal frame latency, and a downstream ASR measurement slot.

### Processing mode

The project is designed around chunks arriving while a speaker is talking. The enhancer keeps state across chunks, uses only current and permitted past frames, and flushes the final overlap-add tail. An offline model can still be compared, but it must not be described as causal without a look-ahead audit.

### Dereverberation

In the current code, dereverberation is an explicitly switchable one-delay complex predictor. This gives a baseline comparison:

```text
denoising only:     dereverb_strength = 0
joint baseline:     dereverb_strength > 0
```

The five papers provide stronger or more specialized dereverberation ideas. The project's contribution at this stage is to make the trade-off measurable under a streaming constraint, not to claim a new dereverberation theory.

### Speech preservation

The project uses a gain floor and conservative suppression because the goal is not maximum removal of every noise bin. It evaluates SNR and SI-SDR now, with PESQ, STOI, and WER left as explicit optional measurements. Removing noise while damaging consonants or speaker identity is not a successful enhancement result.

### Deployment and reuse

Each paper is a particular algorithm or model. The project is a reusable audio front-end boundary:

```text
microphone / recording -> enhancement engine -> STT, voice typing,
                              recording, meetings, or conversational AI
```

The enhancement engine can later be replaced by RNNoise, DeepFilterNet, DCCRN, WPE+DNN, or the selected custom model without rebuilding the surrounding measurement and application interface.

## 6. What is distinctive about the current implementation?

The defensible distinction is the combination and evaluation framing:

- low-latency streaming is a first-class requirement;
- dereverberation is not hidden inside a denoising-only claim;
- the output is intended for multiple downstream speech applications;
- the same run records acoustic quality and real-time behavior; and
- the project documents failure cases and ASR evaluation rather than reporting only a cleaner-looking waveform.

This is a system-level contribution and an experimental methodology. It should not be described as a new neural architecture until a custom trained model and an appropriate experimental comparison exist.

## 7. Concise project explanation

> RNNoise taught us that a small hybrid DSP and recurrent model can be useful at very low latency. WPE plus DNN showed that denoising and dereverberation should be considered together. DCCRN showed the value of phase-aware complex modeling. DeepFilterNet2 showed how deep filtering can achieve a strong quality-versus-compute trade-off, while the 2025 efficient dereverberation work is closest to the joint target. The current implementation therefore starts with a transparent causal STFT baseline: conservative spectral noise suppression, a bounded one-delay dereverberation experiment, overlap-add reconstruction, and measurements of SNR, SI-SDR, real-time factor, and latency. The project evaluates enhancement as a reusable streaming front-end for downstream speech systems, not only as an offline waveform cleaner.
