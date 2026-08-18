# Dataset Selection, Collection, and Justification

## 1. Purpose

The project has two related goals: improve speech captured in noise and reduce the effect of room reverberation while processing continuously on modest hardware. No single source represents both problems equally well. The collection plan therefore combines paired denoising data, varied noise and room responses, reverberation-focused evaluation, and small controlled mixtures for repeatable development.

Raw audio is not committed to this repository. The repository stores code, manifests, configuration, checksums, and small synthetic fixtures only.

## 2. What the five papers used

| Research paper | Dataset(s) used by the paper | Why the researchers used those data | Relation to this project |
|---|---|---|---|
| [Valin, 2018 — A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement](https://arxiv.org/abs/1709.08243) | Artificial mixtures made from clean speech sources, including the McGill TSP speech database and the NTT Multi-Lingual Speech Database, with added noise and augmentation. | Artificial mixing gives clean targets and makes the small real-time model easy to train and compare across noise conditions. | We follow the clean-target idea for controlled debugging, but add explicit room responses and a separate reverberation evaluation because dereverberation is central here. |
| [Li et al., 2017 — Integrated Speech Enhancement Method Based on WPE and DNN for Dereverberation and Denoising](https://arxiv.org/abs/1708.08251) | The 2014 [REVERB Challenge](https://reverb2014.audiolabs-erlangen.de/index.html), built from WSJCAM0 speech convolved with measured room impulse responses and mixed with noise. | REVERB provides paired reverberant/clean conditions, simulated and real-room test data, and a common dereverberation benchmark. | We use REVERB as the later formal dereverberation benchmark, while beginning with smaller controlled mixtures so causal behavior and failure cases can be debugged first. |
| [Hu et al., 2020 — DCCRN](https://www.isca-archive.org/interspeech_2020/hu20g_interspeech.html) | A simulated WSJ0 set with MUSAN noise, followed by the [DNS Challenge](https://github.com/microsoft/DNS-Challenge) data with clean speech, noise, and simulated room responses. | WSJ0/MUSAN supports controlled ablations over SNR; DNS provides a larger, diverse common testbed and includes reverberant conditions for real-time comparison. | We use the same controlled-mixture principle and plan to use DNS for broader comparison, but keep the initial implementation dependency-light and CPU-profiled. |
| [Schröter et al., 2022 — DeepFilterNet2](https://arxiv.org/abs/2205.05474) | English DNS4 training data; VoiceBank+DEMAND test data; DNS4 blind test data. The paper reports clean speech, noise, and room/transfer-response data with train/validation/test splits. | DNS4 gives large full-band training diversity; VoiceBank+DEMAND gives a paired, speaker-exclusive test set; the blind test checks generalization. | VoiceBank-DEMAND is our first supervised sanity benchmark because it is compact and paired. DNS is the planned larger training/development source. |
| [Rosenbaum et al., 2025 — Efficient Real-Time Speech Enhancement and Dereverberation](https://www.mdpi.com/1424-8220/25/3/630) | DNS Challenge training data with speech, noise, and room responses; VCTK/DEMAND-style test data for speech enhancement and dereverberation comparisons. | The data match the paper's joint goal: varied noise, reverberation, and a measurable real-time deployment trade-off. | This is the selected base-paper direction. We retain the comparable DNS/DEMAND idea, but add a staged low-power pipeline, controlled ablations, explicit manifests, and end-to-end latency measurements. |

## 3. Dataset selection for this project

| Project source | Role in the work | Why it is needed | Planned use |
|---|---|---|---|
| [VoiceBank-DEMAND / Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/2791) | Paired clean and noisy speech | The clean/noisy pairing allows before/after objective measures without estimating an unknown reference. It is also small enough for an initial experiment. | Start with a speaker-disjoint test/development subset. Use it for denoising sanity checks and, when supported by the metric package, PESQ/STOI-style comparison. |
| [DNS Challenge data and scripts](https://github.com/microsoft/DNS-Challenge) | Broad clean speech, noise, and room-response source | It provides varied real-world noise and a documented mixture-generation workflow. It is the most useful larger source for later training and generalization tests. | Download only an approved development subset first. Keep the full source outside Git because of size and storage requirements. |
| [REVERB Challenge](https://reverb2014.audiolabs-erlangen.de/index.html) | Formal dereverberation and distant-speech evaluation | It directly tests reverberation using measured rooms and includes real and simulated conditions. | Add after the streaming baseline is stable. Use it to compare dereverberation separately from denoising. |
| Controlled project mixtures | Development, ablation, and demonstration fixture | Clean speech, SNR, room response, RT60, and random seed are known exactly. This makes algorithm changes reproducible and exposes causal or overlap-add bugs quickly. | Use for smoke tests, denoising-only versus joint settings, latency checks, and the current demonstration. Do not present it as a human-speech benchmark. |

## 4. Why the project uses both the same and different data

The project deliberately shares dataset families with the literature where comparability is valuable. DNS is shared with DCCRN, DeepFilterNet2, and the selected base paper because it contains varied speech/noise conditions and supports challenge-style evaluation. VoiceBank-DEMAND is shared with DeepFilterNet2 because its paired clean/noisy test set makes a small, understandable quality check possible. REVERB is shared with the WPE+DNN line because it tests room reverberation directly.

The project also differs in the first stage. The full challenge datasets are large and their training pipelines can hide whether a streaming bug comes from the model, the data loader, or the overlap-add implementation. Controlled mixtures are therefore used first, with fixed SNR, RT60, sample rate, and random seed. This is a development choice, not a claim that synthetic data are sufficient for final validation.

The final evaluation should report results in separate groups:

1. controlled mixtures for repeatability and ablations;
2. VoiceBank-DEMAND for paired denoising comparison;
3. DNS or an approved subset for varied noise and reverberation; and
4. REVERB for a focused dereverberation comparison.

This separation prevents a good result on a synthetic fixture from being presented as proof of general performance.

## 5. Collection and storage procedure

For every source, record the download or access date, version, licence, citation, local path, sample rate, channel count, file count, and checksum where available. Keep raw files in local storage and generate a manifest for each split.

```text
data/
  raw/          original downloaded audio; never committed
  processed/    resampled or normalized working copies; never committed
  manifests/    JSON metadata and checksums; safe to commit
```

Use speaker-disjoint train, validation, and test splits wherever the source supports them. Convert audio only into `data/processed/`; never overwrite the original. The current manifest command is:

```powershell
python -m realtime_speech_enhancement.dataset data/raw/voicebank_demand/test `
  --dataset-name voicebank_demand `
  --split test `
  --output data/manifests/voicebank_demand_test.json
```

## 6. Practical decision for the current milestone

The current milestone uses the controlled fixture because it is immediately reproducible and does not require redistributing external audio. The next data milestone is a small VoiceBank-DEMAND subset, followed by an approved DNS subset and then REVERB when formal dereverberation testing is required. This order supports the project's main constraint: demonstrate useful processing with low latency and low CPU cost before increasing dataset and model size.
