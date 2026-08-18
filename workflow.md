# Review 1 Workflow

## 1. Review 1 boundary

This workflow completes the technical core described in the attached Review 1 brief. It stops at a reproducible enhancement baseline and evidence package. A product UI, a full Whisper application, and final model selection are outside this review.

## 2. One-command smoke test

Run the unit checks first:

```powershell
python -m unittest discover -s tests -v
```

Then run the deterministic end-to-end fixture:

```powershell
python -m realtime_speech_enhancement demo --out-dir artifacts/realtime_speech_enhancement_demo
```

This creates:

```text
artifacts/realtime_speech_enhancement_demo/
  clean_fixture.wav
  reverberant_fixture.wav
  noisy_reverberant_fixture.wav
  enhanced_fixture.wav
  room_impulse_response.wav
  generation.json
  evaluation.json
```

The fixture checks the software path only. It is not a speech benchmark.

## 3. Real-data workflow

### Step 1: choose aligned source material

For a defensible result, keep a clean reference, a degraded input, and the enhanced output aligned in sample count and sample rate. Use speaker-disjoint train, validation, and test partitions where possible.

Suitable Review 1 starting points from the brief are VoiceBank-DEMAND for supervised denoising, DNS Challenge material for noise diversity, REVERB-style corpora for reverberation, and controlled mixtures made from clean speech plus noise and room impulse responses.

### Step 2: create the degraded condition

When a clean reference and a noise recording are available:

```powershell
python -m realtime_speech_enhancement generate --out-dir artifacts/realtime_speech_enhancement_fixture --snr-db 5 --rt60 0.35
```

The generator is primarily a controlled development fixture. For the final report, document the source speech, noise type, SNR, room impulse response or measured room, and speaker split.

### Step 3: run streaming enhancement

```powershell
python -m realtime_speech_enhancement enhance input.wav `
  --output artifacts/enhanced.wav `
  --report artifacts/enhancement.json `
  --chunk-size 256
```

The reported real-time factor is processing time divided by audio duration. Values below `1.0` mean that this run processed faster than real time on that machine. The configured frame latency is reported separately because RTF and interactive latency are different quantities.

### Step 4: compare before and after

```powershell
python -m realtime_speech_enhancement evaluate `
  --clean clean.wav `
  --degraded input.wav `
  --enhanced artifacts/enhanced.wav `
  --report artifacts/evaluation.json
```

The report contains input/output SNR and SI-SDR plus their improvements. For the academic version, add PESQ/STOI if the chosen implementations are installed and run the enhanced audio through the selected ASR system to obtain WER before and after enhancement.

### Step 5: inspect failure cases

At minimum, listen to or visualize:

- speech with stationary background noise;
- speech with non-stationary competing noise;
- strong reverberation with little additive noise;
- an utterance whose first frames contain speech instead of noise-only context; and
- silence, where over-suppression or pumping may be audible.

Record when consonants disappear, the voice becomes metallic, reverberant tails remain, or the predictor becomes unstable. A failure case is evidence for the next model iteration; it is not a reason to hide the result.

## 4. Demonstration order for faculty

1. State the problem: uncontrolled microphones add noise and reverberation.
2. Show the input waveform or spectrogram.
3. Explain that the current baseline is causal STFT processing with conservative noise suppression and a one-delay dereverberation experiment.
4. Play the degraded recording.
5. Run or replay the enhanced output.
6. Show input/output acoustic metrics and measured RTF/latency.
7. If available, show ASR WER before and after enhancement.
8. Play one failure case and state the next controlled experiment.

## 5. Review 1 completion checklist

- [x] Problem and research gap documented.
- [x] Five core papers compared with the project.
- [x] Controlled noise plus room-reverberation data generation implemented.
- [x] Causal frame-based enhancement baseline implemented.
- [x] Dereverberation experiment switch implemented.
- [x] Before/after objective metrics implemented without fabricated results.
- [x] Real-time factor and nominal frame latency logged.
- [x] Deterministic smoke-test demonstration available.
- [ ] Real human-speech measurements supplied by the project owner.
- [ ] Optional PESQ/STOI/WER measurements supplied after selecting the evaluation tools and ASR model.

The last two items are data-dependent evidence collection, not missing software scaffolding. They cannot be honestly filled without the recordings and measurement choices.
