# Implementation Algorithm

## 1. Purpose

The current implementation is a causal, single-channel baseline for streaming speech enhancement. It targets two distortions at once:

1. additive/background noise; and
2. late reverberation or a reverberant tail.

The output is an enhanced audio stream with bounded frame-based latency. The implementation is designed to be measurable and replaceable, so a future neural model can occupy the same enhancement-engine boundary.

## 2. Signal path

```text
input WAV / microphone chunks
        |
        v
buffer short overlapping frames
        |
        v
window + causal STFT
        |
        +--> update noise PSD when the frame is likely non-speech
        |
        +--> conservative spectral gain for noise suppression
        |
        +--> one-delay adaptive complex predictor for dereverberation
        |
        v
enhanced complex spectrum
        |
        v
inverse STFT + overlap-add
        |
        v
enhanced stream and timing report
```

## 3. Frame configuration

The default configuration is:

| Parameter | Default | Meaning |
|---|---:|---|
| Sample rate | input rate | The model does not silently resample the input. |
| Frame size | 512 samples | Analysis window and maximum frame look-ahead. |
| Hop size | 128 samples | 75% overlap for smooth overlap-add reconstruction. |
| Noise bootstrap | 8 frames | Initial PSD estimate; the start of the file should contain useful noise context. |
| Gain floor | 0.12 | Prevents complete removal of speech frequency bins. |
| Suppression strength | 0.85 | Controls the conservative spectral suppression. |
| Dereverb strength | 0.20 | Strength of the causal one-delay predictor subtraction. |
| Predictor delay | 2 frames | Uses permitted history only; no future frames are used. |

For sample rate `f_s`, the nominal algorithmic frame latency is `512 / f_s` seconds. The measured real-time factor includes the actual machine, Python runtime, input length, and chunk size.

## 4. Noise suppression

For each frame `x_t`, the implementation calculates the one-sided complex spectrum:

```text
X_t = RFFT(window * x_t)
P_t = |X_t|^2
```

During the bootstrap period, `P_t` is averaged to form a noise power spectral density estimate `N_t`. When a later frame is below the simple energy-based voice-activity threshold, the estimate is updated slowly:

```text
N_t = (1 - alpha) * N_(t-1) + alpha * P_t
```

The gain is a clipped spectral-subtraction style gain:

```text
G_t = clip(1 - strength * N_t / (P_t + epsilon), gain_floor, 1)
```

A small frequency smoothing operation reduces isolated, unstable gain peaks. The lower bound is important: this baseline prefers residual noise over aggressive speech damage.

## 5. Dereverberation experiment

The first implementation uses a one-delay causal complex predictor:

```text
late_prediction_t = C_t * X_(t-D)
Y_t = X_t - dereverb_strength * late_prediction_t
```

`C_t` is updated with a normalized complex LMS-style rule using the current residual and the delayed spectrum. The predicted magnitude is capped before subtraction so the experiment cannot create an arbitrarily large correction.

This is a deliberately bounded experiment. It is WPE-inspired because it predicts a delayed spectral component, but it is not a full batch or multichannel WPE implementation. That distinction should be stated clearly during the review.

The noise-suppression gain is then applied:

```text
Z_t = G_t * Y_t
```

The code exposes `--dereverb-strength 0` so the denoising-only condition can be measured against the joint condition.

## 6. Reconstruction

The enhanced spectrum is converted back to a time-domain frame with inverse RFFT. Each frame is windowed and overlap-added. The accumulated window power is used for normalization, so the output length is aligned to the input length after the final flush.

The same stateful path is used for small CLI chunks and for a complete file. This prevents an offline-only implementation from being mistaken for a streaming implementation.

## 7. Pseudocode

```text
initialize frame buffer, overlap-add buffers, noise PSD, history, predictor

for each incoming chunk:
    append chunk to pending samples
    while pending contains one frame:
        frame = first frame from pending
        advance pending by one hop

        X = STFT(frame)
        P = abs(X)^2

        if noise PSD is not ready:
            collect P into bootstrap estimate
        else:
            speech = frame_energy > 2 * noise_energy
            if not speech:
                update noise PSD slowly

        G = clipped spectral suppression gain

        if delayed history exists:
            prediction = predictor * delayed spectrum
            cap prediction magnitude
            Y = X - dereverb_strength * prediction
            update predictor with normalized complex residual
        else:
            Y = X

        enhanced_frame = ISTFT(G * Y)
        overlap-add enhanced_frame
        emit one hop of normalized output

flush the final padded frame and overlap-add tail
trim output to input length
```

## 8. What this algorithm does and does not prove

It establishes a runnable causal baseline with explicit switches for denoising, dereverberation, chunking, and timing. It does not prove that the baseline is better than the five papers, that it improves real speech, or that it is production-ready. Those claims require aligned human-speech recordings, a speaker-disjoint test set, acoustic metrics, ASR WER, and hardware-specific latency measurements.
