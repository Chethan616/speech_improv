"""Command-line entry points for the Review 1 experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .audio import read_wav, write_wav
from .degradation import degrade_audio
from .enhancer import EnhancerConfig, enhance_audio
from .metrics import evaluate_quality, real_time_factor


def _write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _synthetic_voiced_fixture(sample_rate: int, duration: float, seed: int = 11) -> np.ndarray:
    """Make a deterministic voiced-like test signal, not a speech corpus sample."""

    if sample_rate <= 0 or duration <= 0:
        raise ValueError("sample_rate and duration must be positive")
    rng = np.random.default_rng(seed)
    length = int(round(sample_rate * duration))
    time_axis = np.arange(length, dtype=np.float64) / sample_rate
    signal = np.zeros(length, dtype=np.float64)
    segment_count = max(1, int(duration * 2.5))
    segment_length = max(1, length // segment_count)
    for segment in range(segment_count):
        start = segment * segment_length
        end = min(length, (segment + 1) * segment_length)
        if end <= start:
            continue
        fundamental = 105.0 + 22.0 * np.sin(0.7 * segment)
        local_time = time_axis[start:end] - time_axis[start]
        phase = 2.0 * np.pi * (fundamental + 8.0 * local_time) * local_time
        envelope = np.sin(np.linspace(0.0, np.pi, end - start)) ** 1.3
        voiced = sum((1.0 / harmonic) * np.sin((harmonic * phase) + 0.1 * harmonic) for harmonic in range(1, 7))
        signal[start:end] += 0.12 * envelope * voiced
    signal += 0.004 * rng.standard_normal(length)
    peak = max(float(np.max(np.abs(signal))), 1e-12)
    return (0.72 * signal / peak).astype(np.float32)


def _enhancer_config(args: argparse.Namespace) -> EnhancerConfig:
    return EnhancerConfig(
        frame_size=args.frame_size,
        hop_size=args.hop_size,
        noise_frames=args.noise_frames,
        suppression_strength=args.suppression_strength,
        gain_floor=args.gain_floor,
        dereverb_strength=args.dereverb_strength,
        dereverb_delay_frames=args.dereverb_delay_frames,
    )


def _enhance_with_report(samples: np.ndarray, sample_rate: int, config: EnhancerConfig, chunk_size: int) -> tuple[np.ndarray, dict]:
    started = time.perf_counter()
    enhanced = enhance_audio(samples, config=config, chunk_size=chunk_size)
    elapsed = time.perf_counter() - started
    duration = samples.size / sample_rate if sample_rate else 0.0
    report = {
        "sample_rate_hz": sample_rate,
        "input_samples": int(samples.size),
        "input_duration_seconds": duration,
        "processing_seconds": elapsed,
        "real_time_factor": real_time_factor(elapsed, duration),
        "algorithmic_latency_seconds": config.frame_size / sample_rate,
        "chunk_size_samples": chunk_size,
        "config": {
            "frame_size": config.frame_size,
            "hop_size": config.hop_size,
            "noise_frames": config.noise_frames,
            "suppression_strength": config.suppression_strength,
            "gain_floor": config.gain_floor,
            "dereverb_strength": config.dereverb_strength,
            "dereverb_delay_frames": config.dereverb_delay_frames,
        },
        "notes": [
            "RTF is measured on the machine that ran this command.",
            "The dereverberation path is a one-delay causal predictor experiment, not full batch WPE.",
            "Run a separate ASR command to report WER; this project does not invent ASR results.",
        ],
    }
    return enhanced, report


def cmd_generate(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    clean = _synthetic_voiced_fixture(args.sample_rate, args.duration, seed=args.seed)
    rng = np.random.default_rng(args.seed + 1)
    noise = rng.standard_normal(clean.size).astype(np.float32)
    pair = degrade_audio(
        clean,
        noise,
        args.sample_rate,
        snr_db=args.snr_db,
        rt60_seconds=args.rt60,
        seed=args.seed,
    )
    write_wav(out_dir / "clean_fixture.wav", pair.clean, args.sample_rate)
    write_wav(out_dir / "reverberant_fixture.wav", pair.reverberant, args.sample_rate)
    write_wav(out_dir / "noisy_reverberant_fixture.wav", pair.noisy_reverberant, args.sample_rate)
    write_wav(out_dir / "room_impulse_response.wav", pair.room_impulse_response, args.sample_rate)
    _write_json(
        out_dir / "generation.json",
        {
            "synthetic_fixture": True,
            "warning": "This is a deterministic voiced-like smoke test, not a human speech result.",
            "sample_rate_hz": args.sample_rate,
            "duration_seconds": args.duration,
            "snr_db": args.snr_db,
            "rt60_seconds": args.rt60,
            "seed": args.seed,
        },
    )
    print(f"Generated Review 1 fixture files in {out_dir.resolve()}")


def cmd_enhance(args: argparse.Namespace) -> None:
    audio = read_wav(args.input)
    config = _enhancer_config(args)
    enhanced, report = _enhance_with_report(audio.samples, audio.sample_rate, config, args.chunk_size)
    write_wav(args.output, enhanced, audio.sample_rate)
    if args.report:
        _write_json(args.report, report)
    print(json.dumps(report, indent=2, allow_nan=False))


def cmd_evaluate(args: argparse.Namespace) -> None:
    clean = read_wav(args.clean)
    degraded = read_wav(args.degraded)
    enhanced = read_wav(args.enhanced)
    if not (clean.sample_rate == degraded.sample_rate == enhanced.sample_rate):
        raise ValueError("All evaluation WAV files must use the same sample rate")
    report = evaluate_quality(clean.samples, degraded.samples, enhanced.samples)
    report.update(
        {
            "sample_rate_hz": clean.sample_rate,
            "clean_file": str(Path(args.clean).resolve()),
            "degraded_file": str(Path(args.degraded).resolve()),
            "enhanced_file": str(Path(args.enhanced).resolve()),
            "warning": "Metrics are only evidence for these aligned WAV files; they are not general claims.",
        }
    )
    if args.report:
        _write_json(args.report, report)
    print(json.dumps(report, indent=2, allow_nan=False))


def cmd_demo(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    generate_args = argparse.Namespace(
        out_dir=out_dir,
        sample_rate=args.sample_rate,
        duration=args.duration,
        snr_db=args.snr_db,
        rt60=args.rt60,
        seed=args.seed,
    )
    cmd_generate(generate_args)
    input_path = out_dir / "noisy_reverberant_fixture.wav"
    audio = read_wav(input_path)
    config = _enhancer_config(args)
    enhanced, timing = _enhance_with_report(audio.samples, audio.sample_rate, config, args.chunk_size)
    enhanced_path = out_dir / "enhanced_fixture.wav"
    write_wav(enhanced_path, enhanced, audio.sample_rate)
    clean = read_wav(out_dir / "clean_fixture.wav")
    quality = evaluate_quality(clean.samples, audio.samples, enhanced)
    report = {"timing": timing, "quality": quality, "synthetic_fixture": True}
    _write_json(out_dir / "evaluation.json", report)
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"Enhanced fixture: {enhanced_path.resolve()}")


def _add_algorithm_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--frame-size", type=int, default=512)
    parser.add_argument("--hop-size", type=int, default=128)
    parser.add_argument("--noise-frames", type=int, default=8)
    parser.add_argument("--suppression-strength", type=float, default=0.85)
    parser.add_argument("--gain-floor", type=float, default=0.12)
    parser.add_argument("--dereverb-strength", type=float, default=0.20)
    parser.add_argument("--dereverb-delay-frames", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=256)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review 1 streaming speech enhancement experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate a controlled synthetic degradation fixture")
    generate.add_argument("--out-dir", default="artifacts/realtime_speech_enhancement_fixture")
    generate.add_argument("--sample-rate", type=int, default=16000)
    generate.add_argument("--duration", type=float, default=5.0)
    generate.add_argument("--snr-db", type=float, default=5.0)
    generate.add_argument("--rt60", type=float, default=0.35)
    generate.add_argument("--seed", type=int, default=11)
    generate.set_defaults(function=cmd_generate)

    enhance = subparsers.add_parser("enhance", help="enhance a mono or multi-channel PCM WAV")
    enhance.add_argument("input")
    enhance.add_argument("--output", required=True)
    enhance.add_argument("--report")
    _add_algorithm_options(enhance)
    enhance.set_defaults(function=cmd_enhance)

    evaluate = subparsers.add_parser("evaluate", help="compute aligned before/after objective metrics")
    evaluate.add_argument("--clean", required=True)
    evaluate.add_argument("--degraded", required=True)
    evaluate.add_argument("--enhanced", required=True)
    evaluate.add_argument("--report")
    evaluate.set_defaults(function=cmd_evaluate)

    demo = subparsers.add_parser("demo", help="generate, enhance, and evaluate a synthetic smoke-test fixture")
    demo.add_argument("--out-dir", default="artifacts/realtime_speech_enhancement_demo")
    demo.add_argument("--sample-rate", type=int, default=16000)
    demo.add_argument("--duration", type=float, default=5.0)
    demo.add_argument("--snr-db", type=float, default=5.0)
    demo.add_argument("--rt60", type=float, default=0.35)
    demo.add_argument("--seed", type=int, default=11)
    _add_algorithm_options(demo)
    demo.set_defaults(function=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.function(args)
