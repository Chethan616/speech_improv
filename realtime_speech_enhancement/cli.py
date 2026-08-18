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


def _profile_defaults(profile: str, sample_rate: int) -> dict[str, float | int]:
    """Return settings tuned for the input type without hiding the choices."""

    if profile == "strong-denoise":
        frame_size = 2048 if sample_rate >= 32000 else 1024
        return {
            "frame_size": frame_size,
            "hop_size": frame_size // 4,
            "noise_frames": 1,
            "noise_bootstrap_percentile": 35.0,
            "suppression_strength": 1.0,
            "gain_floor": 0.05,
            "dereverb_strength": 0.0,
        }
    if profile == "balanced":
        frame_size = 1024 if sample_rate >= 32000 else 512
        return {
            "frame_size": frame_size,
            "hop_size": frame_size // 4,
            "noise_frames": 8,
            "noise_bootstrap_percentile": 35.0,
            "suppression_strength": 0.90,
            "gain_floor": 0.08,
            "dereverb_strength": 0.10,
        }
    return {
        "frame_size": 512,
        "hop_size": 128,
        "noise_frames": 8,
        "noise_bootstrap_percentile": 35.0,
        "suppression_strength": 0.85,
        "gain_floor": 0.12,
        "dereverb_strength": 0.20,
    }


def _enhancer_config(args: argparse.Namespace, sample_rate: int) -> EnhancerConfig:
    defaults = _profile_defaults(args.profile, sample_rate)

    def value(name: str):
        configured = getattr(args, name)
        return defaults[name] if configured is None else configured

    return EnhancerConfig(
        frame_size=int(value("frame_size")),
        hop_size=int(value("hop_size")),
        noise_frames=int(value("noise_frames")),
        noise_bootstrap_percentile=float(value("noise_bootstrap_percentile")),
        suppression_strength=float(value("suppression_strength")),
        gain_floor=float(value("gain_floor")),
        dereverb_strength=float(value("dereverb_strength")),
        dereverb_delay_frames=int(value("dereverb_delay_frames")),
    )


def _enhance_with_report(
    samples: np.ndarray,
    sample_rate: int,
    config: EnhancerConfig,
    chunk_size: int,
    profile: str,
) -> tuple[np.ndarray, dict]:
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
        "profile": profile,
        "config": {
            "frame_size": config.frame_size,
            "hop_size": config.hop_size,
            "noise_frames": config.noise_frames,
            "noise_bootstrap_percentile": config.noise_bootstrap_percentile,
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
    config = _enhancer_config(args, audio.sample_rate)
    enhanced, report = _enhance_with_report(audio.samples, audio.sample_rate, config, args.chunk_size, args.profile)
    write_wav(args.output, enhanced, audio.sample_rate)
    if args.report:
        _write_json(args.report, report)
    print(json.dumps(report, indent=2, allow_nan=False))


def cmd_enhance_batch(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    report_root = Path(args.report_dir).resolve() if args.report_dir else output_root.parent / f"{output_root.name}_reports"
    if not input_root.is_dir():
        raise ValueError(f"Input directory does not exist: {input_root}")
    if output_root == input_root or input_root in output_root.parents:
        raise ValueError("Output directory must be separate from and outside the input directory")

    input_paths = sorted(input_root.rglob("*.wav"))
    if not input_paths:
        raise ValueError(f"No WAV files found under {input_root}")

    started = time.perf_counter()
    total_audio_seconds = 0.0
    processed = 0
    for input_path in input_paths:
        relative_path = input_path.relative_to(input_root)
        output_path = output_root / relative_path
        report_path = report_root / relative_path.with_suffix(".json")
        audio = read_wav(input_path)
        config = _enhancer_config(args, audio.sample_rate)
        enhanced, report = _enhance_with_report(audio.samples, audio.sample_rate, config, args.chunk_size, args.profile)
        write_wav(output_path, enhanced, audio.sample_rate)
        report.update(
            {
                "input_file": str(input_path),
                "output_file": str(output_path),
                "batch_profile": args.profile,
            }
        )
        _write_json(report_path, report)
        total_audio_seconds += report["input_duration_seconds"]
        processed += 1
        print(f"[{processed}/{len(input_paths)}] {relative_path}")

    elapsed = time.perf_counter() - started
    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "report_root": str(report_root),
        "profile": args.profile,
        "file_count": processed,
        "audio_duration_seconds": total_audio_seconds,
        "batch_processing_seconds": elapsed,
        "batch_real_time_factor": elapsed / total_audio_seconds if total_audio_seconds else None,
        "note": "This batch command enhances files independently; it does not calculate quality metrics without aligned clean references.",
    }
    _write_json(report_root / "batch_summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False))


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


def cmd_evaluate_batch(args: argparse.Namespace) -> None:
    clean_root = Path(args.clean_root).resolve()
    degraded_root = Path(args.degraded_root).resolve()
    enhanced_root = Path(args.enhanced_root).resolve()
    if not clean_root.is_dir() or not degraded_root.is_dir() or not enhanced_root.is_dir():
        raise ValueError("Clean, degraded, and enhanced roots must all be existing directories")

    totals = {
        "snr_input_db": 0.0,
        "snr_output_db": 0.0,
        "snr_improvement_db": 0.0,
        "si_sdr_input_db": 0.0,
        "si_sdr_output_db": 0.0,
        "si_sdr_improvement_db": 0.0,
    }
    evaluated = 0
    skipped: list[str] = []
    for degraded_path in sorted(degraded_root.rglob("*.wav")):
        relative_path = degraded_path.relative_to(degraded_root)
        clean_path = clean_root / relative_path
        enhanced_path = enhanced_root / relative_path
        if not clean_path.exists() or not enhanced_path.exists():
            skipped.append(str(relative_path))
            continue
        clean = read_wav(clean_path)
        degraded = read_wav(degraded_path)
        enhanced = read_wav(enhanced_path)
        if not (clean.sample_rate == degraded.sample_rate == enhanced.sample_rate):
            raise ValueError(f"Sample-rate mismatch for {relative_path}")
        report = evaluate_quality(clean.samples, degraded.samples, enhanced.samples)
        for key in totals:
            totals[key] += float(report[key])
        evaluated += 1

    if evaluated == 0:
        raise ValueError("No matching clean, degraded, and enhanced WAV triplets were found")
    summary = {
        "clean_root": str(clean_root),
        "degraded_root": str(degraded_root),
        "enhanced_root": str(enhanced_root),
        "file_count": evaluated,
        "skipped_count": len(skipped),
        "skipped_files": skipped,
        **{f"mean_{key}": value / evaluated for key, value in totals.items()},
        "warning": "These means describe only the aligned WAV triplets found in these directories.",
    }
    if args.report:
        _write_json(args.report, summary)
    print(json.dumps(summary, indent=2, allow_nan=False))


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
    config = _enhancer_config(args, audio.sample_rate)
    enhanced, timing = _enhance_with_report(audio.samples, audio.sample_rate, config, args.chunk_size, args.profile)
    enhanced_path = out_dir / "enhanced_fixture.wav"
    write_wav(enhanced_path, enhanced, audio.sample_rate)
    clean = read_wav(out_dir / "clean_fixture.wav")
    quality = evaluate_quality(clean.samples, audio.samples, enhanced)
    report = {"timing": timing, "quality": quality, "synthetic_fixture": True}
    _write_json(out_dir / "evaluation.json", report)
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"Enhanced fixture: {enhanced_path.resolve()}")


def _add_algorithm_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        choices=("low-latency", "balanced", "strong-denoise"),
        default="low-latency",
        help="choose defaults for low latency, balanced enhancement, or noisy speech test sets",
    )
    parser.add_argument("--frame-size", type=int, default=None)
    parser.add_argument("--hop-size", type=int, default=None)
    parser.add_argument("--noise-frames", type=int, default=None)
    parser.add_argument("--noise-bootstrap-percentile", type=float, default=None)
    parser.add_argument("--suppression-strength", type=float, default=None)
    parser.add_argument("--gain-floor", type=float, default=None)
    parser.add_argument("--dereverb-strength", type=float, default=None)
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

    batch = subparsers.add_parser("enhance-batch", help="enhance every WAV under a directory")
    batch.add_argument("input_root")
    batch.add_argument("--output-root", required=True)
    batch.add_argument("--report-dir")
    _add_algorithm_options(batch)
    batch.set_defaults(function=cmd_enhance_batch)

    evaluate = subparsers.add_parser("evaluate", help="compute aligned before/after objective metrics")
    evaluate.add_argument("--clean", required=True)
    evaluate.add_argument("--degraded", required=True)
    evaluate.add_argument("--enhanced", required=True)
    evaluate.add_argument("--report")
    evaluate.set_defaults(function=cmd_evaluate)

    evaluate_batch = subparsers.add_parser(
        "evaluate-batch", help="evaluate matching clean, degraded, and enhanced WAV folders"
    )
    evaluate_batch.add_argument("--clean-root", required=True)
    evaluate_batch.add_argument("--degraded-root", required=True)
    evaluate_batch.add_argument("--enhanced-root", required=True)
    evaluate_batch.add_argument("--report")
    evaluate_batch.set_defaults(function=cmd_evaluate_batch)

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
