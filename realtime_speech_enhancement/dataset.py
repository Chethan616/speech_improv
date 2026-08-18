"""Build a lightweight WAV manifest for collected speech datasets."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import wave


@dataclass(frozen=True)
class WavRecord:
    relative_path: str
    sample_rate_hz: int
    channels: int
    sample_width_bytes: int
    frame_count: int
    duration_seconds: float


def describe_wav(path: Path, root: Path) -> WavRecord:
    with wave.open(str(path), "rb") as handle:
        frame_count = handle.getnframes()
        sample_rate = handle.getframerate()
        return WavRecord(
            relative_path=path.relative_to(root).as_posix(),
            sample_rate_hz=sample_rate,
            channels=handle.getnchannels(),
            sample_width_bytes=handle.getsampwidth(),
            frame_count=frame_count,
            duration_seconds=frame_count / sample_rate if sample_rate else 0.0,
        )


def build_manifest(root: str | Path, dataset_name: str, split: str) -> dict:
    """Scan WAV headers without loading the audio samples into memory."""

    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    records: list[WavRecord] = []
    errors: list[dict[str, str]] = []
    for path in sorted(root.rglob("*.wav")):
        try:
            records.append(describe_wav(path, root))
        except (OSError, wave.Error, ValueError) as error:
            errors.append({"relative_path": path.relative_to(root).as_posix(), "error": str(error)})
    sample_rates = sorted({record.sample_rate_hz for record in records})
    channels = sorted({record.channels for record in records})
    return {
        "dataset_name": dataset_name,
        "split": split,
        "root": str(root),
        "file_count": len(records),
        "total_duration_seconds": sum(record.duration_seconds for record in records),
        "sample_rates_hz": sample_rates,
        "channel_counts": channels,
        "errors": errors,
        "files": [asdict(record) for record in records],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create a manifest for a collected WAV dataset")
    parser.add_argument("root", help="directory containing WAV files")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", required=True, choices=["train", "validation", "test", "demo"])
    parser.add_argument("--output", required=True, help="JSON manifest path")
    args = parser.parse_args(argv)
    manifest = build_manifest(args.root, args.dataset_name, args.split)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("dataset_name", "split", "file_count", "total_duration_seconds", "sample_rates_hz", "errors")}, indent=2))


if __name__ == "__main__":
    main()

