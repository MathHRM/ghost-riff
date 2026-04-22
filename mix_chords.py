#!/usr/bin/env python3
import argparse
import base64
import json
import shutil
import sys
from io import BytesIO
from pathlib import Path

from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError


def load_mp3(mp3_bytes: bytes) -> AudioSegment:
    return AudioSegment.from_file(BytesIO(mp3_bytes), format="mp3")


def mix_with_strum(segments: list, delay_ms: int = 0) -> AudioSegment:
    total_dur = max(len(s) for s in segments) + delay_ms * len(segments)
    output = AudioSegment.silent(duration=total_dur)
    for i, seg in enumerate(segments):
        output = output.overlay(seg, position=i * delay_ms)
    return output - (len(segments) * 2)


def process(json_path: Path, out_dir: Path, strum_delay: int = 0) -> None:
    try:
        data = json.loads(json_path.read_text())
    except json.JSONDecodeError as e:
        print(f"  ERROR: invalid JSON — {e}", file=sys.stderr)
        return

    if not data:
        print("  SKIP: empty JSON")
        return

    segments = []
    for note_name, b64_string in data.items():
        try:
            mp3_bytes = base64.b64decode(b64_string)
            segments.append(load_mp3(mp3_bytes))
        except (CouldntDecodeError, Exception) as e:
            print(f"  WARN: skip '{note_name}' — {e}", file=sys.stderr)

    if not segments:
        print("  SKIP: no valid segments", file=sys.stderr)
        return

    mixed = mix_with_strum(segments, delay_ms=strum_delay)
    out = out_dir / json_path.with_suffix(".wav").name
    mixed.export(str(out), format="wav")
    print(f"  -> {out.name} ({len(segments)} notes, {len(mixed)}ms)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode base64 MP3s from JSON files and mix into chord WAVs."
    )
    parser.add_argument("--folder", default="sounds/base64-chords",
                        help="Folder containing .json files (default: sounds/base64-chords)")
    parser.add_argument("--output", default="sounds/chords",
                        help="Output folder for WAV files (default: sounds/chords)")
    parser.add_argument("--strum", type=int, default=27,
                        help="Delay between notes in ms (0=dry chord, 20-40=strum)")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"ERROR: '{folder}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    if shutil.which("ffmpeg") is None:
        print(
            "ERROR: ffmpeg not found. Install: sudo apt install ffmpeg",
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(folder.glob("*.json"))
    if not json_files:
        print(f"No .json files found in '{folder}'.")
        sys.exit(0)

    for json_path in json_files:
        print(f"Processing {json_path.name}...")
        process(json_path, out_dir, strum_delay=args.strum)


if __name__ == "__main__":
    main()
