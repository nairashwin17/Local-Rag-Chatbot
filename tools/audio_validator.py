"""Audio-file checks, ready for Localization QA."""

from __future__ import annotations

from pathlib import Path


def validate_audio_files(directory: Path) -> list[str]:
    if not directory.exists():
        return [f"Missing directory: {directory}"]
    return [f"Empty audio file: {path.name}" for path in directory.glob("*") if path.suffix.lower() in {".mp3", ".wav", ".m4a"} and path.stat().st_size == 0]
