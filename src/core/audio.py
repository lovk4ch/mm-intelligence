import subprocess
from pathlib import Path

from src.utils import require_file


def extract_audio(video_path: Path, audio_path: Path) -> None:
    require_file(video_path)

    if audio_path.exists():
        print(f"Audio already exists: {audio_path}")
        return

    audio_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        str(audio_path),
    ]

    subprocess.run(command, check=True)