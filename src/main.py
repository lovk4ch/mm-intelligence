from pathlib import Path
import subprocess

from src.utils import require_file


def extract_audio(video_path: Path, audio_path: Path) -> None:
    require_file(video_path)
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

def main():
    video_path = Path("data/input/video.mp4")
    audio_path = Path("data/output/audio.wav")

    audio_path.parent.mkdir(parents=True, exist_ok=True)

    extract_audio(video_path, audio_path)

if __name__ == "__main__":
    main()