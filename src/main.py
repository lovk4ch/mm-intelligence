from pathlib import Path
import subprocess
from transformers import pipeline

from src.core.config import VIDEO_PATH, AUDIO_PATH, TRANSCRIPT_PATH, EMBEDDINGS_PATH
from src.core.embeddings import TextEmbedder
from src.core.search import SemanticSearch
from src.utils import require_file, save_json, load_json


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

def transcribe_audio(video_path: Path, audio_path: Path):
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    extract_audio(video_path, audio_path)

    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        chunk_length_s=30,
    )

    result = asr(
        str(audio_path),
        return_timestamps="word",
        generate_kwargs={
            "task": "transcribe"
        },
    )
    return result

def build_chunks(
    words: list[dict],
    min_duration: float = 15.0,
    max_duration: float = 30.0,
) -> list[dict]:
    chunks = []
    current_words = []
    start = None
    end = None

    for word in words:
        word_start, word_end = word["timestamp"]

        if start is None:
            start = word_start

        current_words.append(word["text"])
        end = word_end

        duration = end - start
        text = "".join(current_words).strip()

        sentence_ended = text.endswith((".", "!", "?"))

        if (
            duration >= min_duration
            and sentence_ended
        ) or duration >= max_duration:
            chunks.append({
                "timestamp": [start, end],
                "text": text,
            })

            current_words = []
            start = None
            end = None

    if current_words:
        chunks.append({
            "timestamp": [start, end],
            "text": "".join(current_words).strip(),
        })

    return chunks

def main():
    if TRANSCRIPT_PATH.is_file():
        transcript = load_json(Path(TRANSCRIPT_PATH))
    else:
        transcript = transcribe_audio(
            VIDEO_PATH,
            AUDIO_PATH
        )
        save_json(
            transcript,
            Path(TRANSCRIPT_PATH)
        )

    words = transcript["chunks"]
    chunks = build_chunks(words, max_duration=30)

    print("Loading embedding model...")

    embedder = TextEmbedder()

    print("Model loaded")
    print("Encoding...")

    search_engine = SemanticSearch(
        embedder=embedder,
        chunks=chunks,
        embeddings_path=EMBEDDINGS_PATH,
    )

    results = search_engine.find("Где говорят про музей?")

    if results:
        for result in results:
            print(result)
    else:
        print("No results")

if __name__ == "__main__":
    main()