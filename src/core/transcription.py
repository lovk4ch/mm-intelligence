from pathlib import Path

from transformers import pipeline

from src.core.audio import extract_audio


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
