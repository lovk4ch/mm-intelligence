from pathlib import Path
import subprocess

import numpy as np
import torch
from PIL import Image
from transformers import pipeline

from src.core.config import VIDEO_PATH, AUDIO_PATH, TRANSCRIPT_PATH, TEXT_EMBEDDINGS_PATH, FRAMES_DIR, VISUAL_EMBEDDINGS_PATH
from src.core.embeddings import TextEmbedder, VisualEmbedder, create_visual_embeddings
from src.core.frame_extractor import extract_frames
from src.core.search import SemanticSearch, VisualSearch
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

def test_siglip_search(
    visual_embedder: VisualEmbedder,
    frames: list[dict],
    query: str,
    batch_size: int = 8,
    top_k: int = 5,
) -> None:
    results = []

    for start in range(0, len(frames), batch_size):
        batch_frames = frames[start:start + batch_size]

        images = [
            Image.open(frame["path"]).convert("RGB")
            for frame in batch_frames
        ]

        inputs = visual_embedder.processor(
            text=[query],
            images=images,
            padding="max_length",
            return_tensors="pt",
        ).to(visual_embedder.device)

        with torch.inference_mode():
            outputs = visual_embedder.model(**inputs)

        scores = outputs.logits_per_text[0].cpu().numpy()

        for frame, score in zip(batch_frames, scores):
            results.append({
                "score": float(score),
                "timestamp": frame["timestamp"],
                "path": frame["path"],
            })

    results.sort(
        key=lambda result: result["score"],
        reverse=True,
    )

    for result in results[:top_k]:
        print(
            result["score"],
            result["timestamp"],
            result["path"],
        )

        Image.open(result["path"]).show()

def main():
    """
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

    search_engine = create_semantic_search(
        embedder,
        chunks,
        EMBEDDINGS_PATH,
    )

    results = search_engine.find(
        "Где говорят про музей?"
    )

    if results:
        for result in results:
            print(result)
    else:
        print("No results")
    """

    frames = extract_frames(
        VIDEO_PATH,
        FRAMES_DIR,
        interval_s=4.0,
    )

    visual_embedder = VisualEmbedder()

    visual_embeddings = create_visual_embeddings(
        visual_embedder,
        frames,
        VISUAL_EMBEDDINGS_PATH,
    )

    visual_search = VisualSearch(
        visual_embeddings,
        frames,
        visual_embedder,
    )

    results = visual_search.search(
        "skyscraper",
        top_k=5,
    )

    for result in results:
        print(result)
        Image.open(result["path"]).show()


if __name__ == "__main__":
    main()