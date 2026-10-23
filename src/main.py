import subprocess
from pathlib import Path

import torch
from PIL import Image
from transformers import pipeline

from src.core.config import VIDEO_PATH, AUDIO_PATH, TRANSCRIPT_PATH, TEXT_EMBEDDINGS_PATH, FRAMES_DIR, \
    VISUAL_EMBEDDINGS_PATH
from src.core.embeddings import TextEmbedder, VisualEmbedder
from src.core.frame_extractor import extract_frames
from src.core.search import MultimodalSearch, create_semantic_search, create_visual_search
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
    # --- Text ---
    print("[TEXT] Preparing transcript...")

    if TRANSCRIPT_PATH.is_file():
        transcript = load_json(TRANSCRIPT_PATH)
    else:
        transcript = transcribe_audio(
            VIDEO_PATH,
            AUDIO_PATH,
        )
        save_json(
            transcript,
            TRANSCRIPT_PATH,
        )

    chunks = build_chunks(
        transcript["chunks"],
        max_duration=30,
    )

    print("[TEXT] Loading model...")
    text_embedder = TextEmbedder()

    print("[TEXT] Preparing search...")
    semantic_search = create_semantic_search(
        text_embedder,
        chunks,
        TEXT_EMBEDDINGS_PATH,
    )

    print(f"[TEXT] Ready: {len(chunks)} chunks")

    # --- Visual ---
    print("[VISUAL] Preparing frames...")

    frames = extract_frames(
        VIDEO_PATH,
        FRAMES_DIR,
        interval_s=4.0,
    )

    print("[VISUAL] Loading model...")
    visual_embedder = VisualEmbedder()

    print("[VISUAL] Preparing search...")
    visual_search = create_visual_search(
        visual_embedder,
        frames,
        VISUAL_EMBEDDINGS_PATH,
    )

    print(f"[VISUAL] Ready: {len(frames)} frames")

    # --- Multimodal ---
    print("[MULTIMODAL] Initializing...")

    multimodal_search = MultimodalSearch(
        semantic_search,
        visual_search,
    )

    query = "anime toys"

    print(f'[MULTIMODAL] Query: "{query}"')

    results = multimodal_search.find(
        query,
        top_k=5,
    )

    print("\n[MULTIMODAL]")
    for result in results:
        print(result)

        if result["frame_path"] is not None:
            Image.open(result["frame_path"]).show()


if __name__ == "__main__":
    main()