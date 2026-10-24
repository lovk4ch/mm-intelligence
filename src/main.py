from PIL import Image

from src.core.context import ContextBuilder
from src.core.transcription import transcribe_audio, build_chunks
from src.core.config import VIDEO_PATH, AUDIO_PATH, TRANSCRIPT_PATH, TEXT_EMBEDDINGS_PATH, FRAMES_DIR, \
    VISUAL_EMBEDDINGS_PATH
from src.core.embeddings import TextEmbedder, VisualEmbedder
from src.core.frame_extractor import extract_frames
from src.core.search import MultimodalSearch, create_semantic_search, create_visual_search
from src.core.video_qa import VideoQA
from src.utils import save_json, load_json


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

    query = "Что автор говорит про музей?"

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

    context_builder = ContextBuilder()
    context = context_builder.build(results)

    video_qa = VideoQA()
    answer = video_qa.answer(
        query,
        context,
    )

    print("\n[ANSWER]")
    print(answer)


if __name__ == "__main__":
    main()