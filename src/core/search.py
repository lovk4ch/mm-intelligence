from pathlib import Path

import numpy as np

from src.core.embeddings import TextEmbedder, get_embeddings_hash, load_or_create_embeddings, VisualEmbedder


class SemanticSearch:
    def __init__(
        self,
        embedder: TextEmbedder,
        chunks: list[dict],
        embeddings: np.ndarray,
    ):
        self.embedder = embedder
        self.chunks = chunks
        self.embeddings = embeddings

    def find(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.4,
    ) -> list[dict]:
        query_embedding = self.embedder.encode([query])[0]

        scores = self.embeddings @ query_embedding
        indices = np.argsort(scores)[::-1]

        results = []

        for i in indices:
            if scores[i] < min_score:
                continue

            results.append({
                "score": float(scores[i]),
                "timestamp": self.chunks[i]["timestamp"],
                "text": self.chunks[i]["text"],
            })

            if len(results) >= top_k:
                break

        return results

class VisualSearch:
    def __init__(
        self,
        embeddings: np.ndarray,
        frames: list[dict],
        embedder: VisualEmbedder,
    ) -> None:
        self.embeddings = embeddings
        self.frames = frames
        self.embedder = embedder

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        query_embedding = self.embedder.encode_text(query)

        scores = self.embeddings @ query_embedding
        indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "timestamp": self.frames[i]["timestamp"],
                "path": self.frames[i]["path"],
                "score": float(scores[i]),
            }
            for i in indices
        ]

def create_semantic_search(
    embedder: TextEmbedder,
    chunks: list[dict],
    embeddings_path: Path,
) -> SemanticSearch:
    texts = [chunk["text"] for chunk in chunks]

    current_hash = get_embeddings_hash(
        model_name=embedder.MODEL_NAME,
        data=chunks,
    )

    embeddings = load_or_create_embeddings(
        embeddings_path,
        current_hash,
        lambda: embedder.encode(texts),
    )

    return SemanticSearch(
        embedder=embedder,
        chunks=chunks,
        embeddings=embeddings,
    )