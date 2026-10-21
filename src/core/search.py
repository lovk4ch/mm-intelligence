import hashlib
import json
from pathlib import Path

import numpy as np

from src.core.embeddings import TextEmbedder


class SemanticSearch:
    def __init__(
        self,
        embedder: TextEmbedder,
        chunks: list[dict],
        embeddings_path: Path,
    ):
        self.embedder = embedder
        self.chunks = chunks
        self.embeddings_path = embeddings_path
        self.hash_path = embeddings_path.with_suffix(".hash")

        self.embeddings = self._load_or_create_embeddings()

    def _get_chunks_hash(self) -> str:
        data = {
            "model": self.embedder.MODEL_NAME,
            "chunks": self.chunks,
        }

        serialized = json.dumps(
            data,
            ensure_ascii=False,
            sort_keys=True,
        )

        return hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

    def _load_or_create_embeddings(self) -> np.ndarray:
        current_hash = self._get_chunks_hash()
        print(f"Current embeddings hash: {current_hash}")

        if (
            self.embeddings_path.is_file()
            and self.hash_path.is_file()
            and self.hash_path.read_text(
                encoding="utf-8"
            ) == current_hash
        ):
            return np.load(self.embeddings_path)

        texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        embeddings = self.embedder.encode(texts)

        self.embeddings_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(self.embeddings_path, embeddings)
        self.hash_path.write_text(
            current_hash,
            encoding="utf-8"
        )

        return embeddings

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