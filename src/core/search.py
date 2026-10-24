from pathlib import Path

import numpy as np

from src.core.embeddings import TextEmbedder, get_embeddings_hash, load_or_create_embeddings, VisualEmbedder, \
    create_visual_embeddings


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
        embedder: VisualEmbedder,
        frames: list[dict],
        embeddings: np.ndarray,
    ) -> None:
        self.embedder = embedder
        self.frames = frames
        self.embeddings = embeddings

    def find(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
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
            if scores[i] >= min_score
        ]


class MultimodalSearch:
    def __init__(
        self,
        semantic_search: SemanticSearch,
        visual_search: VisualSearch,
    ) -> None:
        self.semantic_search = semantic_search
        self.visual_search = visual_search

    def find(
        self,
        query: str,
        top_k: int = 5,
        time_tolerance: float = 5.0,
        multimodal_bonus: float = 0.1,
    ) -> list[dict]:
        semantic_results = self.semantic_search.find(
            query,
            top_k=top_k,
        )

        visual_results = self.visual_search.find(
            query,
            top_k=top_k,
        )

        semantic_scores = normalize_scores(
            semantic_results,
        )

        visual_scores = normalize_scores(
            visual_results,
        )

        matches = []

        used_semantic = set()
        used_visual = set()

        for semantic_index, semantic in enumerate(semantic_results):
            start, end = semantic["timestamp"]

            nearby_visuals = []

            for visual_index, visual in enumerate(visual_results):
                if visual_index in used_visual:
                    continue

                visual_time = visual["timestamp"]

                if (
                    start - time_tolerance
                    <= visual_time
                    <= end + time_tolerance
                ):
                    nearby_visuals.append(
                        (visual_index, visual)
                    )

            if not nearby_visuals:
                continue

            _, best_visual = max(
                nearby_visuals,
                key=lambda item: item[1]["score"],
            )

            visual_index, best_visual = max(
                nearby_visuals,
                key=lambda item: item[1]["score"],
            )

            normalized_semantic = semantic_scores[semantic_index]
            normalized_visual = visual_scores[visual_index]

            multimodal_score = (
                0.5 * normalized_semantic
                + 0.5 * normalized_visual
                + multimodal_bonus
            )

            matches.append({
                "timestamp": best_visual["timestamp"],
                "text": semantic["text"],
                "frame_path": best_visual["path"],
                "semantic_score": semantic["score"],
                "visual_score": best_visual["score"],
                "score": multimodal_score,
                "modalities": ["text", "visual"],
            })

            used_semantic.add(semantic_index)

            for index, _ in nearby_visuals:
                used_visual.add(index)

        for i, semantic in enumerate(semantic_results):
            if i in used_semantic:
                continue

            matches.append({
                "timestamp": semantic["timestamp"],
                "text": semantic["text"],
                "frame_path": None,
                "semantic_score": semantic["score"],
                "visual_score": None,
                "score": semantic_scores[i],
                "modalities": ["text"],
            })

        for i, visual in enumerate(visual_results):
            if i in used_visual:
                continue

            matches.append({
                "timestamp": visual["timestamp"],
                "text": None,
                "frame_path": visual["path"],
                "semantic_score": None,
                "visual_score": visual["score"],
                "score": visual_scores[i],
                "modalities": ["visual"],
            })

        matches.sort(
            key=lambda result: result["score"],
            reverse=True,
        )

        return matches[:top_k]


def create_semantic_search(
    embedder: TextEmbedder,
    chunks: list[dict],
    embeddings_path: Path,
) -> SemanticSearch:
    current_hash = get_embeddings_hash(
        embedder.MODEL_NAME,
        chunks,
    )

    embeddings = load_or_create_embeddings(
        embeddings_path,
        current_hash,
        lambda: embedder.encode(
            [chunk["text"] for chunk in chunks]
        ),
    )

    return SemanticSearch(
        embedder,
        chunks,
        embeddings,
    )


def create_visual_search(
    embedder: VisualEmbedder,
    frames: list[dict],
    embeddings_path: Path,
) -> VisualSearch:
    embeddings = create_visual_embeddings(
        embedder,
        frames,
        embeddings_path,
    )

    return VisualSearch(
        embedder,
        frames,
        embeddings,
    )


def normalize_scores(
    results: list[dict],
) -> dict[int, float]:
    if not results:
        return {}

    scores = [result["score"] for result in results]

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return {
            i: 1.0
            for i in range(len(results))
        }

    return {
        i: (result["score"] - min_score)
        / (max_score - min_score)
        for i, result in enumerate(results)
    }