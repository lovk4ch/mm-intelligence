import numpy as np
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self) -> None:
        self.model = SentenceTransformer(self.MODEL_NAME)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
        )