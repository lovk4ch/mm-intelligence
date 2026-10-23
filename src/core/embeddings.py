import hashlib
import json
from typing import Callable

import torch
import numpy as np

from pathlib import Path
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModel


class TextEmbedder:
    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self) -> None:
        self.model = SentenceTransformer(self.MODEL_NAME)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
        )

class VisualEmbedder:
    MODEL_NAME = "google/siglip-base-patch16-224"

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = AutoProcessor.from_pretrained(
            self.MODEL_NAME
        )

        self.model = AutoModel.from_pretrained(
            self.MODEL_NAME
        ).to(self.device)

        self.model.eval()

    def encode_text(
        self,
        text: str,
    ) -> np.ndarray:
        inputs = self.processor(
            text=[text],
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            output = self.model.get_text_features(**inputs)
            embedding = output.pooler_output

        embedding = torch.nn.functional.normalize(
            embedding,
            dim=-1,
        )

        return embedding[0].cpu().numpy()

    def encode_images(
        self,
        image_paths: list,
    ) -> np.ndarray:
        images = [
            Image.open(path).convert("RGB")
            for path in image_paths
        ]

        inputs = self.processor(
            images=images,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            output = self.model.get_image_features(**inputs)
            embeddings = output.pooler_output

        embeddings = torch.nn.functional.normalize(
            embeddings,
            dim=-1,
        )

        return embeddings.cpu().numpy()

    def encode_images_batched(
        self,
        image_paths: list[Path],
        batch_size: int = 8,
    ) -> np.ndarray:
        all_embeddings = []

        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]

            embeddings = self.encode_images(batch_paths)
            all_embeddings.append(embeddings)

        return np.concatenate(all_embeddings, axis=0)


def get_embeddings_hash(
    model_name: str,
    data: object,
) -> str:
    payload = {
        "model": model_name,
        "data": data,
    }

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


def create_visual_embeddings(
    visual_embedder: VisualEmbedder,
    frames: list[dict],
    embeddings_path: Path,
    batch_size: int = 8,
) -> np.ndarray:
    image_paths = [
        frame["path"]
        for frame in frames
    ]

    current_hash = get_embeddings_hash(
        visual_embedder.MODEL_NAME,
        frames,
    )

    return load_or_create_embeddings(
        embeddings_path,
        current_hash,
        lambda: visual_embedder.encode_images_batched(
            image_paths,
            batch_size=batch_size,
        ),
    )


def load_or_create_embeddings(
    embeddings_path: Path,
    current_hash: str,
    create_embeddings: Callable[[], np.ndarray],
) -> np.ndarray:
    hash_path = embeddings_path.with_suffix(".hash")
    print(f"Current embeddings hash for {str(embeddings_path)}"
          f": {current_hash}")

    if (
        embeddings_path.is_file()
        and hash_path.is_file()
        and hash_path.read_text(encoding="utf-8") == current_hash
    ):
        return np.load(embeddings_path)

    embeddings = create_embeddings()

    embeddings_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(embeddings_path, embeddings)

    hash_path.write_text(
        current_hash,
        encoding="utf-8",
    )

    return embeddings