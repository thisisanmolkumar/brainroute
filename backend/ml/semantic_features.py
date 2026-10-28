from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceEmbeddingExtractor:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        revision: str | None = None,
        device: str = "cpu",
        model=None,
    ):
        self.model_name = model_name
        self.revision = revision

        self.model = (
            model
            if model is not None
            else SentenceTransformer(
                model_name,
                revision=revision,
                device=device,
                trust_remote_code=False,
            )
        )

        self.dimension = int(self.model.get_sentence_embedding_dimension())

        if self.dimension <= 0:
            raise ValueError("Invalid embedding dimension")

    def encode(
        self,
        prompts: list[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        if not prompts:
            return np.empty(
                (0, self.dimension),
                dtype=np.float32,
            )

        if any(not prompt.strip() for prompt in prompts):
            raise ValueError("Cannot encode empty prompts")

        unique_prompts = list(dict.fromkeys(prompts))

        vectors = np.asarray(
            self.model.encode(
                unique_prompts,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

        expected_shape = (
            len(unique_prompts),
            self.dimension,
        )

        if vectors.shape != expected_shape:
            raise ValueError(
                f"Expected embedding shape {expected_shape}, received {vectors.shape}"
            )

        if not np.isfinite(vectors).all():
            raise ValueError("Nonfinite embedding values")

        lookup = {prompt: index for index, prompt in enumerate(unique_prompts)}

        indices = [lookup[prompt] for prompt in prompts]

        return vectors[indices]


@dataclass
class PromptPCA:
    mean: np.ndarray
    components: np.ndarray

    @classmethod
    def fit(
        cls,
        vectors: np.ndarray,
        dimensions: int = 16,
    ) -> "PromptPCA":
        vectors = np.asarray(
            vectors,
            dtype=np.float32,
        )

        if vectors.ndim != 2 or len(vectors) < 2:
            raise ValueError("PCA requires at least two training prompts")

        if dimensions < 1:
            raise ValueError("PCA dimensions must be positive")

        actual_dimensions = min(
            dimensions,
            len(vectors) - 1,
            vectors.shape[1],
        )

        center = vectors.mean(
            axis=0,
            dtype=np.float64,
        )

        centered = vectors.astype(np.float64) - center

        _, _, vt = np.linalg.svd(
            centered,
            full_matrices=False,
        )

        return cls(
            mean=center.astype(np.float32),
            components=vt[:actual_dimensions].astype(np.float32),
        )

    def transform(
        self,
        vectors: np.ndarray,
    ) -> np.ndarray:
        vectors = np.asarray(
            vectors,
            dtype=np.float32,
        )

        if vectors.ndim != 2 or vectors.shape[1] != len(self.mean):
            raise ValueError("PCA input dimension mismatch")

        projected = (vectors - self.mean) @ self.components.T

        return projected.astype(np.float32)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            mean=self.mean,
            components=self.components,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "PromptPCA":
        with np.load(path, allow_pickle=False) as data:
            mean = data["mean"].copy()
            components = data["components"].copy()

        return cls(
            mean=mean,
            components=components,
        )
