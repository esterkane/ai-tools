from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np

from sentence_transformers import SentenceTransformer


@dataclass
class Embedder:
    model: SentenceTransformer

    @classmethod
    def from_model(cls, model_name_or_path: str, device: str = "cpu") -> "Embedder":
        model = SentenceTransformer(model_name_or_path, device=device)
        return cls(model=model)

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        vecs = self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)
