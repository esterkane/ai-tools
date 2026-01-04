from __future__ import annotations

from typing import Iterable
from dataclasses import dataclass
from pathlib import Path
import pickle

from rank_bm25 import BM25Okapi


@dataclass
class BM25Result:
    chunk_id: str
    score: float
    text: str


class BM25Index:
    def __init__(self, docs: Iterable[str], ids: Iterable[str]):
        self.docs = list(docs)
        self.ids = list(ids)
        # simple whitespace tokenizer, lowercased
        self.tokenized = [d.lower().split() for d in self.docs]
        if any(self.tokenized):
            self.bm25 = BM25Okapi(self.tokenized)
        else:
            self.bm25 = None

    @classmethod
    def from_store(cls, store) -> "BM25Index":
        pts = store.fetch_all_chunks()
        docs = []
        ids = []
        for p in pts:
            payload = p.get("payload") if isinstance(p, dict) else None
            if not payload:
                continue
            text = payload.get("text")
            chunk_id = payload.get("chunk_id")
            if text and chunk_id:
                docs.append(text)
                ids.append(chunk_id)
        return cls(docs=docs, ids=ids)

    def search(self, query: str, top_k: int = 8) -> list[BM25Result]:
        if not self.bm25:
            return []
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [BM25Result(chunk_id=self.ids[i], score=float(scores[i]), text=self.docs[i]) for i in ranked]

    def save(self, path: Path) -> None:
        """Persist the BM25 index (docs + ids) to disk using pickle."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"docs": self.docs, "ids": self.ids}
        with path.open("wb") as fh:
            pickle.dump(data, fh)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        """Load a persisted BM25 index from disk."""
        with path.open("rb") as fh:
            data = pickle.load(fh)
        return cls(docs=data.get("docs", []), ids=data.get("ids", []))
