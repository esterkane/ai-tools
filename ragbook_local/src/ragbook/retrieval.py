from __future__ import annotations

from typing import Iterable
from dataclasses import dataclass
from pathlib import Path
import pickle
import re

from rank_bm25 import BM25Okapi


@dataclass
class BM25Result:
    chunk_id: str
    score: float
    text: str


class BM25Index:
    def __init__(self, docs: Iterable[str], ids: Iterable[str], language: str | None = None):
        self.docs = list(docs)
        self.ids = list(ids)
        # language hint (e.g., 'de' for German, 'en' for English, or 'auto')
        self.language = (language or "auto").lower()

        # prepare tokenizer and optional stemmer for German
        self._stemmer = None
        if self.language.startswith("de"):
            try:
                from nltk.stem.snowball import SnowballStemmer

                self._stemmer = SnowballStemmer("german")
            except Exception:
                # if nltk not available, fall back to no stemming
                self._stemmer = None

        self.tokenized = [self._tokenize(d) for d in self.docs]
        if any(self.tokenized):
            self.bm25 = BM25Okapi(self.tokenized)
        else:
            self.bm25 = None

    @classmethod
    def from_store(cls, store, language: str | None = None) -> "BM25Index":
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
        return cls(docs=docs, ids=ids, language=language)

    def _tokenize(self, text: str) -> list[str]:
        # simple unicode-aware word tokenizer
        tokens = re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)
        if self._stemmer is not None:
            try:
                tokens = [self._stemmer.stem(t) for t in tokens]
            except Exception:
                pass
        return tokens

    def search(self, query: str, top_k: int = 8) -> list[BM25Result]:
        if not self.bm25:
            return []
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [BM25Result(chunk_id=self.ids[i], score=float(scores[i]), text=self.docs[i]) for i in ranked]

    def save(self, path: Path) -> None:
        """Persist the BM25 index (docs + ids) to disk using pickle."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"docs": self.docs, "ids": self.ids, "language": self.language}
        with path.open("wb") as fh:
            pickle.dump(data, fh)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        """Load a persisted BM25 index from disk."""
        with path.open("rb") as fh:
            data = pickle.load(fh)
        return cls(docs=data.get("docs", []), ids=data.get("ids", []), language=data.get("language", None))