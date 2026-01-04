from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct, PayloadSchemaType


@dataclass
class QdrantStore:
    client: QdrantClient
    collection: str

    @classmethod
    def connect(cls, url: str, collection: str) -> "QdrantStore":
        client = QdrantClient(url=url)
        return cls(client=client, collection=collection)

    def ensure_collection(self, vector_size: int) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="doc_title",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    def upsert(self, points: Iterable[PointStruct]) -> None:
        self.client.upsert(collection_name=self.collection, points=list(points))

    def search(self, query_vector, limit: int = 8, filter_: Any | None = None):
        return self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=limit,
            query_filter=filter_,
        )

    def fetch_all_chunks(self) -> list[dict]:
        """Fetch all points (with payload) from the collection and return a list of dicts with 'id' and 'payload'.

        Uses the Qdrant scroll API to page through results. This is used to build an offline BM25 index.
        """
        points: list[dict] = []
        offset = 0
        limit = 1000
        while True:
            res = self.client.scroll(collection_name=self.collection, limit=limit, offset=offset, with_payload=True)
            if not res:
                break
            for p in res:
                # p might be a PointStruct-like object or a dict, handle both
                pid = getattr(p, "id", None) or (p.get("id") if isinstance(p, dict) else None)
                payload = getattr(p, "payload", None) or (p.get("payload") if isinstance(p, dict) else None)
                points.append({"id": pid, "payload": payload})
            if len(res) < limit:
                break
            offset += len(res)
        return points
