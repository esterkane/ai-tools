import numpy as np

from ragbook.chat_engine import ChatEngine


class FakeHit:
    def __init__(self, score, payload):
        self.score = score
        self.payload = payload


class FakeStore:
    def __init__(self, vector_hits, bm25_docs):
        # vector_hits: list of (chunk_id, score)
        self._vector_hits = vector_hits
        # bm25_docs: list of (chunk_id, text)
        self._bm25_docs = bm25_docs

    def search(self, query_vector, limit=8, filter_=None):
        return [FakeHit(s, {"chunk_id": cid, "text": ""}) for cid, s in self._vector_hits]

    def fetch_all_chunks(self):
        out = []
        for cid, text in self._bm25_docs:
            out.append({"id": cid, "payload": {"chunk_id": cid, "text": text}})
        return out


class DummyEmbedder:
    def embed(self, texts):
        # return dummy vector; not used by FakeStore
        return np.zeros((len(texts), 1), dtype=float)


class DummyLLM:
    def generate(self, prompt):
        return "dummy"


def test_fusion_and_sorting():
    # BM25 docs (chunk_id, text)
    bm_docs = [("c1", "apple banana"), ("c2", "banana orange"), ("c3", "apple grape")]
    # vector hits (chunk_id, score)
    vec_hits = [("c2", 0.9), ("c3", 0.5)]

    store = FakeStore(vector_hits=vec_hits, bm25_docs=bm_docs)
    engine = ChatEngine(
        store=store,
        embedder=DummyEmbedder(),
        llm=DummyLLM(),
        top_k=3,
        min_score=0.0,
        max_passages=3,
        alpha=0.6,
    )

    res = engine.ask("apple")
    passages = res["passages"]

    # ensure fused_score field exists and list is sorted descending
    scores = [p["fused_score"] for p in passages]
    assert scores == sorted(scores, reverse=True)

    # ensure top passage chunk_id is one of the known ids
    assert passages[0]["payload"]["chunk_id"] in {"c1", "c2", "c3"}

    # check that BM25-only item (c1) can appear even if not in vector hits
    ids = [p["payload"]["chunk_id"] for p in passages]
    assert "c1" in ids
