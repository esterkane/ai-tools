from pathlib import Path
from ragbook.chat_engine import ChatEngine
from ragbook.retrieval import BM25Index


class FakeEmbedder:
    def embed(self, texts):
        return [[0.0] for _ in texts]


class FakeStore:
    def search(self, query_vector, limit=8, filter_=None):
        return []


class FakeLLM:
    def generate(self, prompt):
        # return empty answer for simplicity
        return ""


def test_chat_engine_loads_persisted_bm25(tmp_path, monkeypatch):
    called = {"loaded": False}

    def fake_load(path: Path):
        called["loaded"] = True
        return BM25Index(docs=["hello world"], ids=["doc::p1::c1"])

    monkeypatch.setattr(BM25Index, "load", staticmethod(fake_load))

    # create empty file to satisfy existence check
    p = tmp_path / "bm25.pkl"
    p.write_text("")

    engine = ChatEngine(
        store=FakeStore(),
        embedder=FakeEmbedder(),
        llm=FakeLLM(),
        top_k=5,
        min_score=0.0,
        max_passages=2,
        bm25_path=p,
    )

    # call ask; it should attempt to load persisted BM25
    _ = engine.ask("hello")

    assert called["loaded"] is True
