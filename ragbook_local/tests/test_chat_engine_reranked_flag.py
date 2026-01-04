from ragbook.chat_engine import ChatEngine


class FakeEmbedder:
    def embed(self, texts):
        return [[0.0] for _ in texts]


class FakeStore:
    def search(self, query_vector, limit=8, filter_=None):
        return []


class FakeLLM:
    def generate(self, prompt):
        return ""


def test_reranked_flag_present_and_default_false():
    engine = ChatEngine(
        store=FakeStore(),
        embedder=FakeEmbedder(),
        llm=FakeLLM(),
        top_k=5,
        min_score=0.0,
        max_passages=2,
    )

    r = engine.ask("hello")
    assert "reranked" in r
    assert r["reranked"] is False
