import warnings
import builtins

from ragbook.chat_engine import ChatEngine


class FakeStore:
    def search(self, query_vector, limit=8, filter_=None):
        class H:
            def __init__(self, score, payload):
                self.score = score
                self.payload = payload

        return [H(0.9, {"chunk_id": "c1", "text": "foo"}), H(0.5, {"chunk_id": "c2", "text": "bar"})]

    def fetch_all_chunks(self):
        return [{"id": "c1", "payload": {"chunk_id": "c1", "text": "foo"}}, {"id": "c2", "payload": {"chunk_id": "c2", "text": "bar"}}]


class DummyEmbedder:
    def embed(self, texts):
        return [[0.0]]


class DummyLLM:
    def generate(self, prompt):
        return "ok"


def test_rerank_disabled_when_model_missing(monkeypatch):
    # Simulate CrossEncoder import failure by patching builtins.__import__
    orig_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sentence_transformers":
            raise ImportError("no sentence_transformers")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    store = FakeStore()
    engine = ChatEngine(store=store, embedder=DummyEmbedder(), llm=DummyLLM(), top_k=2, min_score=0.0, max_passages=2, rerank_enabled=True, rerank_model="not-a-real-model", rerank_candidates=5)

    # capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = engine.ask("q")
        # there should be at least one warning about reranker
        assert any("Reranker" in str(x.message) for x in w)
        assert res["answer"] == "ok"


def test_rerank_reorders_candidates(monkeypatch):
    # Provide a fake CrossEncoder that scores second passage higher
    class FakeCrossEncoder:
        def __init__(self, model):
            self.model = model

        def predict(self, texts):
            # return scores inversely proportional to index (so second gets higher)
            return [0.1 if i == 0 else 0.9 for i in range(len(texts))]

    # Patch the CrossEncoder class in sentence_transformers
    try:
        import sentence_transformers as st

        monkeypatch.setattr(st, "CrossEncoder", FakeCrossEncoder)
    except Exception:
        # If the package isn't installed, inject a dummy module
        import types

        mod = types.SimpleNamespace(CrossEncoder=FakeCrossEncoder)
        monkeypatch.setitem(globals(), "sentence_transformers", mod)
        monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", mod)

    class Store2(FakeStore):
        def search(self, query_vector, limit=8, filter_=None):
            class H:
                def __init__(self, score, payload):
                    self.score = score
                    self.payload = payload

            # initial vector scores favor c1
            return [H(0.9, {"chunk_id": "c1", "text": "foo"}), H(0.5, {"chunk_id": "c2", "text": "bar"})]

    store = Store2()
    engine = ChatEngine(store=store, embedder=DummyEmbedder(), llm=DummyLLM(), top_k=2, min_score=0.0, max_passages=2, rerank_enabled=True, rerank_model="fake", rerank_candidates=2)

    res = engine.ask("q")
    passages = res["passages"]
    # After rerank, the second passage should be scored higher and appear first
    assert passages[0]["payload"]["chunk_id"] == "c2"
