import numpy as np

from ragbook.chat_engine import ChatEngine


class FakeStore:
    def __init__(self, docs):
        # docs: list of (chunk_id, text)
        self._docs = docs

    def search(self, query_vector, limit=8, filter_=None):
        class H:
            def __init__(self, score, payload):
                self.score = score
                self.payload = payload

        # return all docs with decreasing dummy scores
        hits = []
        for i, (cid, text) in enumerate(self._docs):
            hits.append(H(1.0 - i * 0.1, {"chunk_id": cid, "text": text, "doc_title": "doc", "page": 1}))
        return hits

    def fetch_all_chunks(self):
        out = []
        for cid, text in self._docs:
            out.append({"id": cid, "payload": {"chunk_id": cid, "text": text, "page": 1}})
        return out


class DummyEmbedder:
    def embed(self, texts):
        return np.zeros((len(texts), 1), dtype=float)


class SeqLLM:
    def __init__(self, responses):
        # responses: list of strings to return on successive generate() calls
        self._responses = list(responses)

    def generate(self, prompt):
        if not self._responses:
            return ""
        return self._responses.pop(0)


def test_claim_check_refuse_mode():
    # LLM returns an answer, then mark second sentence as unsupported
    responses = [
        "Supported sentence. Unsupported claim.",
        '["Unsupported claim."]',
    ]
    store = FakeStore([("c1", "supported text")])
    llm = SeqLLM(responses)
    engine = ChatEngine(store=store, embedder=DummyEmbedder(), llm=llm, top_k=1, min_score=0.0, max_passages=1, claim_check_mode="refuse")

    res = engine.ask("q")
    assert res["answer"] == "Not enough information in the books."
    assert res["claim_check"]["unsupported"] == ["Unsupported claim."]
    assert "claim-checked" in res["reason"]
    assert res["passages"]  # still included


def test_claim_check_strip_mode():
    # LLM returns an answer, then marks second sentence unsupported; strip should remove it
    responses = [
        "Supported sentence. Unsupported claim. Final supported.",
        '["Unsupported claim."]',
    ]
    store = FakeStore([("c1", "supported text")])
    llm = SeqLLM(responses)
    engine = ChatEngine(store=store, embedder=DummyEmbedder(), llm=llm, top_k=1, min_score=0.0, max_passages=1, claim_check_mode="strip")

    res = engine.ask("q")
    assert "Unsupported claim." not in res["answer"]
    assert "Supported sentence." in res["answer"]
    assert res["claim_check"]["unsupported"] == ["Unsupported claim."]
    assert "claim-checked" in res["reason"]
    assert res["passages"]


def test_claim_check_strip_leads_to_empty():
    # If stripping removes all content, final answer should be refusal message
    responses = [
        "Only unsupported.",
        '["Only unsupported."]',
    ]
    store = FakeStore([("c1", "supported text")])
    llm = SeqLLM(responses)
    engine = ChatEngine(store=store, embedder=DummyEmbedder(), llm=llm, top_k=1, min_score=0.0, max_passages=1, claim_check_mode="strip")

    res = engine.ask("q")
    assert res["answer"] == "Not enough information in the books."
    assert res["claim_check"]["unsupported"] == ["Only unsupported."]
    assert "claim-checked" in res["reason"]
    assert res["passages"]
