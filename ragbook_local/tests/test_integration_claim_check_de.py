from ragbook.chat_engine import ChatEngine


class FakeStore:
    def __init__(self, docs):
        self._docs = docs

    def search(self, query_vector, limit=8, filter_=None):
        class H:
            def __init__(self, score, payload):
                self.score = score
                self.payload = payload

        hits = []
        for i, (cid, text) in enumerate(self._docs):
            hits.append(H(1.0 - i * 0.1, {"chunk_id": cid, "text": text, "doc_title": "doc", "page": 1}))
        return hits

    def fetch_all_chunks(self):
        out = []
        for cid, text in self._docs:
            out.append({"id": cid, "payload": {"chunk_id": cid, "text": text, "page": 1}})
        return out


class SeqLLM:
    def __init__(self, responses):
        self._responses = list(responses)

    def generate(self, prompt):
        if not self._responses:
            return ""
        return self._responses.pop(0)


def test_claim_check_refuse_mode_german():
    # German sentences: second sentence is unsupported
    responses = [
        "Unterstützter Satz. Nicht unterstützte Behauptung.",
        '["Nicht unterstützte Behauptung."]',
    ]
    store = FakeStore([("c1", "gestützter Inhalt")])
    llm = SeqLLM(responses)

    class DummyEmbedder:
        def embed(self, texts):
            return [[0.0] for _ in texts]

    engine = ChatEngine(store=store, embedder=DummyEmbedder(), llm=llm, top_k=1, min_score=0.0, max_passages=1, claim_check_mode="refuse", language="de")

    res = engine.ask("q")
    assert res["answer"] == "Nicht genug Information in den Büchern."
    assert res["claim_check"]["unsupported"] == ["Nicht unterstützte Behauptung."]
    assert "claim-checked" in res["reason"]
    assert res["passages"]
