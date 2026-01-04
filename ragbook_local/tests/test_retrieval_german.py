from ragbook.retrieval import BM25Index


def test_bm25_german_stemming():
    docs = [
        "Das Auto fährt sehr schnell.",
        "Dieses Dokument handelt von Elektrik.",
    ]
    ids = ["d1", "d2"]

    idx = BM25Index(docs=docs, ids=ids, language="de")

    # query uses different inflection; stemming/tokenization should still find the relevant doc
    res = idx.search("Autos fahren schnell", top_k=1)
    assert res, "Expected at least one BM25 result"
    assert res[0].chunk_id == "d1"
    assert "Auto" in res[0].text or "Auto" in docs[0]
