import os
import pickle
import faiss
import numpy as np

INDEX_PATH = "data/index.faiss"
DOCSTORE_PATH = "data/docstore.pkl"

def load_index():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("FAISS index not found.")
    return faiss.read_index(INDEX_PATH)

def load_docstore():
    with open(DOCSTORE_PATH, "rb") as f:
        data = pickle.load(f)
    return data["ids"], data["docs"]

def search_similar_docs(query_vector: np.ndarray, top_k: int = 3) -> list[str]:
    index = load_index()
    ids, docs = load_docstore()

    D, I = index.search(np.array([query_vector]), top_k)
    results = [docs[i] for i in I[0] if i < len(docs)]

    return results
