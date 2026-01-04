import json
import pickle
import numpy as np
import faiss
from app.embedding_service import embed_batch

def load_kb_articles(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    articles = load_kb_articles("data/kb_articles.json")
    texts = [article["content"] for article in articles]
    ids = [article["id"] for article in articles]

    embeddings = embed_batch(texts)

    # Save embeddings + metadata
    with open("data/docstore.pkl", "wb") as f:
        pickle.dump({"ids": ids, "docs": texts}, f)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, "data/index.faiss")

    print(f"Ingested {len(texts)} articles into FAISS.")

if __name__ == "__main__":
    main()
