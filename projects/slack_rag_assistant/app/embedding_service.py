from sentence_transformers import SentenceTransformer

# Load the embedding model once
_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_batch(texts: list[str]):
    """
    Converts a list of text documents to normalized dense embeddings.
    Returns a NumPy array shaped (num_texts, embedding_dim).
    """
    return _model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

def embed_text(text: str):
    """
    Converts a single input string to a normalized dense embedding.
    Returns a 1D NumPy array of shape (embedding_dim,).
    """
    return _model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
