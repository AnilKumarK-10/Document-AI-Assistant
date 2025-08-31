# app/backend/embeddings.py

from sentence_transformers import SentenceTransformer
from typing import List

_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
# 384


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Encode a list of texts into embedding vectors.
    """
    # convert_to_numpy=True for speed
    return _MODEL.encode(texts, convert_to_numpy=True).tolist()
