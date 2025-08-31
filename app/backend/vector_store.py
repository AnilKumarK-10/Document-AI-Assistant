# app/backend/vector_store.py

import faiss
import numpy as np
from typing import List


class VectorStore:
    """
    A simple FAISS-backed store for embedding vectors and their corresponding texts.
    """

    def __init__(self, dim: int):
        self._index = faiss.IndexFlatL2(dim)
        self._docs: List[str] = []

    def add(self, texts: List[str], embeddings: List[List[float]]) -> None:
        mat = np.array(embeddings, dtype="float32")
        self._index.add(mat)
        self._docs.extend(texts)

    def query(self, emb: List[float], top_k: int = 5) -> List[str]:
        D, I = self._index.search(np.array([emb], dtype="float32"), top_k)
        return [self._docs[i] for i in I[0]]
