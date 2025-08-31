# app/backend/preprocessor.py

from typing import List


def split_into_chunks(
    text: str,
    chunk_size: int = 1_000,
    overlap: int = 200,
) -> List[str]:
    """
    Split `text` into overlapping chunks of approximately chunk_size characters.
    """
    if not text:
        return []

    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap if end < len(text) else len(text)
    return chunks
