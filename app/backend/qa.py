# app/backend/qa.py

from typing import List
from transformers import pipeline

_QA_PIPELINE = pipeline(
    "question-answering", model="distilbert-base-cased-distilled-squad", device=-1
)


def answer_question(question: str, chunks: List[str]) -> str:
    """
    Run extractive QA on each text chunk and return the best answer span.
    """
    best = {"score": 0.0, "answer": "Unable to find an answer in the document."}
    for chunk in chunks:
        try:
            res = _QA_PIPELINE(question=question, context=chunk)
            if res["score"] > best["score"]:
                best = res
        except Exception:
            continue
    return best["answer"]
