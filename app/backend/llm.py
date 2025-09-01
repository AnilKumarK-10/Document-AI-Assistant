# app/backend/llm.py

import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict

# Load API key
load_dotenv()
_API_KEY = os.getenv("OPENAI_API_KEY")
if not _API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in environment")

_client = OpenAI(api_key=_API_KEY)


_SYSTEM_PROMPT = (
    "You are a document summarizer and Q&A assistant. "
    "When asked to summarize, produce a precise, concise summary of the PDF. "
    "When asked a question, answer based solely on the provided context."
)


def ask_llm(
    messages: list[dict[str, str]],
    model: str = "gpt-4o-mini",
    max_tokens: int = 512,
    temperature: float = 1.0,
) -> str:
    """
    Send a chat-completion request and return the assistant’s reply.
    Uses max_tokens (supported by gpt-4o-mini and earlier models).
    """
    payload: dict[str, object] = {
        "model": model,
        "messages": [{"role": "system", "content": _SYSTEM_PROMPT}] + messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    resp = _client.chat.completions.create(**payload)
    return resp.choices[0].message.content.strip()




def summarize_llm(
    text: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 150,
    temperature: float = 1.0,
) -> str:
    """
    Ask the LLM for an abstractive summary of the given text.
    """
    prompt = f"Please summarize the following PDF content concisely:\n\n{text}"
    return ask_llm(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
