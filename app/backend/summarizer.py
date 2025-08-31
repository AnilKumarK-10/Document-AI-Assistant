import re
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, Pipeline

# Load model & tokenizer
_TOKENIZER = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
_MODEL = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
_SUMMARIZER: Pipeline = pipeline(
    "summarization", model=_MODEL, tokenizer=_TOKENIZER, device=-1
)


def _truncate_to_100_words(raw: str) -> str:
    words = re.split(r"\s+", raw.strip())
    if len(words) <= 100:
        return raw.strip()
    return " ".join(words[:100]) + " …"


def summarize_text(
    text: str,
    max_output_tokens: Optional[int] = 150,
    min_output_tokens: Optional[int] = 80,
) -> str:
    """
    Abstractive summary (~100 words) using BART, truncating input to 1024 tokens.
    """
    if not text.strip():
        return ""
    encoded = _TOKENIZER(
        text,
        max_length=1024,
        truncation=True,
        return_tensors="pt",
    )
    summary_ids = _MODEL.generate(
        input_ids=encoded.input_ids,
        attention_mask=encoded.attention_mask,
        max_length=max_output_tokens,
        min_length=min_output_tokens,
        do_sample=False,
    )
    raw_summary = _TOKENIZER.decode(summary_ids[0], skip_special_tokens=True)
    return _truncate_to_100_words(raw_summary)
