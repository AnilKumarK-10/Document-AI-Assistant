# app/backend/pdf_loader.py

from typing import Union
import io
from PyPDF2 import PdfReader


def load_contract(path_or_bytes: Union[str, bytes]) -> str:
    """
    Extract all text from a PDF file or raw bytes.
    """
    reader = (
        PdfReader(io.BytesIO(path_or_bytes))
        if isinstance(path_or_bytes, (bytes, bytearray))
        else PdfReader(path_or_bytes)
    )
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()
