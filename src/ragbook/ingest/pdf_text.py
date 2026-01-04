from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import fitz  # pymupdf


@dataclass
class PageText:
    page: int
    text: str


def _cleanup(text: str) -> str:
    text = text.replace("\u00ad", "")  # soft hyphen
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages_text(pdf_path: Path) -> list[PageText]:
    doc = fitz.open(str(pdf_path))
    pages: list[PageText] = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        txt = page.get_text("text") or ""
        txt = _cleanup(txt)
        if txt:
            pages.append(PageText(page=i + 1, text=txt))
    return pages
