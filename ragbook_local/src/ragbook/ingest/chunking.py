from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .pdf_text import PageText


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page_start: int
    page_end: int
    section: str | None
    pre_context: str | None
    post_context: str | None


def _is_heading(para: str) -> bool:
    # Heuristic: short line (<=6 words), no terminal period, not too long
    words = para.split()
    if len(words) <= 6 and len(para) <= 80 and not para.strip().endswith("."):
        return True
    return False


def _trim_context(s: str, max_chars: int = 500) -> str:
    if s is None:
        return s
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3].rstrip() + "..."


def chunk_pages(
    pages: Iterable[PageText], *, max_chars: int = 2500, overlap_chars: int = 200, doc_id: str
) -> list[Chunk]:
    chunks: list[Chunk] = []
    n = 0

    for p in pages:
        paras = [x.strip() for x in p.text.split("\n\n") if x.strip()]
        buf_paras: list[str] = []
        section = None
        for idx, para in enumerate(paras):
            # update section heading
            if _is_heading(para):
                section = para

            if not buf_paras:
                buf_paras = [para]
                start_idx = idx
            elif sum(len(x) for x in buf_paras) + 2 * (len(buf_paras) - 1) + len(para) <= max_chars:
                buf_paras.append(para)
            else:
                n += 1
                chunk_text = "\n\n".join(buf_paras)
                pre = paras[start_idx - 1] if start_idx - 1 >= 0 else None
                post = paras[idx] if idx < len(paras) else None
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}::p{p.page}::c{n}",
                        text=chunk_text,
                        page_start=p.page,
                        page_end=p.page,
                        section=section,
                        pre_context=_trim_context(pre),
                        post_context=_trim_context(post),
                    )
                )
                if overlap_chars > 0:
                    # create a buf from tail of last chunk
                    tail = chunk_text[-overlap_chars:]
                    buf_paras = [tail, para]
                    start_idx = idx
                else:
                    buf_paras = [para]
                    start_idx = idx

        if buf_paras:
            n += 1
            chunk_text = "\n\n".join(buf_paras)
            pre = paras[start_idx - 1] if start_idx - 1 >= 0 else None
            post = None
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::p{p.page}::c{n}",
                    text=chunk_text,
                    page_start=p.page,
                    page_end=p.page,
                    section=section,
                    pre_context=_trim_context(pre),
                    post_context=_trim_context(post),
                )
            )

    return chunks
