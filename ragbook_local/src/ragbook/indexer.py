from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import hashlib

from qdrant_client.http.models import PointStruct

from .ingest import extract_pages_text, chunk_pages
from .embeddings import Embedder
from .store import QdrantStore
from .ingest.ocr import ocr_pdf_if_needed


def _doc_id_from_path(p: Path) -> str:
    h = hashlib.sha1(str(p).encode("utf-8")).hexdigest()[:12]
    return f"{p.stem}-{h}"


@dataclass
class IndexResult:
    docs_indexed: int
    chunks_indexed: int


def index_pdfs(
    pdf_paths: Iterable[Path],
    *,
    store: QdrantStore,
    embedder: Embedder,
    max_chars: int,
    overlap_chars: int,
    ocr_out_dir: Path | None,
) -> IndexResult:
    docs = 0
    chunks_total = 0

    dummy = embedder.embed(["test"])
    store.ensure_collection(vector_size=int(dummy.shape[1]))

    for pdf in pdf_paths:
        pdf_use = pdf
        if ocr_out_dir is not None:
            pdf_use = ocr_pdf_if_needed(pdf, out_dir=ocr_out_dir)

        pages = extract_pages_text(pdf_use)
        doc_id = _doc_id_from_path(pdf)
        doc_title = pdf.stem

        chunks = chunk_pages(pages, max_chars=max_chars, overlap_chars=overlap_chars, doc_id=doc_id)
        if not chunks:
            continue

        vecs = embedder.embed([c.text for c in chunks])
        points = []
        for local_idx, (c, v) in enumerate(zip(chunks, vecs), start=0):
            payload = {
                "chunk_id": c.chunk_id,
                "doc_id": doc_id,
                "doc_title": doc_title,
                "source_path": str(pdf.resolve()),
                "file_link": f"file://{str(pdf.resolve())}#page={int(c.page_start)}",
                "page": int(c.page_start),
                "page_start": int(c.page_start),
                "page_end": int(c.page_end),
                "section": c.section,
                "pre_context": c.pre_context or "",
                "post_context": c.post_context or "",
                "text": c.text,
                "local_idx": int(local_idx),
            }
            points.append(PointStruct(id=c.chunk_id, vector=v.tolist(), payload=payload))

        store.upsert(points)
        docs += 1
        chunks_total += len(points)

    return IndexResult(docs_indexed=docs, chunks_indexed=chunks_total)
