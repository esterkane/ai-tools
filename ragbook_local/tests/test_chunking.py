from ragbook.ingest.chunking import chunk_pages
from ragbook.ingest.pdf_text import PageText


def test_chunk_pages_basic():
    pages = [PageText(page=1, text="A\n\nB\n\nC")]
    chunks = chunk_pages(pages, max_chars=3, overlap_chars=0, doc_id="doc")
    assert len(chunks) >= 1
    assert chunks[0].chunk_id.startswith("doc::p1")
