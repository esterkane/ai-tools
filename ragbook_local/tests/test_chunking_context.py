from ragbook.ingest.chunking import chunk_pages
from ragbook.ingest.pdf_text import PageText


def test_chunk_page_context_and_section():
    text = (
        "INTRODUCTION\n\n"  # heading style
        "This is the first paragraph with some content.\n\n"
        "This is the second paragraph which is also informative.\n\n"
        "This is the third paragraph with extra details."
    )
    pages = [PageText(page=5, text=text)]
    chunks = chunk_pages(pages, max_chars=60, overlap_chars=0, doc_id="doc")
    assert chunks
    for c in chunks:
        assert hasattr(c, "page_start") and c.page_start == 5
        assert hasattr(c, "page_end") and c.page_end == 5
        # section should be captured as 'INTRODUCTION'
        assert c.section is not None
        assert "INTRODUCTION" in c.section
        # pre/post context are strings (maybe empty)
        assert c.pre_context is None or isinstance(c.pre_context, str)
        assert c.post_context is None or isinstance(c.post_context, str)


def test_context_trimming():
    long_para = "A" * 2000
    text = "Heading\n\n" + long_para + "\n\n" + "After"
    pages = [PageText(page=1, text=text)]
    chunks = chunk_pages(pages, max_chars=500, overlap_chars=0, doc_id="d")
    assert chunks
    for c in chunks:
        if c.pre_context:
            assert len(c.pre_context) <= 500
        if c.post_context:
            assert len(c.post_context) <= 500
