from typer.testing import CliRunner
from pathlib import Path
import yaml
from ragbook.cli import app
from ragbook.retrieval import BM25Index
from ragbook.store import QdrantStore


def test_bm25_rebuild_creates_file(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config = {
        "paths": {"books_dir": str(tmp_path / "books"), "data_dir": str(data_dir), "ocr_out_dir": str(data_dir / "ocr")},
        "qdrant": {"url": "http://localhost:6333", "collection": "test"},
        "embedding": {"model_name_or_path": "all-MiniLM-L6-v2"},
        "retrieval": {},
        "chunking": {},
        "llm": {},
    }
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(config))

    # fake store
    class FakeStore:
        def fetch_all_chunks(self):
            return [
                {"id": 1, "payload": {"chunk_id": "doc::p1::c1", "text": "hello world"}},
                {"id": 2, "payload": {"chunk_id": "doc::p1::c2", "text": "another text"}},
            ]

    monkeypatch.setattr(QdrantStore, "connect", lambda url, collection: FakeStore())

    runner = CliRunner()
    out_path = data_dir / "bm25_test.pkl"
    result = runner.invoke(app, ["bm25-rebuild", "--config", str(cfg_file), "--output", str(out_path)])
    assert result.exit_code == 0, result.output
    assert out_path.exists()

    idx = BM25Index.load(out_path)
    assert "doc::p1::c1" in idx.ids
    assert "hello world" in idx.docs
