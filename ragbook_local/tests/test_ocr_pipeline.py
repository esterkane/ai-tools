from pathlib import Path
import json
import tempfile

from ragbook.ingest import ocr


def test_dry_run_and_report(tmp_path):
    # Create fake PDFs
    p1 = tmp_path / "with_text.pdf"
    p2 = tmp_path / "no_text.pdf"
    p1.write_bytes(b"PDF-with-text")
    p2.write_bytes(b"PDF-no-text")

    # monkeypatch _has_text_layer to simulate
    from ragbook.ingest.ocr import _has_text_layer, batch_ocr

    orig_has = _has_text_layer

    def fake_has(path: Path):
        if path.name == "with_text.pdf":
            return True
        return False

    try:
        # monkeypatch by assignment
        import ragbook.ingest.ocr as ocrmod

        ocrmod._has_text_layer = fake_has

        report_file = tmp_path / "ocr_report.jsonl"
        needs = batch_ocr(tmp_path, tmp_path / "out", dry_run=True, workers=1, report_path=report_file)
        assert len(needs) == 1
        assert needs[0].name == "no_text.pdf"

        # report file should exist and contain two lines
        lines = report_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        data = [json.loads(l) for l in lines]
        from pathlib import Path as _Path
        statuses = {_Path(d["path"]).name: d["status"] for d in data}
        assert statuses["with_text.pdf"] == "skipped_has_text_layer"
        assert statuses["no_text.pdf"] == "needs_ocr"
    finally:
        ocrmod._has_text_layer = orig_has


def test_ocr_run_with_failures(tmp_path, monkeypatch):
    # Create fake PDF
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"bad")

    import ragbook.ingest.ocr as ocrmod

    def fake_has(path: Path):
        return False

    def fake_run(cmd, check, capture_output, text):
        raise RuntimeError("failed")

    orig_has = ocrmod._has_text_layer
    orig_run = ocrmod.subprocess.run

    try:
        ocrmod._has_text_layer = fake_has
        ocrmod.subprocess.run = fake_run

        report_file = tmp_path / "ocr_report.jsonl"
        outs = ocrmod.batch_ocr(tmp_path, tmp_path / "out", dry_run=False, workers=1, report_path=report_file)
        # outputs should be empty because OCR failed
        assert outs == []

        lines = report_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        d = json.loads(lines[0])
        assert d["status"] == "failed"
        assert "failed" in d.get("error", "")
    finally:
        ocrmod._has_text_layer = orig_has
        ocrmod.subprocess.run = orig_run
