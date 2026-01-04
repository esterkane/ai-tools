from __future__ import annotations

from pathlib import Path
import subprocess
import json
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


def _has_text_layer(pdf_path: Path) -> bool:
    try:
        import fitz  # pymupdf
    except Exception:
        return False

    doc = fitz.open(str(pdf_path))
    if doc.page_count == 0:
        return False
    page = doc.load_page(0)
    txt = page.get_text("text") or ""
    return len(txt.strip()) > 50


def ocr_pdf_if_needed(pdf_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / pdf_path.name

    if out_path.exists():
        return out_path

    if _has_text_layer(pdf_path):
        return pdf_path

    cmd = [
        "ocrmypdf",
        "--skip-text",
        "--deskew",
        "--rotate-pages",
        "--output-type",
        "pdf",
        str(pdf_path),
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            "ocrmypdf not found. Install it (pip) and ensure Tesseract+Ghostscript are available."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"OCR failed: {e.stderr[:800]}") from e

    return out_path


def batch_ocr(
    input_dir: Path,
    out_dir: Path,
    *,
    dry_run: bool = False,
    workers: int = 1,
    report_path: Path | None = None,
) -> list[Path]:
    """Batch OCR PDFs in input_dir and write results to out_dir.

    Args:
        dry_run: If True, only report which PDFs need OCR without running ocrmypdf.
        workers: Number of parallel workers (default 1).
        report_path: Optional path to write a JSONL report; if None, writes to out_dir.parent/ocr_report.jsonl

    Returns:
        For dry_run: list of Path needing OCR. For normal run: list of output Paths (either original or OCRed PDFs).
    """
    pdfs = sorted(input_dir.rglob("*.pdf"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if report_path is None:
        report_path = out_dir.parent / "ocr_report.jsonl"

    reports: list[dict] = []

    # quick dry-run: check which need OCR
    if dry_run:
        needs = []
        for p in pdfs:
            try:
                has = _has_text_layer(p)
            except Exception:
                has = False
            if not has:
                needs.append(p)
                reports.append({"path": str(p), "status": "needs_ocr", "timestamp": datetime.utcnow().isoformat()})
            else:
                reports.append({"path": str(p), "status": "skipped_has_text_layer", "timestamp": datetime.utcnow().isoformat()})
        # write report
        with report_path.open("a", encoding="utf-8") as fh:
            for r in reports:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        return needs

    # non-dry-run: process files (possibly in parallel)
    def _process(p: Path):
        entry = {"path": str(p), "status": "unknown", "timestamp": datetime.utcnow().isoformat()}
        try:
            if _has_text_layer(p):
                entry["status"] = "skipped_has_text_layer"
                logger.info("Skipping (has text): %s", p)
                return entry, p
            out_path = ocr_pdf_if_needed(p, out_dir=out_dir)
            entry["status"] = "ocr_done"
            logger.info("OCR done: %s -> %s", p, out_path)
            return entry, out_path
        except Exception as e:
            entry["status"] = "failed"
            entry["error"] = str(e)
            logger.exception("OCR failed for %s", p)
            return entry, None

    results = []
    if workers and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_process, p): p for p in pdfs}
            for fut in as_completed(futures):
                entry, out = fut.result()
                reports.append(entry)
                if out:
                    results.append(out)
    else:
        for p in pdfs:
            entry, out = _process(p)
            reports.append(entry)
            if out:
                results.append(out)

    # append report lines
    with report_path.open("a", encoding="utf-8") as fh:
        for r in reports:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    return results
