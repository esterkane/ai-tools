from __future__ import annotations

from pathlib import Path
import typer

from .config import load_config
from .store import QdrantStore
from .embeddings import Embedder
from .indexer import index_pdfs
from .llm import LLM
from .chat_engine import ChatEngine
from .retrieval import BM25Index
from .ui import launch_ui
from .ingest.ocr import batch_ocr

app = typer.Typer(add_completion=False, help="ragbook_local CLI")


@app.command()
def ingest(
    input_path: Path = typer.Argument(..., exists=True, help="Ordner mit PDFs oder einzelne PDF"),
    config: Path = typer.Option(Path("./config.yaml"), help="Pfad zur config.yaml"),
    ocr: bool = typer.Option(True, help="OCR anwenden, wenn PDF keine Textschicht hat"),
):
    cfg = load_config(config)

    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        pdfs = [input_path]
    else:
        pdfs = sorted(input_path.rglob("*.pdf"))

    if not pdfs:
        raise typer.BadParameter("Keine PDFs gefunden.")

    store = QdrantStore.connect(cfg.qdrant.url, cfg.qdrant.collection)
    embedder = Embedder.from_model(cfg.embedding.model_name_or_path, device=cfg.embedding.device)

    res = index_pdfs(
        pdfs,
        store=store,
        embedder=embedder,
        max_chars=cfg.chunking.max_chars,
        overlap_chars=cfg.chunking.overlap_chars,
        ocr_out_dir=cfg.paths.ocr_out_dir if ocr else None,
    )

    typer.echo(f"Indexing fertig: docs={res.docs_indexed} chunks={res.chunks_indexed}")


@app.command()
def ocr(
    input_dir: Path = typer.Argument(..., exists=True, help="Ordner mit PDFs (Scan)"),
    config: Path = typer.Option(Path("./config.yaml"), help="Pfad zur config.yaml"),
    dry_run: bool = typer.Option(False, help="Dry run: show which PDFs need OCR"),
):
    cfg = load_config(config)
    outs = batch_ocr(input_dir, cfg.paths.ocr_out_dir, dry_run=dry_run, workers=cfg.ocr.workers, report_path=cfg.paths.data_dir / "ocr_report.jsonl")
    if dry_run:
        typer.echo("Dry run complete. PDFs needing OCR:")
        for p in outs:
            typer.echo(f" - {p}")
        typer.echo(f"Report written to: {cfg.paths.data_dir / 'ocr_report.jsonl'}")
    else:
        typer.echo(f"OCR complete. Files (Original or OCR): {len(outs)}")
        typer.echo(f"Report written to: {cfg.paths.data_dir / 'ocr_report.jsonl'}")


@app.command()
def bm25_rebuild(
    config: Path = typer.Option(Path("./config.yaml"), help="Pfad zur config.yaml"),
    output: Path | None = typer.Option(None, help="Output path for BM25 pickle"),
    force: bool = typer.Option(False, help="Overwrite existing file if present"),
):
    """Build BM25 index from the Qdrant store and persist it to disk."""
    cfg = load_config(config)
    out = output or Path(cfg.retrieval.bm25_path)

    store = QdrantStore.connect(cfg.qdrant.url, cfg.qdrant.collection)
    idx = BM25Index.from_store(store)

    if out.exists() and not force:
        typer.echo(f"{out} already exists. Use --force to overwrite.")
        raise typer.Exit(code=1)

    idx.save(out)
    typer.echo(f"BM25 index saved to: {out}")


@app.command()
def ui(
    config: Path = typer.Option(Path("./config.yaml"), help="Pfad zur config.yaml"),
):
    cfg = load_config(config)

    store = QdrantStore.connect(cfg.qdrant.url, cfg.qdrant.collection)
    embedder = Embedder.from_model(cfg.embedding.model_name_or_path, device=cfg.embedding.device)

    llm = LLM.from_config(
        cfg.llm.backend,
        model_path=cfg.llm.model_path,
        n_ctx=cfg.llm.n_ctx,
        temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens,
        ollama_url=cfg.llm.url,
        ollama_model=cfg.llm.model,
    )

    engine = ChatEngine(
        store=store,
        embedder=embedder,
        llm=llm,
        top_k=cfg.retrieval.top_k,
        min_score=cfg.retrieval.min_score,
        max_passages=cfg.retrieval.max_passages,
        alpha=cfg.retrieval.alpha,
        bm25_path=Path(cfg.retrieval.bm25_path) if cfg.retrieval.bm25_path else None,
        rerank_enabled=cfg.rerank.enabled,
        rerank_model=cfg.rerank.model,
        rerank_candidates=cfg.rerank.candidates,
        claim_check_mode=cfg.retrieval.claim_check_mode,
    )
    launch_ui(engine, host=cfg.ui.host, port=cfg.ui.port)


if __name__ == "__main__":
    app()
