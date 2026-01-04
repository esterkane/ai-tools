# Copilot / AI Agent Instructions for ragbook_local

Quick, focused guide so an AI coding assistant can be immediately productive in this codebase.

## Big picture (what this repo does)
- Local RAG system for technical PDFs (incl. scanned PDFs + OCR).
- Ingest -> chunk -> embed -> index (Qdrant) -> retrieve -> decide (guardrails) -> LLM answer.
- UI (Gradio) shows answers grounded only in retrieved passages; if evidence is weak, the system asks probing questions instead of hallucinating.

## Where to start (commands you can run locally) ✅
- Start Qdrant (docker-compose includes a `qdrant` service):
  - `docker compose up -d qdrant`
- Create/activate Python venv and install dev deps:
  - `python -m venv .venv && source .venv/bin/activate && pip install -U pip && pip install -e ".[dev]"`
- Copy `config.example.yaml` -> `config.yaml` and edit paths, qdrant url/collection, models.
- Ingest PDFs (optionally OCR):
  - `python -m ragbook.cli ingest ./books --config ./config.yaml`
- Run OCR separately (requires Tesseract + Ghostscript):
  - `python -m ragbook.cli ocr ./books --config ./config.yaml`
- Start the UI (Gradio):
  - `python -m ragbook.cli ui --config ./config.yaml`
- Run tests: `pytest` (dev deps include `pytest`).

## Key files & responsibilities 🔧
- `src/ragbook/cli.py` — CLI entrypoints: `ingest`, `ocr`, `ui`.
- `src/ragbook/config.py` — Loads `config.yaml` into `AppConfig`. Note: it creates `data_dir` and `ocr_out_dir` if missing.
- `src/ragbook/ingest/` — PDF extraction and chunking (`pdf_text.py`, `chunking.py`), plus OCR helpers (`ocr.py`).
- `src/ragbook/embeddings.py` — Embedder wrapper around `sentence-transformers` (normalized vectors).
- `src/ragbook/store.py` — Qdrant client wrapper; creates collection when needed.
- `src/ragbook/indexer.py` — Orchestrates chunking → embedding → upsert to Qdrant.
- `src/ragbook/chat_engine.py` — Retrieval and decision flow (calls guardrails then prompts LLM).
- `src/ragbook/prompting.py` — Builds the grounded prompt. Important: UI and LLM rely on this template.
- `src/ragbook/guardrails.py` — Decides whether evidence suffices (uses `min_score`) and suggests probing questions.
- `src/ragbook/llm.py` — LLM abstraction: supports `llama_cpp` (local GGUF via `llama-cpp-python`) and `ollama` (HTTP API).

## Project conventions & important patterns 🧭
- Retrieval-first answers only: the assistant should never add facts not present in passages. The prompting and guardrails enforce this.
- Prompt construction: `build_grounded_prompt(question, passages)` embeds passages with metadata. When editing prompts, preserve the explicit rule list (not inventing facts, returning probing questions when evidence is weak).
- Chunk IDs: format is `docid::p{page}::c{n}`; chunk payloads include `chunk_id`, `doc_id`, `doc_title`, `page`, `text`, `local_idx`.
- Vector embeddings are normalized (`SentenceTransformer(..., normalize_embeddings=True)`) and the Qdrant collection is created with cosine distance; `indexer.index_pdfs` probes embedder once to determine `vector_size`.
- Guardrail behavior: if best score < `min_score` (from config), `decide_or_ask` returns probing questions rather than answering.
- Output shape from `ChatEngine.ask`: `{'answer', 'reason', 'passages', 'probing_questions'}` — tests and UI rely on that.

## Configuration surface & defaults ⚙️
- `config.example.yaml` keys map to dataclasses in `config.py` (paths, qdrant, embedding, retrieval, chunking, llm, ui).
- Retrieval: supports vector-only and hybrid retrieval. New options:
  - `retrieval.alpha` (float 0..1) — fusion weight for vectors (fused = alpha*vector + (1-alpha)*bm25).
  - `retrieval.rerank.enabled` (bool) — enable optional CrossEncoder re-ranking.
  - `retrieval.rerank.model` (string) — cross-encoder model name (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`).
  - `retrieval.rerank.candidates` (int) — number of top candidates to re-rank (default: 30).
- LLM choices:
  - `llama_cpp`: requires `llama-cpp-python` and a local GGUF model (`llm.model_path`).
  - `ollama`: uses HTTP API at `llm.url` with `llm.model` name.
- OCR: system dependencies required (Tesseract, Ghostscript). See README.

Notes:
- BM25 uses `rank-bm25` locally on chunk `text` payloads; this runs offline (no external API).
- CrossEncoder re-ranking (if enabled) uses `sentence-transformers` CrossEncoder; the model is optional — if unavailable the system logs a warning and continues without re-ranking.

## Debugging tips & developer workflows 🐞
- To inspect Qdrant: check `http://localhost:6333` (container exposes 6333/6334). Data persists to `./data/qdrant` per `docker-compose.yml`.
- If embeddings fail to load, ensure the SentenceTransformer model name/path in config is accessible and dependencies installed.
- For LLM issues:
  - `llama-cpp-python` instantiation raises informative runtime errors when missing; for `ollama`, check `OLLAMA` API reachable at configured `llm.url`.
- Tests: current test coverage is small (e.g. `tests/test_chunking.py`); extend tests to cover retrieval & prompt building where appropriate.
- Linting: ruff config in `pyproject.toml`; prefer keeping line length <= 100.

## What to watch for when changing behavior 💡
- Changing prompt rules or guardrails affects user-facing correctness; add tests that assert the `ChatEngine.ask` output format and `decide_or_ask` decisions.
- Changing chunking params affects doc coverage and embedding counts; observe how `indexer.index_pdfs` computes `vector_size` by embedding a dummy input first.
- If you modify the `payload` content, update UI and prompt code that references these keys (e.g., `doc_title`, `page`, `text`, `chunk_id`).

---
If any section is unclear or you want different language style (e.g., English), tell me which parts to expand or adjust and I will iterate. ✅
