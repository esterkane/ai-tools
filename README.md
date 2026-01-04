# ragbook_local — Local RAG for technical PDFs (including scanned PDFs)

This project is located in `ragbook_local/` (moved from the repository root).

- PDFs (including scanned PDFs) are ingested (OCR optional).
- Text is extracted per page, split into chunks, and indexed in **Qdrant**.
- The chat UI (Gradio) answers questions **only** based on found passages.
- When evidence is insufficient, the system suggests **probing questions** instead of hallucinating.
- The UI displays **full source passages** (highlights/citations).

---

# Repository Projects

This repository contains multiple projects:

- `ragbook_local/` — the local RAG system (this project). See `ragbook_local/README.md` for usage.
- `projects/slack_rag_assistant/` — Slack RAG Assistant (independent project).
- `projects/udacity-project-apply-lightweight-fine-tuning/` — Udacity project workspace.

To run tests or install dependencies for the RAG project, `cd` into `ragbook_local/` and follow its `README.md`.

## Quickstart (recommended)
### 1) Start Qdrant (Docker)
```bash
docker compose up -d qdrant
```

### 2) Python environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

### 3) Configuration
Copy `config.example.yaml` to `config.yaml` and adjust paths.

### 4) Ingest (index PDFs)
```bash
python -m ragbook.cli ingest ./books --config ./config.yaml
```

### 5) Start UI
```bash
python -m ragbook.cli ui --config ./config.yaml
```

## OCR for scanned PDFs
OCR is optional and requires system dependencies (Tesseract + Ghostscript).
- Linux (Debian/Ubuntu): `sudo apt-get install tesseract-ocr ghostscript`
- Then:
```bash
python -m ragbook.cli ocr ./books --config ./config.yaml
```
This writes OCR PDFs to `data/ocr_out/` by default.

## Hybrid retrieval (BM25 + vector)
A hybrid retrieval mode combines BM25 (lexical) scores with vector similarity from Qdrant. Configure the fusion weight in `config.yaml` under `retrieval.alpha` (float between 0 and 1). The final fused score is:

    fused = alpha * vector_norm + (1 - alpha) * bm25_norm

BM25 uses `rank-bm25` on the stored chunk `text` payloads and runs locally (offline), while vectors are retrieved from Qdrant.

### Optional re-ranking
If you enable `retrieval.rerank.enabled = true`, the system will attempt to load a CrossEncoder from `sentence-transformers` (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`) and re-score the top N candidates (`retrieval.rerank.candidates`). If the model is unavailable or prediction fails, the system logs a warning and proceeds without re-ranking. When re-ranking is applied, the `reason` field returned by the `ChatEngine` will include `(re-ranked)`.

**Language support (German / multilingual)**
- If your books are in German, set `retrieval.language: "de"` in `config.yaml` to enable German tokenization/stemming for BM25 and to localize claim-check prompts and refusal messages.
- We recommend a multilingual embedding model (e.g., `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) and a German-capable reranker if you use re-ranking.

Example `retrieval` config snippet (add to `config.yaml`):

```yaml
retrieval:
  top_k: 8
  min_score: 0.20
  max_passages: 5
  language: "de"
```

Note: install `nltk` in your environment (`pip install nltk`) — the project already lists `nltk` as a dependency in `pyproject.toml` for German stemming support. If you don't want to install NLTK, BM25 will still work without stemming, but results may be less robust for German inflections.

### Persisted BM25 index & CLI
To avoid rebuilding the BM25 index from Qdrant on every startup you can persist it to disk with the new CLI command:

```bash
python -m ragbook.cli bm25-rebuild --config ./config.yaml --output ./data/bm25.pkl
```

Notes:
- `--output` is optional; if omitted the command writes to the configured default `retrieval.bm25_path` (defaults to `data_dir/bm25.pkl`).
- If the file already exists the command will exit with an error unless you pass `--force` to overwrite.
- `ChatEngine` will try to load the persisted index from `retrieval.bm25_path` at query time and fall back to building from the Qdrant store if loading fails.

### Claim-check after generation
A post-generation claim-check step validates which sentences in the LLM answer are directly supported by the retrieved passages. Configure the behavior under `retrieval.claim_check.mode` with either:

- `strip` — remove unsupported sentences from the answer (if everything is removed, respond: "Not enough information in the books.")
- `refuse` — replace the answer with "Not enough information in the books." when unsupported sentences are detected.

In all cases the UI continues to display the full passages. The claim-check prompt asks the LLM to return a JSON array of unsupported sentences and the system will fall back gracefully if parsing fails.
## Local LLM
Default: **llama-cpp-python** (local GGUF model).
- Place a GGUF model file (e.g., `models/your-model.gguf`) and set the path in `config.yaml`.

Alternative (optional): Ollama (if you want to use it), set `llm.backend: ollama`.

## Notes on accuracy / anti-hallucination
The system enforces:
- Answer only when retrieval evidence exceeds a threshold
- Each answer is supported by full source passages
- Otherwise: "Not enough info" + probing questions

## Lizenz
Projekt-Code: MIT (siehe `LICENSE`).
