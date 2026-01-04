# ragbook_local — Lokales RAG für technische PDFs (inkl. Scanner-PDFs)

Dieses Projekt ist ein **lokales** (offline-fähiges) RAG-System:
- PDFs (auch gescannte PDFs) werden ingestiert (optional OCR).
- Text wird seitenweise extrahiert, in Chunks gesplittet und in **Qdrant** indexiert.
- Chat-UI (Gradio) beantwortet Fragen **nur** auf Basis gefundener Passagen.
- Bei geringer Evidenz werden **Rückfragen** vorgeschlagen statt zu halluzinieren.
- Die UI zeigt **vollständige Quellen-Passagen** (Highlight/Beleg).

## Schnellstart (empfohlen)
### 1) Qdrant starten (Docker)
```bash
docker compose up -d qdrant
```

### 2) Python-Umgebung
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

### 3) Konfiguration
Kopiere `config.example.yaml` nach `config.yaml` und passe Pfade an.

### 4) Ingest (PDFs indexieren)
```bash
python -m ragbook.cli ingest ./books --config ./config.yaml
```

### 5) UI starten
```bash
python -m ragbook.cli ui --config ./config.yaml
```

## OCR für Scanner-PDFs
OCR ist optional und benötigt System-Dependencies (Tesseract + Ghostscript).
- Linux (Debian/Ubuntu): `sudo apt-get install tesseract-ocr ghostscript`
- Dann:
```bash
python -m ragbook.cli ocr ./books --config ./config.yaml
```
Das schreibt standardmäßig OCR-PDFs in `data/ocr_out/`.

## Hybrid retrieval (BM25 + vector)
A hybrid retrieval mode combines BM25 (lexical) scores with vector similarity from Qdrant. Configure the fusion weight in `config.yaml` under `retrieval.alpha` (float between 0 and 1). The final fused score is:

    fused = alpha * vector_norm + (1 - alpha) * bm25_norm

BM25 uses `rank-bm25` on the stored chunk `text` payloads and runs locally (offline), while vectors are retrieved from Qdrant.

### Optional re-ranking
If you enable `retrieval.rerank.enabled = true`, the system will attempt to load a CrossEncoder from `sentence-transformers` (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`) and re-score the top N candidates (`retrieval.rerank.candidates`). If the model is unavailable or prediction fails, the system logs a warning and proceeds without re-ranking. When re-ranking is applied, the `reason` field returned by the `ChatEngine` will include `(re-ranked)`.

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
## Lokales LLM
Standard: **llama-cpp-python** (lokal GGUF Model).
- Lege ein GGUF Modell ab (z.B. `models/your-model.gguf`) und trage den Pfad in `config.yaml` ein.

Alternative (optional): Ollama (wenn du es verwenden willst), dann `llm.backend: ollama` setzen.

## Hinweise zu Genauigkeit / Anti-Halluzination
Das System erzwingt:
- Antwort nur, wenn Retrieval-Evidenz über Schwellwert
- Jede Antwort wird mit vollständigen Quellenpassagen begründet
- Andernfalls: "Nicht genug Info" + Rückfragen

## Lizenz
Projekt-Code: MIT (siehe `LICENSE`).
