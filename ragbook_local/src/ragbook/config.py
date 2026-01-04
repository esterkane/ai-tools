from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class PathsConfig:
    books_dir: Path
    data_dir: Path
    ocr_out_dir: Path


@dataclass
class QdrantConfig:
    url: str
    collection: str


@dataclass
class EmbeddingConfig:
    model_name_or_path: str
    device: str = "cpu"


@dataclass
class RetrievalConfig:
    top_k: int = 8
    min_score: float = 0.2
    max_passages: int = 5
    alpha: float = 0.5
    # claim check mode: 'strip' to remove unsupported sentences, 'refuse' to return refusal message
    claim_check_mode: str = "refuse"
    # path to persisted BM25 index (optional). If not set, defaults to data_dir / 'bm25.pkl'
    bm25_path: str | None = None
    # language hint for retrieval / claim-check (e.g., 'de' for German, 'en' for English, 'auto')
    language: str = "auto"


@dataclass
class ChunkingConfig:
    max_chars: int = 2500
    overlap_chars: int = 200


@dataclass
class LLMConfig:
    backend: str = "llama_cpp"  # llama_cpp | ollama
    model_path: str = "./models/your-model.gguf"
    n_ctx: int = 4096
    temperature: float = 0.1
    max_tokens: int = 512
    # ollama
    url: str = "http://localhost:11434"
    model: str = "llama3.1:8b"


@dataclass
class RerankConfig:
    enabled: bool = False
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    candidates: int = 30


@dataclass
class OCRConfig:
    workers: int = 1


@dataclass
class UIConfig:
    host: str = "127.0.0.1"
    port: int = 7860


@dataclass
class AppConfig:
    paths: PathsConfig
    qdrant: QdrantConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    chunking: ChunkingConfig
    llm: LLMConfig
    rerank: RerankConfig
    ui: UIConfig
    ocr: OCRConfig


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    paths = data["paths"]
    qd = data["qdrant"]
    emb = data["embedding"]
    ret = data["retrieval"]
    ch = data["chunking"]
    llm = data["llm"]
    ui = data.get("ui", {})

    cfg = AppConfig(
        paths=PathsConfig(
            books_dir=Path(paths["books_dir"]).expanduser(),
            data_dir=Path(paths["data_dir"]).expanduser(),
            ocr_out_dir=Path(paths["ocr_out_dir"]).expanduser(),
        ),
        qdrant=QdrantConfig(url=qd["url"], collection=qd["collection"]),
        embedding=EmbeddingConfig(
            model_name_or_path=emb["model_name_or_path"], device=emb.get("device", "cpu")
        ),
        retrieval=RetrievalConfig(
            top_k=int(ret.get("top_k", 8)),
            min_score=float(ret.get("min_score", 0.2)),
            max_passages=int(ret.get("max_passages", 5)),
            alpha=float(ret.get("alpha", 0.5)),
            claim_check_mode=ret.get("claim_check", {}).get("mode", "refuse") if isinstance(ret.get("claim_check", {}), dict) else "refuse",
            language=(ret.get("language") if isinstance(ret, dict) else "auto") or "auto",
        ),
        chunking=ChunkingConfig(
            max_chars=int(ch.get("max_chars", 2500)),
            overlap_chars=int(ch.get("overlap_chars", 200)),
        ),
        llm=LLMConfig(
            backend=llm.get("backend", "llama_cpp"),
            model_path=llm.get("model_path", "./models/your-model.gguf"),
            n_ctx=int(llm.get("n_ctx", 4096)),
            temperature=float(llm.get("temperature", 0.1)),
            max_tokens=int(llm.get("max_tokens", 512)),
            url=llm.get("url", "http://localhost:11434"),
            model=llm.get("model", "llama3.1:8b"),
        ),
        rerank=RerankConfig(
            enabled=bool(ret.get("rerank", {}).get("enabled", False)) if isinstance(ret, dict) else False,
            model=(ret.get("rerank", {}).get("model") if isinstance(ret, dict) else None)
            or "cross-encoder/ms-marco-MiniLM-L-6-v2",
            candidates=int((ret.get("rerank", {}).get("candidates", 30) if isinstance(ret, dict) else 30)),
        ),
        ui=UIConfig(host=ui.get("host", "127.0.0.1"), port=int(ui.get("port", 7860))),
        ocr=OCRConfig(workers=int(data.get("ocr", {}).get("workers", 1))),
    )

    # set bm25_path default if not provided in config
    bm25_cfg_path = ret.get("bm25_path") if isinstance(ret, dict) else None
    cfg.retrieval.bm25_path = bm25_cfg_path if bm25_cfg_path is not None else str(cfg.paths.data_dir / "bm25.pkl")

    cfg.paths.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.ocr_out_dir.mkdir(parents=True, exist_ok=True)
    return cfg
