from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from .embeddings import Embedder
from .store import QdrantStore
from .guardrails import decide_or_ask
from .prompting import (
    build_grounded_prompt,
    build_claim_check_prompt,
    parse_claim_check_response,
    strip_unsubstantiated,
)
from .llm import LLM
from .retrieval import BM25Index


@dataclass
class ChatEngine:
    store: QdrantStore
    embedder: Embedder
    llm: LLM
    top_k: int
    min_score: float
    max_passages: int
    alpha: float = 0.5
    bm25_index: Optional[BM25Index] = None
    bm25_path: Optional[Path] = None
    rerank_enabled: bool = False
    rerank_model: str | None = None
    rerank_candidates: int = 30
    _reranker: object | None = None
    claim_check_mode: str = "refuse"

    def ask(self, question: str) -> dict:
        # ensure BM25 index exists (lazy build). Prefer persisted index if available.
        if self.bm25_index is None:
            # try loading persisted index
            if self.bm25_path is not None:
                try:
                    if Path(self.bm25_path).exists():
                        self.bm25_index = BM25Index.load(self.bm25_path)
                except Exception:
                    # loading failed; will fall back to building from store
                    self.bm25_index = None

            if self.bm25_index is None:
                try:
                    self.bm25_index = BM25Index.from_store(self.store)
                except Exception:
                    # if BM25 cannot be built, continue with vector-only behavior
                    self.bm25_index = None

        qv = self.embedder.embed([question])[0]
        # support numpy arrays or python lists
        if hasattr(qv, "tolist"):
            qv_list = qv.tolist()
        elif isinstance(qv, (list, tuple)):
            qv_list = list(qv)
        else:
            qv_list = [qv]
        hits = self.store.search(query_vector=qv_list, limit=max(self.top_k, self.max_passages))

        # vector scores map: chunk_id -> score
        vec_map = {}
        for h in hits:
            cid = h.payload.get("chunk_id") if h.payload else None
            vec_map[cid] = float(h.score)

        # bm25 scores map: chunk_id -> score
        bm25_map = {}
        if self.bm25_index is not None:
            bm25_hits = self.bm25_index.search(question, top_k=self.top_k)
            for r in bm25_hits:
                bm25_map[r.chunk_id] = float(r.score)

        # union of chunk ids (preserve order deterministically)
        all_ids = list(dict.fromkeys(list(vec_map.keys()) + list(bm25_map.keys())))

        # prepare lists for normalization (order consistent via all_ids list)
        vec_vals = [vec_map.get(i, 0.0) for i in all_ids]
        bm_vals = [bm25_map.get(i, 0.0) for i in all_ids]

        def normalize(vals):
            if not vals:
                return {}
            mx = max(vals)
            mn = min(vals)
            if mx == mn:
                # if all equal and non-zero, map non-zero to 1.0, zeros to 0.0
                return {i: (1.0 if v > 0 else 0.0) for i, v in zip(all_ids, vals)}
            return {i: ((v - mn) / (mx - mn)) for i, v in zip(all_ids, vals)}

        vec_norm = normalize(vec_vals)
        bm_norm = normalize(bm_vals)

        fused = []
        for cid in all_ids:
            v = vec_norm.get(cid, 0.0)
            b = bm_norm.get(cid, 0.0)
            fused_score = float(self.alpha * v + (1 - self.alpha) * b)
            # choose payload from whichever is available (vector hits preferred)
            payload = None
            if cid in vec_map:
                # find hit payload
                for h in hits:
                    if h.payload and h.payload.get("chunk_id") == cid:
                        payload = h.payload
                        break
            if payload is None and cid in bm25_map:
                # no vector payload, try to recover basic payload from bm25 text
                payload = {"chunk_id": cid, "text": self.bm25_index.docs[list(self.bm25_index.ids).index(cid)]}

            fused.append({"fused_score": fused_score, "payload": payload})

        fused_sorted = sorted(fused, key=lambda x: x["fused_score"], reverse=True)

        # decision uses fused_score
        decision = decide_or_ask(question, fused_sorted, min_score=self.min_score)
        if not decision.should_answer:
            return {
                "answer": "Nicht genug Information in den Büchern.",
                "reason": decision.reason,
                "passages": fused_sorted[: self.max_passages],
                "probing_questions": decision.probing_questions,
                "claim_check": {"mode": self.claim_check_mode, "unsupported": []},
                "reranked": False,
            }

        # Optional re-ranking step (top-N)
        rerank_active = False
        if self.rerank_enabled:
            # build reranker lazily
            if self._reranker is None:
                try:
                    from sentence_transformers import CrossEncoder

                    self._reranker = CrossEncoder(self.rerank_model)
                except Exception:
                    # Log warning but continue without reranking
                    import warnings

                    warnings.warn("Reranker model not available; continuing without reranking.")
                    self._reranker = None

            if self._reranker is not None:
                candidates = fused_sorted[: self.rerank_candidates]
                texts = [f"{question} \n\n {c['payload'].get('text','') or ''}" for c in candidates]
                try:
                    scores = self._reranker.predict(texts)
                    for c, s in zip(candidates, scores):
                        c["fused_score"] = float(s)
                    # re-sort by new scores
                    fused_sorted = sorted(candidates + fused_sorted[self.rerank_candidates :], key=lambda x: x["fused_score"], reverse=True)
                    rerank_active = True
                except Exception:
                    import warnings

                    warnings.warn("Reranker prediction failed; proceeding without reranking.")

        passages = fused_sorted[: self.max_passages]
        prompt = build_grounded_prompt(question, passages)
        answer = self.llm.generate(prompt)

        # Claim-check step: ask LLM to mark unsupported sentences
        claim_check_prompt = build_claim_check_prompt(answer, passages)
        unsupported = []
        try:
            resp = self.llm.generate(claim_check_prompt)
            unsupported = parse_claim_check_response(resp)
        except Exception:
            # if claim-check fails, treat as no unsupported sentences (fail-open)
            unsupported = []

        final_answer = answer
        claim_check_applied = False
        if unsupported:
            claim_check_applied = True
            if (self.claim_check_mode or "").lower() == "strip":
                final_answer = strip_unsubstantiated(answer, unsupported)
                if not final_answer.strip():
                    final_answer = "Not enough information in the books."
            else:
                # default/refuse behavior
                final_answer = "Not enough information in the books."

        reason_suffix = decision.reason
        if rerank_active:
            reason_suffix = f"{reason_suffix} (re-ranked)"
        if claim_check_applied:
            reason_suffix = f"{reason_suffix} (claim-checked)"

        return {
            "answer": final_answer,
            "reason": reason_suffix,
            "passages": passages,  # UI should still show full passages
            "probing_questions": [],
            "claim_check": {"mode": self.claim_check_mode, "unsupported": unsupported},
            "reranked": rerank_active,
        }
