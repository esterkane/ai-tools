from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import re


@dataclass
class EvidenceDecision:
    should_answer: bool
    reason: str
    probing_questions: list[str]


def _tokenize(s: str) -> list[str]:
    return [t for t in re.findall(r"\w+", s.lower()) if t]


def decide_or_ask(
    query: str,
    fused_hits: Sequence[dict],
    *,
    min_score: float,
    gap_threshold: float = 0.15,
    min_token_overlap: float = 0.15,
    require_doc_coverage: int = 2,
) -> EvidenceDecision:
    """Decide whether to answer or ask probing questions.

    Additional heuristics:
    - If best score < min_score: ask
    - If score gap between top and second < gap_threshold: ask (ambiguous)
    - If token overlap between query and top passage < min_token_overlap: ask
    - If number of unique docs in top N < require_doc_coverage and there are several hits: ask (low coverage)
    """
    if not fused_hits:
        return EvidenceDecision(
            should_answer=False,
            reason="No hits in index.",
            probing_questions=_suggest_questions(query),
        )

    best = float(fused_hits[0].get("fused_score", fused_hits[0].get("score", 0.0)))
    if best < min_score:
        return EvidenceDecision(
            should_answer=False,
            reason=f"Top score too weak (Score {best:.3f} < {min_score:.3f}).",
            probing_questions=_suggest_questions(query),
        )

    # coverage check: look at top min(5, len(fused_hits))
    top_n = fused_hits[: min(5, len(fused_hits))]
    docs = set()
    for h in top_n:
        payload = h.get("payload") or {}
        docs.add(payload.get("doc_id") or payload.get("doc_title") or "")
    if len(top_n) >= 3 and len(docs) < require_doc_coverage:
        return EvidenceDecision(
            should_answer=False,
            reason=f"Low source coverage (only {len(docs)} source(s) in top results).",
            probing_questions=_suggest_questions(query),
        )

    # gap check
    if len(fused_hits) > 1:
        second = float(fused_hits[1].get("fused_score", fused_hits[1].get("score", 0.0)))
        gap = best - second
        if gap < gap_threshold:
            return EvidenceDecision(
                should_answer=False,
                reason=f"Ambiguous top hits (score gap {gap:.3f} < {gap_threshold:.3f}).",
                probing_questions=_suggest_questions(query),
            )

    # token overlap check with top passage (skip for very short queries)
    top_payload = top_n[0].get("payload") or {}
    top_text = top_payload.get("text", "")
    q_tokens = _tokenize(query)
    p_tokens = _tokenize(top_text)
    if len(q_tokens) >= 2:
        overlap = len(set(q_tokens) & set(p_tokens)) / len(q_tokens)
        if overlap < min_token_overlap:
            return EvidenceDecision(
                should_answer=False,
                reason=f"Low lexical overlap with top passage (overlap {overlap:.2f} < {min_token_overlap:.2f}).",
                probing_questions=_suggest_questions(query),
            )
    elif len(q_tokens) == 1:
        # single token queries: require the token to appear in passage
        tok = q_tokens[0]
        # skip overly short tokens (e.g., single-letter queries)
        if len(tok) >= 2 and tok not in set(p_tokens):
            return EvidenceDecision(
                should_answer=False,
                reason=f"Low lexical overlap with top passage (token '{tok}' missing).",
                probing_questions=_suggest_questions(query),
            )

    return EvidenceDecision(should_answer=True, reason="OK", probing_questions=[])


def _suggest_questions(query: str) -> list[str]:
    qs = [
        "Which subfield are you referring to (e.g., Technical Mechanics, Machine Elements, Materials)?",
        "Which chapter/term/standard is relevant (if known)?",
        "Which quantities are given (e.g., force, moment, geometry, material properties) and which should be computed?",
        "Is this a definition/explanation or a concrete worked example?",
        "For formulas: what boundary conditions apply (static/dynamic, linear/nonlinear, safety factors)?",
    ]
    return [f"{q} (Original query: '{query}')" for q in qs[:5]]
