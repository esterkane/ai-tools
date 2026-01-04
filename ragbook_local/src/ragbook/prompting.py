from __future__ import annotations


def build_grounded_prompt(question: str, passages: list[dict], language: str = "en") -> str:
    """Build a localized grounded prompt. If `language` starts with 'de' a German prompt will be returned."""
    ctx = []
    for i, p in enumerate(passages, start=1):
        payload = p["payload"]
        ctx.append(
            f"[PASSAGE {i}] doc_title={payload.get('doc_title')} page={payload.get('page')} chunk_id={payload.get('chunk_id')}\n"
            f"{payload.get('text')}\n"
        )

    context_block = "\n\n".join(ctx)

    if (language or "").lower().startswith("de"):
        return (
            "Du bist ein präziser Assistent für technische Lehrbücher (Maschinenbau).\n"
            "Regeln (sehr wichtig):\n"
            "- Antworte *nur* basierend auf den nachfolgenden PASSAGEN.\n"
            "- Wenn die PASSAGEN nicht ausreichen, antworte: \"Nicht genug Information in den Büchern.\" und schlage 3-5 konkrete Folgefragen vor.\n"
            "- Erfinde nichts. Keine Annahmen, keine externen Fakten.\n\n"
            f"Frage:\n{question}\n\n"
            f"PASSAGEN:\n{context_block}\n\n"
            "Ausgabeformat:\n"
            "1) Antwort (kurz, sachlich)\n"
            "2) Belege: Liste der verwendeten chunk_id (mit kurzer Notiz: welche Behauptung wo gestützt wird)\n"
            "3) Fehlende Infos / Folgefragen (falls vorhanden)\n"
        )

    return (
        "You are a precise assistant for technical textbooks (mechanical engineering).\n"
        "Rules (very important):\n"
        "- Answer *only* based on the PASSAGES below.\n"
        '- If the PASSAGES are insufficient, say: "Not enough information in the books." and suggest 3-5 concrete follow-up questions.\n'
        "- Do not invent anything. No assumptions, no external facts.\n\n"
        f"Question:\n{question}\n\n"
        f"PASSAGES:\n{context_block}\n\n"
        "Output format:\n"
        "1) Answer (short, factual)\n"
        "2) Evidence: list of used chunk_id (and short note: which claim is supported where)\n"
        "3) Missing info / follow-up questions (if any)\n"
    )


def build_claim_check_prompt(answer: str, passages: list[dict], language: str = "en") -> str:
    """Build a prompt asking the LLM to mark sentences in `answer` that are NOT directly
    supported by the provided passages. The LLM must return a JSON array of unsupported
    sentences only (e.g. ["sentence 1.", "sentence 2."]). If none, return an empty JSON array `[]`.
    The prompt is localized based on `language` (supports German when language startswith 'de').
    """
    ctx = []
    for i, p in enumerate(passages, start=1):
        payload = p["payload"]
        ctx.append(f"[PASSAGE {i}] page={payload.get('page')} chunk_id={payload.get('chunk_id')}\n{payload.get('text')}\n")

    passages_block = "\n\n".join(ctx)

    if (language or "").lower().startswith("de"):
        return (
            "Überprüfe, welche Sätze in der gegebenen ANTWORT NICHT direkt durch die PASSAGEN gestützt werden. "
            "Gib AUSSCHLIEßLICH ein JSON-Array zurück (keinen zusätzlichen Text) mit den nicht unterstützten Sätzen (exakte Teilsätze aus der Antwort).\n\n"
            "ANTWORT:\n"
            f"{answer}\n\n"
            "PASSAGEN:\n"
            f"{passages_block}\n\n"
            "Wenn alle Sätze gestützt sind, gib `[]` zurück."
        )

    return (
        "Verify which sentences in the provided ANSWER are NOT directly supported by the PASSAGES. "
        "Return ONLY a JSON array (no extra text) listing unsupported sentences (exact substrings from the answer).\n\n"
        "ANSWER:\n"
        f"{answer}\n\n"
        "PASSAGES:\n"
        f"{passages_block}\n\n"
        "If all sentences are supported, return `[]`."
    )


def parse_claim_check_response(response: str) -> list[str]:
    """Parse the LLM response as JSON array of unsupported sentences. Fall back to simple heuristics if parsing fails."""
    import json

    resp = response.strip()
    try:
        parsed = json.loads(resp)
        if isinstance(parsed, list):
            return [s.strip() for s in parsed if isinstance(s, str) and s.strip()]
    except Exception:
        pass

    # Fallback: look for lines starting with '-' or 'UNSUPPORTED:'
    out = []
    for line in resp.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("-"):
            out.append(line.lstrip("- ").strip())
        elif line.upper().startswith("UNSUPPORTED:"):
            out.append(line.split("UNSUPPORTED:", 1)[1].strip())
    return out


def strip_unsubstantiated(answer: str, unsupported: list[str]) -> str:
    """Remove sentences that match any of the unsupported sentences (exact match after trimming).

    This is a simple heuristic: split answer by sentence-ending punctuation and remove matches.
    """
    import re

    if not unsupported:
        return answer

    # naive sentence splitter (keeps punctuation)
    parts = re.split(r"(?<=[.!?])\s+", answer.strip())
    keep = []
    sup_stripped = [s.strip() for s in unsupported]
    for p in parts:
        p_str = p.strip()
        if any(p_str == s or p_str.endswith(s) or s.endswith(p_str) for s in sup_stripped):
            continue
        keep.append(p)
    return " ".join(keep).strip()
