from __future__ import annotations

import gradio as gr
from .chat_engine import ChatEngine


def _format_passages(passages: list[dict]) -> str:
    lines = []
    for i, h in enumerate(passages, start=1):
        p = h["payload"]
        # file link (fallback to absolute path)
        file_link = p.get("file_link") or p.get("source_path")
        file_md = f"[{file_link}]({file_link})" if file_link else ""

        pre = (p.get("pre_context") or "").strip()
        post = (p.get("post_context") or "").strip()

        # show up to 500 chars of context
        def _short(s: str) -> str:
            return (s[:497] + "...") if s and len(s) > 500 else s

        pre_md = f"**context before:** { _short(pre) }\n\n" if pre else ""
        post_md = f"\n\n**context after:** { _short(post) }" if post else ""

        page_range = f"Page {p.get('page_start')}" if p.get("page_start") == p.get("page_end") else f"Pages {p.get('page_start')}-{p.get('page_end')}"

        lines.append(
            f"### Passage {i} — {p.get('doc_title')} ({page_range})\n"
            f"**chunk_id:** `{p.get('chunk_id')}`  \n"
            f"**score:** {h.get('fused_score', h.get('score', 0.0)):.3f}\n"
            f"**source:** {file_md}  \n"
            f"**section:** {p.get('section') or '_n/a_'}\n\n"
            f"{pre_md}{p.get('text')}{post_md}\n"
        )
    return "\n\n---\n\n".join(lines) if lines else "_No passages._"


def launch_ui(engine: ChatEngine, host: str = "127.0.0.1", port: int = 7860):
    with gr.Blocks(title="ragbook_local") as demo:
        gr.Markdown("# ragbook_local — Local RAG (with citations)")

        q = gr.Textbox(label="Question", placeholder="Ask a question about your books…", lines=3)
        ask_btn = gr.Button("Answer")
        clear_btn = gr.Button("Clear")

        ans = gr.Markdown(label="Answer")
        reason = gr.Markdown(label="Decision / Reason")
        meta = gr.Markdown(label="Meta")
        probes = gr.Markdown(label="Probing questions (if needed)")
        passages = gr.Markdown(label="Source passages (full)")

        def _ask(question: str):
            if not question or not question.strip():
                return "", "", "", "", ""
            r = engine.ask(question.strip())
            probes_md = ""
            if r.get("probing_questions"):
                probes_md = "\n".join([f"- {x}" for x in r["probing_questions"]])

            # metadata: rerank + claim-check
            meta_lines = []
            if r.get("reranked"):
                meta_lines.append("**Re-ranked:** ✅")
            else:
                meta_lines.append("**Re-ranked:** ❌")

            cc = r.get("claim_check") or {}
            if cc:
                unsup = cc.get("unsupported") or []
                meta_lines.append(f"**Claim-check:** mode={cc.get('mode')}  — unsupported sentences={len(unsup)}")
                if unsup:
                    meta_lines.append("\n**Unsupported:**\n" + "\n".join([f"- {s}" for s in unsup]))

            meta_md = "\n\n".join(meta_lines)

            return (
                r.get("answer", ""),
                f"**Reason:** {r.get('reason','')}",
                meta_md,
                probes_md,
                _format_passages(r.get("passages", [])),
            )

        ask_btn.click(_ask, inputs=[q], outputs=[ans, reason, meta, probes, passages])
        clear_btn.click(lambda: ("", "", "", "", ""), outputs=[q, ans, reason, meta, probes, passages])

        gr.Markdown(
            "Note: The system only responds when evidence is sufficient. "
            "Otherwise probing questions are suggested."
        )

    demo.launch(server_name=host, server_port=port)
