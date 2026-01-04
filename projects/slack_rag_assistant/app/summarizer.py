def format_prompt(user_query: str, retrieved_docs: list[str], max_context_chars: int = 3000) -> str:
    """
    Builds a RAG-style prompt: user query + concatenated document context.
    """
    context_block = "\n\n---\n\n".join(retrieved_docs)

    if len(context_block) > max_context_chars:
        context_block = context_block[:max_context_chars] + "\n\n[Context truncated]"

    prompt = f"""
You are an internal support assistant. Use the following documentation and past cases to answer the user's question.

Context:
{context_block}

User Question:
{user_query}

Answer:
""".strip()

    return prompt
