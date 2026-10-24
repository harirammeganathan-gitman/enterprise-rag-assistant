"""Context-constrained answer generation using an OpenAI-compatible API."""

import os

from src.config import get_settings


def generate_answer(query: str, contexts: list[str]) -> str:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY must be configured before using /chat")
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
        os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
        os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)
        from langfuse.openai import OpenAI
    else:
        from openai import OpenAI

    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or None,
    )
    context_text = "\n\n".join(
        f"[{index}] {context}" for index, context in enumerate(contexts, start=1)
    )
    completion = client.chat.completions.create(
        model=settings.openai_model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the question using only the supplied document context. "
                    "If the context does not contain the answer, say so. "
                    "Cite supporting chunks using their bracketed numbers."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {query}",
            },
        ],
    )
    answer = completion.choices[0].message.content
    if not answer:
        raise RuntimeError("The language model returned an empty answer")
    return answer
