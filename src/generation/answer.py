"""Context-constrained answer generation using a local Ollama model."""

from openai import APIConnectionError, APIStatusError, OpenAI

from src.config import get_settings


def generate_answer(query: str, contexts: list[str]) -> str:
    settings = get_settings()
    client = OpenAI(
        api_key=settings.local_model_api_key,
        base_url=settings.local_model_base_url,
    )
    context_text = "\n\n".join(
        f"[{index}] {context}" for index, context in enumerate(contexts, start=1)
    )
    try:
        completion = client.chat.completions.create(
            model=settings.local_model_name,
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
    except APIConnectionError as error:
        raise RuntimeError(
            "Cannot connect to the local model. Start Ollama and run "
            f"`ollama pull {settings.local_model_name}`."
        ) from error
    except APIStatusError as error:
        raise RuntimeError(
            f"Ollama returned an error ({error.status_code}): {error}. "
            f"Confirm `{settings.local_model_name}` is available with "
            "`ollama list`."
        ) from error
    answer = completion.choices[0].message.content
    if not answer:
        raise RuntimeError("The language model returned an empty answer")
    return answer
