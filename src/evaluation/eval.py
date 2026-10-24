"""Run Ragas metrics against the included handbook question set."""

import argparse
import json
from pathlib import Path

from datasets import Dataset

from src.config import get_settings
from src.generation.answer import generate_answer
from src.retrieval.retriever import HybridRetriever


def run_evaluation(dataset_path: Path, output_dir: Path) -> Path:
    settings = get_settings()
    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_openai import ChatOpenAI
    except ImportError as error:
        raise RuntimeError(
            "Install the evaluation dependencies with requirements.txt"
        ) from error
    llm = ChatOpenAI(
        model=settings.local_model_name,
        base_url=settings.local_model_base_url,
        api_key=settings.local_model_api_key,
        temperature=0,
    )
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

    questions = json.loads(dataset_path.read_text(encoding="utf-8"))
    retriever = HybridRetriever()
    records = []
    for item in questions:
        results = retriever.retrieve(item["question"])
        contexts = [result["text"] for result in results]
        records.append(
            {
                "question": item["question"],
                "answer": generate_answer(item["question"], contexts),
                "contexts": contexts,
                "ground_truth": item["ground_truth"],
            }
        )
    scores = evaluate(
        Dataset.from_list(records),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )
    dataframe = scores.to_pandas()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    summary = {
        metric: float(dataframe[metric].mean())
        for metric in (
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        )
        if metric in dataframe
    }
    json_path.write_text(
        json.dumps({"sample_count": len(records), "metrics": summary}, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# RAG evaluation",
        "",
        f"Evaluated {len(records)} question/context/ground-truth examples.",
        "",
        "| Metric | Mean score |",
        "| --- | ---: |",
        *[f"| {name} | {value:.4f} |" for name, value in summary.items()],
        "",
        "Scores depend on the configured language model and retrieved document index.",
    ]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluation/questions.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/evaluation"))
    args = parser.parse_args()
    print(run_evaluation(args.dataset, args.output_dir))


if __name__ == "__main__":
    main()
