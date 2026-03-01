"""Evaluate RAG model answers against gold answers."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from langfuse import observe
from langfuse.openai import OpenAI
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from src.rag.config import settings
from src.rag.evaluation.llm_judge import LLMJudge
from src.rag.generation.rag import RAGPipeline
from src.rag.models.embedding_model import EmbeddingModel
from src.rag.observability.langfuse_client import configure_langfuse, flush_langfuse
from src.rag.retrieval.retrievers import BM25Retriever, DenseRetriever, HybridRetriever
from src.rag.storage.vector_store import VectorStore

console = Console()

# Configure LangFuse for @observe decorators
configure_langfuse()


class GoldEvaluationExample(BaseModel):
    """Single evaluation example with query and reference answer."""

    query: str
    answer: str = Field(description="Gold/reference answer")
    contexts: list[str] = Field(default_factory=list)
    trace_id: str | None = None


class GoldComparisonResponse(BaseModel):
    """Structured LLM-judge output for model-vs-gold comparison."""

    reasoning: str
    score: int = Field(
        ge=1,
        le=5,
        description="1 = poor match, 5 = excellent match with the gold answer",
    )


def load_examples(examples_file: Path) -> list[GoldEvaluationExample]:
    """Load examples from JSON file."""
    with open(examples_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [GoldEvaluationExample(**item) for item in data]


def _build_rag_pipeline(collection_name: str, retrieval_method: str) -> RAGPipeline:
    """Create a RAG pipeline with the configured retriever."""
    embedding_model = EmbeddingModel(
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_endpoint=settings.azure_openai_endpoint,
        deployment_name=settings.azure_openai_embedding_deployment_name,
    )

    vector_store = VectorStore(
        persist_directory=settings.chroma_persist_directory,
        collection_name=collection_name,
    )

    retrieval_method = retrieval_method.lower()
    if retrieval_method == "dense":
        retriever = DenseRetriever(
            vector_store=vector_store,
            embedding_model=embedding_model,
            top_k=settings.retrieval_dense_top_k,
        )
    elif retrieval_method == "bm25":
        retriever = BM25Retriever(
            vector_store=vector_store,
            top_k=settings.retrieval_sparse_top_k,
        )
    else:
        dense_retriever = DenseRetriever(
            vector_store=vector_store,
            embedding_model=embedding_model,
            top_k=settings.retrieval_dense_top_k,
        )
        sparse_retriever = BM25Retriever(
            vector_store=vector_store,
            top_k=settings.retrieval_sparse_top_k,
        )
        retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            dense_weight=settings.hybrid_search_dense_weight,
            sparse_weight=settings.hybrid_search_sparse_weight,
            final_top_k=settings.retrieval_hybrid_final_top_k,
        )

    llm_client = OpenAI(
        api_key=settings.groq_api_key,
        base_url=settings.groq_base_url,
    )

    return RAGPipeline(
        retriever=retriever,
        llm_client=llm_client,
        model_name=settings.groq_model_name,
        temperature=settings.generation_temperature,
    )


def _judge_model_vs_gold(
    llm_judge: LLMJudge,
    query: str,
    model_answer: str,
    gold_answer: str,
) -> GoldComparisonResponse:
    """Score model answer quality against a gold/reference answer."""
    system_prompt = """You are an expert evaluator comparing a model answer against a gold reference answer.

Evaluate semantic correctness and completeness with respect to the gold answer.
Do not require exact wording.

Scoring rubric:
- 5: Semantically equivalent to gold, complete, no important mistakes.
- 4: Mostly correct, minor omissions or phrasing differences.
- 3: Partially correct, notable missing points or slight inaccuracies.
- 2: Weak answer, major omissions or inaccuracies.
- 1: Incorrect or largely unrelated to gold answer.
"""

    user_prompt = f"""<query>
{query}
</query>

<gold_answer>
{gold_answer}
</gold_answer>

<model_answer>
{model_answer}
</model_answer>

Return JSON with:
- reasoning: concise explanation
- score: integer 1-5
"""

    return llm_judge.evaluate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_format=GoldComparisonResponse,
    )


@observe(name="evaluation-gold")
def main() -> None:
    """Run evaluation against gold answers."""
    parser = argparse.ArgumentParser(
        description="Evaluate model-generated answers against gold/reference answers"
    )
    parser.add_argument(
        "--examples-file",
        type=str,
        default="evaluations/evaluation_examples.json",
        help="Path to evaluation examples JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluations/results",
        help="Directory to save evaluation report",
    )
    parser.add_argument(
        "--collection",
        "-c",
        type=str,
        default=settings.chroma_collection_name,
        help=f"ChromaDB collection name (default: {settings.chroma_collection_name})",
    )
    parser.add_argument(
        "--retrieval",
        "-r",
        type=str,
        default="hybrid",
        choices=["dense", "bm25", "hybrid"],
        help="Retrieval method to use",
    )
    args = parser.parse_args()

    examples_file = Path(args.examples_file)
    output_dir = Path(args.output_dir)

    if not examples_file.exists():
        console.print(f"[red]Error: Examples file not found at {examples_file}[/]")
        return

    examples = load_examples(examples_file)
    if not examples:
        console.print("[red]Error: No examples found in input file[/]")
        return

    console.print(f"[bold blue]Loaded {len(examples)} examples[/]")
    console.print(
        f"[bold blue]Using collection={args.collection}, retrieval={args.retrieval}[/]"
    )

    rag_pipeline = _build_rag_pipeline(
        collection_name=args.collection,
        retrieval_method=args.retrieval,
    )

    llm_client = OpenAI(
        api_key=settings.groq_api_key,
        base_url=settings.groq_base_url,
    )
    llm_judge = LLMJudge(
        client=llm_client,
        model_name=settings.groq_judge_model_name or settings.groq_model_name,
        temperature=0.0,
    )

    results = []
    normalized_scores = []

    try:
        for idx, example in enumerate(examples, start=1):
            console.print(f"[cyan]Evaluating {idx}/{len(examples)}:[/] {example.query}")

            rag_response = rag_pipeline.query(example.query)
            model_answer = rag_response.answer
            retrieved_contexts = [source.text for source in rag_response.sources]

            judgment = _judge_model_vs_gold(
                llm_judge=llm_judge,
                query=example.query,
                model_answer=model_answer,
                gold_answer=example.answer,
            )

            normalized_score = (judgment.score - 1) / 4
            normalized_scores.append(normalized_score)

            results.append(
                {
                    "query": example.query,
                    "gold_answer": example.answer,
                    "model_answer": model_answer,
                    "score_1_to_5": judgment.score,
                    "score_0_to_1": normalized_score,
                    "reasoning": judgment.reasoning,
                    "retrieved_contexts": retrieved_contexts,
                    "trace_id": example.trace_id,
                }
            )

        average_score = sum(normalized_scores) / len(normalized_scores)

        report = {
            "timestamp": datetime.now().isoformat(),
            "num_examples": len(results),
            "model_name": settings.groq_model_name,
            "judge_model_name": settings.groq_judge_model_name or settings.groq_model_name,
            "collection": args.collection,
            "retrieval": args.retrieval,
            "metric": "gold_answer_alignment",
            "average_score_0_to_1": average_score,
            "results": results,
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"evaluation_gold_report_{timestamp}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        table = Table(title="Gold Alignment Summary", show_header=True, header_style="bold")
        table.add_column("#", justify="right")
        table.add_column("Score (0-1)", justify="right")
        table.add_column("Query", overflow="fold")

        for idx, item in enumerate(results, start=1):
            table.add_row(str(idx), f"{item['score_0_to_1']:.3f}", item["query"])

        console.print("\n")
        console.print(table)
        console.print(f"\n[bold green]Average gold alignment: {average_score:.3f}[/]")
        console.print(f"[bold green]✓ Report saved to {output_file}[/]")
    finally:
        flush_langfuse()


if __name__ == "__main__":
    main()
