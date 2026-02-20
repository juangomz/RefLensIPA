"""Script to evaluate RAG system on test examples."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from langfuse import observe
from langfuse.openai import OpenAI
from rich.console import Console

from src.rag.config import settings
from src.rag.evaluation.evaluator import EvaluationExample, RAGEvaluator
from src.rag.evaluation.llm_judge import LLMJudge
from src.rag.evaluation.metrics import (
    AnswerRelevanceMetric,
    ContextRelevanceMetric,
    FaithfulnessMetric,
)
from src.rag.observability.langfuse_client import configure_langfuse, flush_langfuse

console = Console()

# Configure LangFuse for @observe decorators
configure_langfuse()


def load_evaluation_examples(examples_file: Path) -> list[EvaluationExample]:
    """
    Load evaluation examples from JSON file.

    Expected format:
    [
        {
            "query": "What is RAG?",
            "answer": "RAG stands for...",
            "contexts": ["Context 1", "Context 2"],
            "trace_id": "trace_123" (optional)
        },
        ...
    ]

    Args:
        examples_file: Path to JSON file with evaluation examples

    Returns:
        List of EvaluationExample instances
    """
    with open(examples_file, "r") as f:
        data = json.load(f)

    return [EvaluationExample(**example) for example in data]


@observe(name="evaluation")
def main() -> None:
    """Run evaluation on examples."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate RAG system on test examples"
    )
    parser.add_argument(
        "--examples-file",
        type=str,
        default="evaluations/evaluation_examples.json",
        help="Path to evaluation examples JSON file (default: evaluations/evaluation_examples.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluations/results",
        help="Directory to save evaluation reports (default: evaluations/results)",
    )
    args = parser.parse_args()

    # Configuration
    examples_file = Path(args.examples_file)
    output_dir = Path(args.output_dir)

    if not examples_file.exists():
        console.print(f"[red]Error: Examples file not found at {examples_file}[/]")
        console.print("\n[yellow]Create a JSON file with evaluation examples:[/]")
        console.print(
            """
[
    {
        "query": "What is retrieval augmented generation?",
        "answer": "RAG is a technique that combines...",
        "contexts": [
            "Context paragraph 1...",
            "Context paragraph 2..."
        ],
        "trace_id": "trace_123"
    }
]
"""
        )
        return

    try:
        # Load evaluation examples
        console.print(f"[bold blue]Loading evaluation examples from {examples_file}...[/]")
        evaluation_examples = load_evaluation_examples(examples_file)
        console.print(f"[green]Loaded {len(evaluation_examples)} evaluation examples[/]\n")

        # Initialize LLM judge
        llm_client = OpenAI(
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
        )

        llm_judge = LLMJudge(
            client=llm_client,
            model_name=settings.groq_judge_model_name or settings.groq_model_name,
            temperature=0.0,
        )

        # Initialize metrics (RAG Triad)
        metrics = [
            FaithfulnessMetric(llm_judge),
            AnswerRelevanceMetric(llm_judge),
            ContextRelevanceMetric(llm_judge),
        ]

        # Create evaluator
        evaluator = RAGEvaluator(metrics=metrics)

        # Run evaluation (scores are automatically sent to LangFuse)
        report = evaluator.evaluate(evaluation_examples=evaluation_examples)

        # Save report to file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"evaluation_report_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(report.model_dump(), f, indent=2)

        console.print(f"\n[bold green]✓ Evaluation report saved to {output_file}[/]\n")
    finally:
        # Flush LangFuse traces to ensure they are sent
        flush_langfuse()


if __name__ == "__main__":
    main()
