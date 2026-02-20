"""Script to create evaluation dataset by running queries through RAG pipeline."""

import argparse
import json
import random
from pathlib import Path

from langfuse import get_client, observe
from langfuse.openai import OpenAI
from rich.console import Console

from src.rag.config import settings
from src.rag.generation.rag import RAGPipeline
from src.rag.models.embedding_model import EmbeddingModel
from src.rag.observability.langfuse_client import configure_langfuse, flush_langfuse
from src.rag.retrieval.retrievers import DenseRetriever, HybridRetriever, BM25Retriever
from src.rag.storage.vector_store import VectorStore

console = Console()

# Configure LangFuse for @observe decorators
configure_langfuse()


@observe(name="eval-dataset-query")
def run_rag_query(rag_pipeline: RAGPipeline, query: str) -> dict:
    """
    Run a single query through the RAG pipeline.

    Args:
        rag_pipeline: RAG pipeline instance
        query: Query string

    Returns:
        Dictionary with query, answer, contexts, and trace_id
    """
    response = rag_pipeline.query(query)

    # Get trace ID to link evaluation scores later
    langfuse = get_client()
    trace_id = langfuse.get_current_trace_id()

    return {
        "query": query,
        "answer": response.answer,
        "contexts": [source.text for source in response.sources],
        "trace_id": trace_id,
    }


def load_queries(queries_file: Path) -> list[str]:
    """
    Load queries from JSON file.

    Expected format:
    {
        "queries": [
            "What is RAG?",
            "How does retrieval work?",
            ...
        ]
    }

    Args:
        queries_file: Path to JSON file with queries

    Returns:
        List of query strings
    """
    with open(queries_file, "r") as f:
        data = json.load(f)

    return data["queries"]


def main() -> None:
    """Create evaluation dataset by running queries through RAG pipeline."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Create evaluation dataset by running queries through RAG pipeline"
    )
    parser.add_argument(
        "--num-questions",
        "-n",
        type=int,
        default=None,
        help="Number of questions to sample from the pool (default: use all questions)",
    )
    parser.add_argument(
        "--queries-file",
        type=str,
        default="evaluations/evaluation_queries.json",
        help="Path to queries JSON file (default: evaluations/evaluation_queries.json)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="evaluations/evaluation_examples.json",
        help="Path to output file (default: evaluations/evaluation_examples.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling (default: None)",
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
        help="Retrieval method to use: dense, bm25, or hybrid (default: hybrid)",
    )
    args = parser.parse_args()

    # Configuration
    queries_file = Path(args.queries_file)
    output_file = Path(args.output_file)

    # Load queries
    console.print(f"[bold blue]Loading queries from {queries_file}...[/]")
    queries = load_queries(queries_file)
    console.print(f"[green]Loaded {len(queries)} queries[/]")

    # Sample queries if num_questions specified
    if args.num_questions is not None:
        if args.num_questions > len(queries):
            console.print(
                f"[yellow]Warning: Requested {args.num_questions} questions but only {len(queries)} available[/]"
            )
            console.print(f"[yellow]Using all {len(queries)} questions[/]\n")
        else:
            if args.seed is not None:
                random.seed(args.seed)
                console.print(f"[cyan]Using random seed: {args.seed}[/]")

            queries = random.sample(queries, args.num_questions)
            console.print(f"[cyan]Sampled {len(queries)} questions from pool[/]\n")
    else:
        console.print(f"[cyan]Using all {len(queries)} questions[/]\n")

    # Initialize RAG pipeline
    collection_name = args.collection
    console.print("[bold blue]Initializing RAG system...[/]")
    console.print(f"[bold blue]Using collection: {collection_name}[/]")

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

    # Initialize retriever based on selected method
    retrieval_method = args.retrieval.lower()
    console.print(f"[bold blue]Using retrieval method: {retrieval_method}[/]")

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
    else:  # hybrid
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

    rag_pipeline = RAGPipeline(
        retriever=retriever,
        llm_client=llm_client,
        model_name=settings.groq_model_name,
        temperature=settings.generation_temperature,
    )

    console.print("[green]RAG system initialized[/]\n")

    try:
        # Process queries
        console.print("[bold blue]Processing queries through RAG pipeline...[/]")
        evaluation_examples = []

        for i, query in enumerate(queries, 1):
            console.print(f"[cyan]Processing query {i}/{len(queries)}:[/] {query}")

            result = run_rag_query(rag_pipeline, query)
            evaluation_examples.append(result)

            console.print(f"[green]✓ Generated answer with {len(result['contexts'])} contexts[/]\n")

        # Save dataset
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(evaluation_examples, f, indent=2)

        console.print(f"[bold green]✓ Evaluation dataset saved to {output_file}[/]")
        console.print(f"[bold green]✓ {len(evaluation_examples)} examples created[/]\n")
    finally:
        # Flush LangFuse traces to ensure they are sent
        flush_langfuse()


if __name__ == "__main__":
    main()
