"""CLI for testing different retrieval methods."""

import argparse

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from src.rag.config import settings
from src.rag.models.embedding_model import EmbeddingModel
from src.rag.observability.langfuse_client import configure_langfuse, flush_langfuse
from src.rag.retrieval.retrievers import BM25Retriever, DenseRetriever, HybridRetriever
from src.rag.storage.vector_store import VectorStore

console = Console()

configure_langfuse()


def create_retriever(
    method: str,
    vector_store: VectorStore,
    embedding_model: EmbeddingModel,
    dense_weight: float | None = None,
    sparse_weight: float | None = None,
):
    """
    Create a retriever based on the specified method.

    Args:
        method: Retrieval method (dense, bm25, hybrid)
        vector_store: Vector store instance
        embedding_model: Embedding model instance
        dense_weight: Weight for dense retrieval (hybrid only, uses settings if not provided)
        sparse_weight: Weight for sparse retrieval (hybrid only, uses settings if not provided)

    Returns:
        Configured retriever instance
    """
    if method == "dense":
        return DenseRetriever(
            vector_store=vector_store,
            embedding_model=embedding_model,
            top_k=settings.retrieval_dense_top_k,
        )
    elif method == "bm25":
        return BM25Retriever(
            vector_store=vector_store,
            top_k=settings.retrieval_sparse_top_k,
        )
    elif method == "hybrid":
        dense_retriever = DenseRetriever(
            vector_store=vector_store,
            embedding_model=embedding_model,
            top_k=settings.retrieval_dense_top_k,
        )
        sparse_retriever = BM25Retriever(
            vector_store=vector_store,
            top_k=settings.retrieval_sparse_top_k,
        )
        return HybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            dense_weight=dense_weight if dense_weight is not None else settings.hybrid_search_dense_weight,
            sparse_weight=sparse_weight if sparse_weight is not None else settings.hybrid_search_sparse_weight,
            final_top_k=settings.retrieval_hybrid_final_top_k,
        )
    else:
        raise ValueError(f"Unknown retrieval method: {method}")


def display_results(results, method: str):
    """
    Display search results in a formatted table.

    Args:
        results: List of SearchResult objects
        method: Retrieval method used
    """
    if not results:
        console.print("[yellow]No results found.[/]")
        return

    table = Table(
        title=f"Results ({method})",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("#", style="cyan", width=4)
    table.add_column("Score", justify="right", width=10)
    table.add_column("Text", max_width=80)
    table.add_column("Source", style="dim", max_width=30)

    for i, result in enumerate(results, 1):
        source_name = result.metadata.get("filename", "Unknown")
        page = result.metadata.get("page", "")
        source_info = source_name
        if page:
            source_info += f" (p.{page})"

        text_preview = (
            result.text[:200] + "..." if len(result.text) > 200 else result.text
        )

        table.add_row(
            str(i),
            f"{result.score:.4f}",
            text_preview,
            source_info,
        )

    console.print(table)
    console.print()


def run_interactive(retriever, method: str):
    """
    Run interactive retrieval loop.

    Args:
        retriever: Configured retriever instance
        method: Retrieval method name for display
    """
    welcome_text = f"""
# Retrieval Tester

Testing **{method}** retrieval method.

Enter queries to test retrieval. Type `/help` for commands or `/exit` to quit.
"""
    console.print(Panel(welcome_text, border_style="blue"))

    while True:
        try:
            query = Prompt.ask("\n[bold cyan]Query[/]")

            if query.lower() in ["/exit", "/quit", "exit", "quit"]:
                console.print("\n[yellow]Goodbye![/]\n")
                break

            if query.lower() in ["/help", "help"]:
                help_table = Table(title="Available Commands", show_header=True)
                help_table.add_column("Command", style="cyan")
                help_table.add_column("Description")
                help_table.add_row("/help", "Show this help message")
                help_table.add_row("/exit", "Exit the retrieval tester")
                console.print("\n")
                console.print(help_table)
                continue

            if not query.strip():
                continue

            console.print("\n[dim]Searching...[/]")
            results = retriever.search(query)
            display_results(results, method)

        except KeyboardInterrupt:
            console.print("\n\n[yellow]Goodbye![/]\n")
            break
        except Exception as e:
            console.print(f"\n[red]Error:[/] {e}\n")


def run_single_query(retriever, query: str, method: str):
    """
    Run a single query and display results.

    Args:
        retriever: Configured retriever instance
        query: Search query
        method: Retrieval method name for display
    """
    console.print(f"\n[bold]Query:[/] {query}")
    console.print(f"[bold]Method:[/] {method}\n")
    console.print("[dim]Searching...[/]")

    results = retriever.search(query)
    display_results(results, method)


def main() -> None:
    """Main entry point for the retrieval CLI."""
    parser = argparse.ArgumentParser(
        description="Test different retrieval methods (dense, bm25, hybrid)"
    )
    parser.add_argument(
        "method",
        choices=["dense", "bm25", "hybrid"],
        help="Retrieval method to use",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="Single query to run (if not provided, runs interactive mode)",
    )
    parser.add_argument(
        "--collection",
        "-c",
        type=str,
        default=settings.chroma_collection_name,
        help=f"ChromaDB collection name (default: {settings.chroma_collection_name})",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        help="Weight for dense retrieval in hybrid mode (overrides settings)",
    )
    parser.add_argument(
        "--sparse-weight",
        type=float,
        help="Weight for sparse retrieval in hybrid mode (overrides settings)",
    )
    args = parser.parse_args()

    method = args.method
    collection_name = args.collection
    verbose = args.verbose
    dense_weight = args.dense_weight
    sparse_weight = args.sparse_weight

    console.print(f"[bold blue]Initializing {method} retriever...[/]")
    console.print(f"[bold blue]Using collection: {collection_name}[/]")

    if verbose:
        console.print(f"[dim]Persist directory: {settings.chroma_persist_directory}[/]")
        if method in ["dense", "hybrid"]:
            console.print(f"[dim]Dense top_k: {settings.retrieval_dense_top_k}[/]")
        if method in ["bm25", "hybrid"]:
            console.print(f"[dim]Sparse top_k: {settings.retrieval_sparse_top_k}[/]")
        if method == "hybrid":
            console.print(
                f"[dim]Hybrid final top_k: {settings.retrieval_hybrid_final_top_k}[/]"
            )
            # Show which weights are being used (CLI override or settings default)
            actual_dense_weight = dense_weight if dense_weight is not None else settings.hybrid_search_dense_weight
            actual_sparse_weight = sparse_weight if sparse_weight is not None else settings.hybrid_search_sparse_weight
            dense_source = " (CLI override)" if dense_weight is not None else " (from settings)"
            sparse_source = " (CLI override)" if sparse_weight is not None else " (from settings)"
            console.print(f"[dim]Dense weight: {actual_dense_weight}{dense_source}[/]")
            console.print(f"[dim]Sparse weight: {actual_sparse_weight}{sparse_source}[/]")

    # Initialize components
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

    if verbose:
        doc_count = vector_store.count()
        console.print(f"[dim]Documents in collection: {doc_count}[/]")

    retriever = create_retriever(method, vector_store, embedding_model, dense_weight, sparse_weight)

    console.print("[green]✓ Retriever initialized[/]\n")

    try:
        if args.query:
            run_single_query(retriever, args.query, method)
        else:
            run_interactive(retriever, method)
    finally:
        # Flush LangFuse traces to ensure they are sent
        flush_langfuse()


if __name__ == "__main__":
    main()
