"""Interactive CLI Q&A interface for RAG system."""

import argparse

from langfuse.openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from src.rag.observability.langfuse_client import configure_langfuse, flush_langfuse
from src.rag.config import settings
from src.rag.generation.rag import RAGPipeline
from src.rag.models.embedding_model import EmbeddingModel
from src.rag.retrieval.retrievers import BM25Retriever, DenseRetriever, HybridRetriever
from src.rag.storage.vector_store import VectorStore

console = Console()

configure_langfuse()

class RAGQA:
    """Interactive Q&A interface using RAG pipeline. Each query is independent (no conversation memory)."""

    def __init__(self, rag_pipeline: RAGPipeline, verbose: bool = False):
        """
        Initialize Q&A interface.

        Args:
            rag_pipeline: RAG pipeline instance
            verbose: If True, show context sent to the model
        """
        self.rag_pipeline = rag_pipeline
        self.verbose = verbose

    def run(self):
        """Run the interactive Q&A loop."""
        self._print_welcome()

        try:
            while True:
                try:
                    # Get user input
                    query = Prompt.ask("\n[bold cyan]You[/]")

                    # Handle special commands
                    if query.lower() in ["/exit", "/quit", "exit", "quit"]:
                        console.print("\n[yellow]Goodbye![/]\n")
                        break

                    if query.lower() in ["/clear", "clear"]:
                        console.clear()
                        self._print_welcome()
                        continue

                    if query.lower() in ["/help", "help"]:
                        self._print_help()
                        continue

                    if not query.strip():
                        continue

                    # Process query through RAG pipeline
                    console.print("\n[dim]Thinking...[/]")

                    response = self.rag_pipeline.query(query, include_prompt=self.verbose)

                    # Display response
                    self._display_response(response)

                except KeyboardInterrupt:
                    console.print("\n\n[yellow]Goodbye![/]\n")
                    break
                except Exception as e:
                    console.print(f"\n[red]Error:[/] {e}\n")
        finally:
            # Flush LangFuse traces to ensure they are sent
            flush_langfuse()

    def _print_welcome(self):
        """Print welcome message."""
        welcome_text = """
# RAG Q&A System

Ask questions about the documents in the knowledge base.

**Note:** Each query is independent - there is no conversation memory.

Type `/help` for available commands, or `/exit` to quit.
"""
        console.print(Panel(Markdown(welcome_text), border_style="blue"))

    def _print_help(self):
        """Print help message."""
        help_table = Table(title="Available Commands", show_header=True)
        help_table.add_column("Command", style="cyan")
        help_table.add_column("Description")

        help_table.add_row("/help", "Show this help message")
        help_table.add_row("/clear", "Clear the screen")
        help_table.add_row("/exit", "Exit the Q&A system")

        console.print("\n")
        console.print(help_table)
        console.print("\n")

    def _display_response(self, response):
        """
        Display RAG response with answer and sources.

        Args:
            response: RAG response object
        """
        # Display prompt if verbose and available
        if self.verbose and response.prompt is not None:
            console.print("\n[bold cyan]Prompt sent to model[/]")
            console.print(Panel(response.prompt, border_style="cyan"))

        # Display sources
        if response.sources:
            console.print("\n[bold yellow]Sources:[/]")

            sources_table = Table(show_header=True, header_style="bold yellow")
            sources_table.add_column("#", style="cyan", width=4)
            sources_table.add_column("Score", justify="right", width=8)
            sources_table.add_column("Text", max_width=80)
            sources_table.add_column("Source", style="dim")

            ordered_sources = sorted(
                response.sources, key=lambda s: s.score, reverse=True
            )
            for i, source in enumerate(ordered_sources, 1):
                # Extract source information from metadata
                source_name = source.metadata.get("filename", "Unknown")
                page = source.metadata.get("page", "")
                source_info = f"{source_name}"
                if page:
                    source_info += f" (p.{page})"

                # Truncate text for display
                text_preview = source.text[:200] + "..." if len(source.text) > 200 else source.text

                sources_table.add_row(
                    str(i),
                    f"{source.score:.3f}",
                    text_preview,
                    source_info,
                )

            console.print(sources_table)
            console.print()

        # Display answer
        console.print("\n[bold green]Assistant[/]")
        console.print(Panel(Markdown(response.answer), border_style="green"))


def main() -> None:
    """Main entry point for the Q&A interface."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Interactive Q&A interface for RAG system"
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
        "--retriever",
        "-r",
        type=str,
        default="hybrid",
        choices=["dense", "bm25", "hybrid"],
        help="Retrieval method to use (default: hybrid)",
    )
    args = parser.parse_args()

    collection_name = args.collection
    verbose = args.verbose
    retriever_type = args.retriever

    console.print("[bold blue]Initializing RAG system...[/]")
    console.print(f"[bold blue]Using collection: {collection_name}[/]")

    if verbose:
        console.print(f"[dim]Persist directory: {settings.chroma_persist_directory}[/]")
        console.print(f"[dim]Retrieval top_k: {settings.retrieval_dense_top_k}[/]")
        console.print(f"[dim]Model: {settings.groq_model_name}[/]")
        console.print(f"[dim]Temperature: {settings.generation_temperature}[/]")

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

    # Initialize retriever based on type
    if retriever_type == "dense":
        retriever = DenseRetriever(
            vector_store=vector_store,
            embedding_model=embedding_model,
            top_k=settings.retrieval_dense_top_k,
        )
        if verbose:
            console.print(f"[dim]Retriever: dense (top_k={settings.retrieval_dense_top_k})[/]")
    elif retriever_type == "bm25":
        retriever = BM25Retriever(
            vector_store=vector_store,
            top_k=settings.retrieval_sparse_top_k,
        )
        if verbose:
            console.print(f"[dim]Retriever: bm25 (top_k={settings.retrieval_sparse_top_k})[/]")
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
        if verbose:
            console.print(f"[dim]Retriever: hybrid (dense_weight={settings.hybrid_search_dense_weight}, sparse_weight={settings.hybrid_search_sparse_weight}, final_top_k={settings.retrieval_hybrid_final_top_k})[/]")

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

    # Start Q&A interface
    qa = RAGQA(rag_pipeline, verbose=verbose)
    qa.run()


if __name__ == "__main__":
    main()
