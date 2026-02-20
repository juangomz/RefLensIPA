"""Script to ingest documents into the vector store."""

import argparse
from pathlib import Path

from langfuse import observe
from rich.console import Console

from rag.config import settings
from rag.ingestion.chunkers import FixedSizeChunker, RecursiveChunker, SemanticChunker
from rag.ingestion.pipeline import IngestionPipeline
from rag.models.embedding_model import EmbeddingModel
from rag.observability.langfuse_client import configure_langfuse, flush_langfuse
from rag.storage.vector_store import VectorStore

console = Console()

# Configure LangFuse for @observe decorators
configure_langfuse()


def main() -> None:
    """Run the ingestion pipeline."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Ingest documents into the vector store"
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
    args = parser.parse_args()

    collection_name = args.collection
    verbose = args.verbose

    console.print(f"[bold blue]Using collection: {collection_name}[/]")

    if verbose:
        console.print(f"[dim]Chunk strategy: {settings.ingestion_chunk_strategy}[/]")
        console.print(f"[dim]Chunk size: {settings.ingestion_chunk_size}[/]")
        console.print(f"[dim]Chunk overlap: {settings.ingestion_chunk_overlap}[/]")
        console.print(f"[dim]Data directory: {settings.ingestion_data_dir}[/]")
        console.print(f"[dim]File pattern: {settings.ingestion_file_pattern}[/]")
        console.print(f"[dim]Persist directory: {settings.chroma_persist_directory}[/]")

    # Initialize embedding model
    embedding_model = EmbeddingModel(
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_endpoint=settings.azure_openai_endpoint,
        deployment_name=settings.azure_openai_embedding_deployment_name,
    )

    # Initialize vector store
    vector_store = VectorStore(
        persist_directory=settings.chroma_persist_directory,
        collection_name=collection_name,
    )
    print(settings.ingestion_chunk_strategy)
    # Create chunker based on strategy
    if settings.ingestion_chunk_strategy == "fixed":
        chunker = FixedSizeChunker(
            
            chunk_size=settings.ingestion_chunk_size,
            chunk_overlap=settings.ingestion_chunk_overlap,
        )
    elif settings.ingestion_chunk_strategy == "recursive":
        chunker = RecursiveChunker(
            chunk_size=settings.ingestion_chunk_size,
            chunk_overlap=settings.ingestion_chunk_overlap,
        )
    elif settings.ingestion_chunk_strategy == "semantic":
        chunker = SemanticChunker(
            embedding_model=embedding_model,
            max_chunk_size=settings.ingestion_chunk_size,
            similarity_threshold=settings.ingestion_similarity_threshold,
        )
    else:
        raise ValueError(f"Unknown chunking strategy: {settings.ingestion_chunk_strategy}")

    # Create ingestion pipeline
    pipeline = IngestionPipeline(
        vector_store=vector_store,
        embedding_model=embedding_model,
        chunker=chunker,
    )

    try:  
        # Run ingestion
        pipeline.ingest_directory(
            directory=Path(settings.ingestion_data_dir),
            pattern=settings.ingestion_file_pattern,
            save_processed=settings.ingestion_save_processed,
            processed_dir=(
                Path(settings.ingestion_processed_dir)
                if settings.ingestion_save_processed
                else None
            ),
        )

        if verbose:
            doc_count = vector_store.count()
            console.print(f"[dim]Total documents in collection: {doc_count}[/]")
    finally:
        # Flush LangFuse traces to ensure they are sent
        flush_langfuse()


if __name__ == "__main__":
    main()
