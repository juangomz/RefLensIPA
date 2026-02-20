# """Ingestion pipeline for processing documents."""

# import json
# from pathlib import Path

# from langfuse import observe
# from rich.console import Console
# from tqdm import tqdm

# from src.rag.ingestion.chunkers import Chunk, Chunker
# from src.rag.ingestion.loaders import get_loader
# from src.rag.models.embedding_model import EmbeddingModel
# from src.rag.storage.vector_store import VectorStore

# console = Console()


# class IngestionPipeline:
#     """Pipeline for ingesting documents into the vector store."""

#     def __init__(
#         self,
#         vector_store: VectorStore,
#         embedding_model: EmbeddingModel,
#         chunker: Chunker,
#     ):
#         """
#         Initialize the ingestion pipeline.

#         Args:
#             vector_store: Vector store to store chunks
#             embedding_model: Model to generate embeddings
#             chunker: Chunker instance for text chunking
#         """
#         self.vector_store = vector_store
#         self.embedding_model = embedding_model
#         self.chunker = chunker

#     @observe(name="ingest-directory")
#     def ingest_directory(
#         self,
#         directory: Path,
#         pattern: str = "**/*.pdf",
#         save_processed: bool = True,
#         processed_dir: Path | None = None,
#     ) -> None:
#         """
#         Ingest all documents from a directory.

#         Args:
#             directory: Directory containing documents
#             pattern: Glob pattern for matching files
#             save_processed: Whether to save processed chunks to disk
#             processed_dir: Directory to save processed chunks (if save_processed=True)
#         """
#         files = list(directory.glob(pattern))

#         if not files:
#             console.print(f"[yellow]No files found matching pattern: {pattern}[/]")
#             return

#         console.print(f"[bold blue]Found {len(files)} files to process[/]")
#         console.print(f"[bold blue]Chunker:[/] {self.chunker.__class__.__name__}\n")

#         all_chunks = []

#         # Load and chunk documents
#         console.print("[bold]Step 1: Loading and chunking documents...[/]")
#         for file_path in tqdm(files, desc="Processing documents"):
#             try:
#                 chunks = self._process_file(file_path)
#                 all_chunks.extend(chunks)
#             except Exception as e:
#                 console.print(f"[red]Error processing {file_path.name}:[/] {e}")

#         if not all_chunks:
#             console.print("[yellow]No chunks were created[/]")
#             return

#         console.print(f"[green]✓ Created {len(all_chunks)} chunks[/]\n")

#         # Save processed chunks if requested
#         if save_processed and processed_dir:
#             console.print("[bold]Step 2: Saving processed chunks...[/]")
#             self._save_chunks(all_chunks, processed_dir)
#             console.print(f"[green]✓ Saved to {processed_dir}[/]\n")

#         # Generate embeddings in batches
#         console.print(f"[bold]Step 3: Generating embeddings for {len(all_chunks)} chunks...[/]")
#         texts = [chunk.text for chunk in all_chunks]
#         embeddings = []
#         batch_size = 250

#         for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
#             batch = texts[i : i + batch_size]
#             batch_embeddings = self.embedding_model.embed_batch(batch, batch_size=batch_size)
#             embeddings.extend(batch_embeddings)

#         console.print(f"[green]✓ Generated {len(embeddings)} embeddings[/]\n")

#         # Store in vector database
#         console.print("[bold]Step 4: Storing in vector database...[/]")
#         ids = [f"chunk_{i}" for i in range(len(all_chunks))]
#         texts = [chunk.text for chunk in all_chunks]
#         metadatas = [
#             {
#                 "doc_id": chunk.doc_id,
#                 "chunk_id": chunk.chunk_id,
#                 "source": chunk.source,
#                 "timestamp": chunk.timestamp,
#             }
#             for chunk in all_chunks
#         ]
#         embeddings_list = [emb.tolist() for emb in embeddings]

#         self.vector_store.add(
#             ids=ids,
#             texts=texts,
#             embeddings=embeddings_list,
#             metadatas=metadatas,
#         )

#         console.print(f"[green]✓ Stored {len(all_chunks)} chunks in vector database[/]")
#         console.print(f"[bold green]Ingestion complete![/]")

#     def _process_file(self, file_path: Path) -> list[Chunk]:
#         """
#         Process a single file: load and chunk.

#         Args:
#             file_path: Path to the file

#         Returns:
#             List of chunks
#         """
#         # Load document
#         loader = get_loader(file_path)
#         document = loader.load(file_path)

#         # Chunk document
#         chunks = self.chunker.chunk(document.content, document.metadata)

#         return chunks

#     def _save_chunks(self, chunks: list[Chunk], output_dir: Path) -> None:
#         """
#         Save processed chunks to disk as JSON.

#         Args:
#             chunks: List of chunks to save
#             output_dir: Directory to save chunks
#         """
#         output_dir.mkdir(parents=True, exist_ok=True)

#         output_file = output_dir / "chunks.json"

#         chunks_data = [chunk.model_dump() for chunk in chunks]

#         with open(output_file, "w", encoding="utf-8") as f:
#             json.dump(chunks_data, f, ensure_ascii=False, indent=2)


"""Ingestion pipeline for processing documents."""

import json
from pathlib import Path

from langfuse import observe
from rich.console import Console
from tqdm import tqdm

from src.rag.ingestion.chunkers import Chunk, Chunker
from src.rag.ingestion.loaders import get_loader
from src.rag.models.embedding_model import EmbeddingModel
from src.rag.storage.vector_store import VectorStore
from scripts.ingest_chunks_to_neo4j import ingest as run_graph_ingestion
console = Console()


class IngestionPipeline:
    """Pipeline for ingesting documents into the vector store."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        chunker: Chunker,
    ):
        """
        Initialize the ingestion pipeline.

        Args:
            vector_store: Vector store to store chunks
            embedding_model: Model to generate embeddings
            chunker: Chunker instance for text chunking
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.chunker = chunker

    @observe(name="ingest-directory")
    def ingest_directory(
        self,
        directory: Path,
        pattern: str = "**/*.pdf",
        save_processed: bool = True,
        processed_dir: Path | None = None,
    ) -> None:
        """
        Ingest all documents from a directory.

        Args:
            directory: Directory containing documents
            pattern: Glob pattern for matching files
            save_processed: Whether to save processed chunks to disk
            processed_dir: Directory to save processed chunks (if save_processed=True)
        """
        files = list(directory.glob(pattern))

        if not files:
            console.print(f"[yellow]No files found matching pattern: {pattern}[/]")
            return

        console.print(f"[bold blue]Found {len(files)} files to process[/]")
        console.print(f"[bold blue]Chunker:[/] {self.chunker.__class__.__name__}\n")

        all_chunks = []

        # Load and chunk documents
        console.print("[bold]Step 1: Loading and chunking documents...[/]")
        for file_path in tqdm(files, desc="Processing documents"):
            try:
                chunks = self._process_file(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                console.print(f"[red]Error processing {file_path.name}:[/] {e}")

        if not all_chunks:
            console.print("[yellow]No chunks were created[/]")
            return

        console.print(f"[green]✓ Created {len(all_chunks)} chunks[/]\n")

        # Save processed chunks if requested
        if save_processed and processed_dir:
            console.print("[bold]Step 2: Saving processed chunks...[/]")
            self._save_chunks(all_chunks, processed_dir)
            console.print(f"[green]✓ Saved to {processed_dir}[/]\n")
            chunks_file = processed_dir / "chunks.json"
    
            # console.print("[bold]Step 2.5: Running Graph Agent extraction (Internal Call)...[/]")
            # try:
            #     # Call the function directly in the same process
            #     # This is faster and shares the same memory/logging
            #     run_graph_ingestion(chunks_path=str(chunks_file))
                
            #     console.print("[green]✓ Graph ingestion complete[/]\n")
            # except Exception as e:
            #     # In B, we catch actual Python exceptions, which is much more informative
            #     console.print(f"[red]Graph Agent failed:[/] {e}")

        # Generate embeddings in batches
        console.print(f"[bold]Step 3: Generating embeddings for {len(all_chunks)} chunks...[/]")
        texts = [chunk.text for chunk in all_chunks]
        embeddings = []
        batch_size = 250

        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i : i + batch_size]
            batch_embeddings = self.embedding_model.embed_batch(batch, batch_size=batch_size)
            embeddings.extend(batch_embeddings)

        console.print(f"[green]✓ Generated {len(embeddings)} embeddings[/]\n")

        # Store in vector database
        console.print("[bold]Step 4: Storing in vector database...[/]")
        # ids = [f"chunk_{i}" for i in range(len(all_chunks))]
        ids = [f"{chunk.doc_id}_{chunk.chunk_id}" for chunk in all_chunks]
        texts = [chunk.text for chunk in all_chunks]
        metadatas = [
            {
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "timestamp": chunk.timestamp,
            }
            for chunk in all_chunks
        ]
        embeddings_list = [emb.tolist() for emb in embeddings]
        #esto se puede hacer más eficiente metiéndolo dentro del loop
        self.vector_store.add(
            ids=ids,
            texts=texts,
            embeddings=embeddings_list,
            metadatas=metadatas,
        )

        console.print(f"[green]✓ Stored {len(all_chunks)} chunks in vector database[/]")
        console.print(f"[bold green]Ingestion complete![/]")
    

    def _process_file(self, file_path: Path) -> list[Chunk]:
        """
        Process a single file: load and chunk.

        Args:
            file_path: Path to the file

        Returns:
            List of chunks
        """
        # Load document
        loader = get_loader(file_path)
        document = loader.load(file_path)

        # Chunk document
        chunks = self.chunker.chunk(document.content, document.metadata)

        return chunks

    def _save_chunks(self, chunks: list[Chunk], output_dir: Path) -> None:
        """
        Save processed chunks to disk as JSON.

        Args:
            chunks: List of chunks to save
            output_dir: Directory to save chunks
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "chunks.json"

        chunks_data = [chunk.model_dump() for chunk in chunks]

        with open(output_file, "w") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=4)