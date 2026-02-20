"""Vector store implementation using ChromaDB."""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from src.rag.retrieval.types import SearchResult


class VectorStore:
    """ChromaDB-based vector store for document chunks."""

    def __init__(self, persist_directory: str, collection_name: str = "documents"):
        """
        Initialize the vector store.

        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection to use
        """
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """
        Add documents to the vector store.

        Automatically batches large collections to respect ChromaDB's batch size limit.

        Args:
            ids: List of unique document IDs
            texts: List of document texts
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
        """
        # ChromaDB has a max batch size of ~5461
        batch_size = 5000
        total = len(ids)

        if total <= batch_size:
            # Single batch
            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        else:
            # Multiple batches
            for i in range(0, total, batch_size):
                end_idx = min(i + batch_size, total)
                self.collection.add(
                    ids=ids[i:end_idx],
                    documents=texts[i:end_idx],
                    embeddings=embeddings[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                )

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Search for similar documents using vector similarity.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            where: Optional metadata filter

        Returns:
            List of search results sorted by similarity (lower distance = more similar)
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )

        search_results = [
            SearchResult(
                doc_id=results["ids"][0][i],
                text=results["documents"][0][i],
                metadata=results["metadatas"][0][i],
                score=results["distances"][0][i],
            )
            for i in range(len(results["ids"][0]))
        ]

        return search_results

    def get_all_documents(self) -> list[dict[str, Any]]:
        """
        Get all documents from the collection.

        Returns:
            List of all documents with their metadata
        """
        results = self.collection.get()

        documents = []
        for i in range(len(results["ids"])):
            documents.append(
                {
                    "id": results["ids"][i],
                    "text": results["documents"][i],
                    "metadata": results["metadatas"][i],
                }
            )

        return documents

    def count(self) -> int:
        """
        Get the number of documents in the collection.

        Returns:
            Number of documents
        """
        return self.collection.count()

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.client.delete_collection(self.collection.name)

    def reset(self) -> None:
        """Reset the collection (delete all documents)."""
        self.delete_collection()
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )
