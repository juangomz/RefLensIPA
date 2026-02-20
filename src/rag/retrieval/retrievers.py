"""Retrieval implementations for RAG system."""

from langfuse import observe
from rank_bm25 import BM25Okapi

from src.rag.models.embedding_model import EmbeddingModel
from src.rag.retrieval.fusion import reciprocal_rank_fusion
from src.rag.retrieval.types import Retriever, SearchResult
from src.rag.storage.vector_store import VectorStore


class DenseRetriever(Retriever):
    """Dense retrieval using semantic vector search."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        top_k: int,
    ):
        """
        Initialize dense retriever.

        Args:
            vector_store: Vector store for searching
            embedding_model: Model to embed queries
            top_k: Number of results to return
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.top_k = top_k

    @observe(name="retrieval-dense")
    def search(self, query: str) -> list[SearchResult]:
        """
        Search for documents using dense vector similarity.

        Args:
            query: Search query

        Returns:
            List of search results sorted by similarity (ascending distance)
        """
        # Embed query
        query_embedding = self.embedding_model.embed(query)

        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding.tolist(),
            top_k=self.top_k,
        )

        return results


class BM25Retriever(Retriever):
    """BM25-based sparse retrieval."""

    def __init__(self, vector_store: VectorStore, top_k: int):
        """
        Initialize BM25 retriever with vector store.

        Args:
            vector_store: Vector store to retrieve documents from
            top_k: Number of results to return
        """
        self.vector_store = vector_store
        self.top_k = top_k

        # Get all documents from vector store
        documents = vector_store.get_all_documents()
        self.doc_ids = [doc["id"] for doc in documents]
        self.doc_texts = [doc["text"] for doc in documents]
        self.doc_metadatas = [doc["metadata"] for doc in documents]

        # Tokenize documents (simple whitespace tokenization)
        tokenized_corpus = [doc.lower().split() for doc in self.doc_texts]

        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)

    @observe(name="retrieval-bm25")
    def search(self, query: str) -> list[SearchResult]:
        """
        Search for documents using BM25.

        Args:
            query: Search query

        Returns:
            List of BM25 search results sorted by score (descending)
        """
        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k results
        top_indices = scores.argsort()[-self.top_k :][::-1]

        results = [
            SearchResult(
                doc_id=self.doc_ids[idx],
                text=self.doc_texts[idx],
                metadata=self.doc_metadatas[idx],
                score=float(scores[idx]),
            )
            for idx in top_indices
        ]

        return results


class HybridRetriever(Retriever):
    """Combines dense and sparse retrieval using Reciprocal Rank Fusion."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: BM25Retriever,
        dense_weight: float,
        sparse_weight: float,
        final_top_k: int,
    ):
        """
        Initialize hybrid searcher.

        Args:
            dense_retriever: Dense retriever for semantic search (with its own top_k)
            sparse_retriever: Sparse retriever for keyword search (with its own top_k)
            dense_weight: Weight for dense retrieval results
            sparse_weight: Weight for sparse retrieval results
            final_top_k: Number of final results to return after fusion
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.final_top_k = final_top_k

    @observe(name="retrieval-hybrid")
    def search(self, query: str) -> list[SearchResult]:
        """
        Search using both dense and sparse retrieval, fused with RRF.

        Args:
            query: Search query

        Returns:
            Fused and re-ranked search results (top final_top_k)
        """
        # Retrieve from both methods (each uses its configured top_k)
        dense_results = self.dense_retriever.search(query)
        sparse_results = self.sparse_retriever.search(query)

        # Fuse results using weighted RRF
        fused_results = reciprocal_rank_fusion(
            result_lists=[dense_results, sparse_results],
            weights=[self.dense_weight, self.sparse_weight],
        )

        # Return top final_top_k results
        return fused_results[: self.final_top_k]
