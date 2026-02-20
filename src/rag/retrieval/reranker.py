"""Reranking retrieved documents using cross-encoder models."""

from sentence_transformers import CrossEncoder

from src.rag.retrieval.types import SearchResult


class Reranker:
    """Reranks search results using a cross-encoder model."""

    def __init__(self, model_name: str, top_k: int):
        """
        Initialize the reranker.

        Args:
            model_name: Name of the cross-encoder model to use
            top_k: Number of top results to return after reranking
        """
        self.model = CrossEncoder(model_name)
        self.top_k = top_k

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """
        Rerank search results using cross-encoder scoring.

        Args:
            query: Original search query
            results: List of search results from retrieval

        Returns:
            List of reranked results sorted by rerank_score (descending)
            Original scores are preserved in metadata['original_score']
        """
        if not results:
            return []

        # Prepare query-document pairs for cross-encoder
        pairs = [[query, result.text] for result in results]

        # Get cross-encoder scores
        scores = self.model.predict(pairs)

        # Create reranked results with original scores in metadata
        reranked = []
        for i, result in enumerate(results):
            # Preserve original score in metadata
            updated_metadata = {**result.metadata, "original_score": result.score}

            reranked.append(
                SearchResult(
                    doc_id=result.doc_id,
                    text=result.text,
                    metadata=updated_metadata,
                    score=float(scores[i]),  # Use rerank score as the new score
                )
            )

        # Sort by rerank score (descending) and take top_k
        reranked.sort(key=lambda x: x.score, reverse=True)

        return reranked[: self.top_k]
