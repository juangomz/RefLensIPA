"""Reciprocal Rank Fusion for combining multiple retrieval results."""

from collections import defaultdict

from src.rag.retrieval.types import SearchResult


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    weights: list[float],
    k: int = 60,
) -> list[SearchResult]:
    """
    Combine multiple ranked lists using weighted Reciprocal Rank Fusion.

    Weighted RRF formula: RRF_score(d) = Σ w_i / (k + rank_i(d))

    where:
    - d is a document
    - rank_i(d) is the rank of document d in list i (1-indexed)
    - w_i is the weight for list i
    - k is a constant (default 60)

    Args:
        result_lists: List of ranked search result lists
        weights: List of weights for each result list (must sum to 1.0)
        k: Constant for RRF formula (default 60)

    Returns:
        Fused and re-ranked list of search results sorted by RRF score (descending)

    Raises:
        ValueError: If number of weights doesn't match number of result lists
    """
    if not result_lists:
        return []

    if len(result_lists) != len(weights):
        raise ValueError(
            f"Number of weights ({len(weights)}) must match number of result lists ({len(result_lists)})"
        )

    # Dictionary to accumulate RRF scores: doc_id -> score
    # defaultdict automatically initializes missing keys with 0.0
    rrf_scores = defaultdict(float)

    # Dictionary to store document details: doc_id -> SearchResult
    doc_map = {}

    # Process each result list with its weight
    for weight, results in zip(weights, result_lists):
        for rank, result in enumerate(results, start=1):
            doc_id = result.doc_id

            # Calculate weighted RRF contribution from this ranking
            rrf_scores[doc_id] += weight / (k + rank)

            # Store document (will keep the version from first list if duplicate)
            if doc_id not in doc_map:
                doc_map[doc_id] = result

    # Create fused results with RRF scores
    fused_results = [
        SearchResult(
            doc_id=doc_id,
            text=doc_map[doc_id].text,
            metadata=doc_map[doc_id].metadata,
            score=rrf_score,
        )
        for doc_id, rrf_score in rrf_scores.items()
    ]

    # Sort by RRF score (descending)
    fused_results.sort(key=lambda x: x.score, reverse=True)

    return fused_results
