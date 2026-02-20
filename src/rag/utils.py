"""Shared utility functions."""

import numpy as np


def is_reasoning_model(model_name: str) -> bool:
    """
    Check if the model is a reasoning model that doesn't support temperature.

    Args:
        model_name: Name of the model/deployment

    Returns:
        True if it's a reasoning model (o1, o3, gpt-5, etc.)
    """
    reasoning_prefixes = ("o1", "o3", "gpt-5")
    return any(model_name.lower().startswith(prefix) for prefix in reasoning_prefixes)


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Cosine similarity score between -1 and 1
    """
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


