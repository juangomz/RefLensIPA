"""Embedding generation using Azure OpenAI."""

import numpy as np
from langfuse.openai import AzureOpenAI
from openai import RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)


class EmbeddingModel:
    """Wrapper for Azure OpenAI embedding model."""

    def __init__(
        self,
        api_key: str,
        api_version: str,
        azure_endpoint: str,
        deployment_name: str,
    ):
        """
        Initialize the Azure OpenAI client.

        Args:
            api_key: Azure OpenAI API key
            api_version: Azure OpenAI API version
            azure_endpoint: Azure OpenAI endpoint URL
            deployment_name: Name of the embedding deployment
        """
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
        )
        self.deployment_name = deployment_name

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential_jitter(initial=1, max=60, jitter=5),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        response = self.client.embeddings.create(
            input=text,
            model=self.deployment_name,
        )

        embedding = response.data[0].embedding
        return np.array(embedding)

    def embed_batch(self, texts: list[str], batch_size: int = 250) -> list[np.ndarray]:
        """
        Generate embeddings for multiple texts in batch.

        Automatically splits large batches into smaller chunks to handle
        API limits and improve reliability.

        Args:
            texts: List of texts to embed
            batch_size: Maximum number of texts per API call (default: 250)

        Returns:
            List of embedding vectors

        Raises:
            ValueError: If batch_size <= 0
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        if not texts:
            return []

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = self._embed_batch_internal(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential_jitter(initial=1, max=60, jitter=5),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _embed_batch_internal(self, texts: list[str]) -> list[np.ndarray]:
        """
        Internal method to embed a single batch with retry logic.

        Args:
            texts: List of texts to embed (pre-batched)

        Returns:
            List of embedding vectors
        """
        response = self.client.embeddings.create(
            input=texts,
            model=self.deployment_name,
        )

        embeddings = [np.array(item.embedding) for item in response.data]
        return embeddings
