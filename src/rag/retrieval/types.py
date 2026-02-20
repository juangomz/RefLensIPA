"""Common types for retrieval operations."""

from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel


class SearchResult(BaseModel):
    """Represents a search result from any retriever."""

    doc_id: str
    text: str
    metadata: dict[str, Any]
    score: float


class Retriever(ABC):
    """Abstract base class for all retrievers."""

    @abstractmethod
    def search(self, query: str) -> list[SearchResult]:
        """
        Search for relevant documents.

        Args:
            query: Search query

        Returns:
            List of search results
        """
        pass
