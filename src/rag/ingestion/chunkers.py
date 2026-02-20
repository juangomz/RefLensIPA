"""Text chunking strategies for document processing."""

import re
from abc import ABC, abstractmethod
from typing import Any, Callable
from datetime import datetime
import numpy as np
from pydantic import BaseModel, Field, ConfigDict

from src.rag.models.embedding_model import EmbeddingModel
from src.rag.utils import cosine_similarity


class Chunk(BaseModel):
    """Represents a text chunk with metadata."""

    text: str
    metadata: dict[str, str | int | float]
    chunk_index: int

class Chunk(BaseModel):
    """Single text chunk to be converted into a knowledge graph update."""
    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(..., description="Document identifier (stable across runs)")
    chunk_id: str = Field(..., description="Chunk identifier unique within doc_id")
    text: str = Field(..., description="Chunk text content")
    source: str = Field(..., description="Source filename or URL")
    timestamp: str = Field(..., description="ISO date or datetime string when chunk was produced")
class Chunker(ABC):
    """Abstract base class for text chunkers."""

    @abstractmethod
    def chunk(self, text: str, metadata: dict[str, Any]) -> list[Chunk]:
        """
        Split text into chunks.

        Args:
            text: Text to chunk
            metadata: Metadata to attach to chunks

        Returns:
            List of chunks
        """
        pass


class FixedSizeChunker(Chunker):
    """Chunks text into fixed-size segments with overlap."""

    def __init__(self, chunk_size: int, chunk_overlap: int):
        """
        Initialize the fixed-size chunker.

        Args:
            chunk_size: Maximum number of characters per chunk
            chunk_overlap: Number of characters to overlap between chunks

        Raises:
            ValueError: If chunk_size <= 0 or chunk_overlap >= chunk_size
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, metadata: dict[str, Any]) -> list[Chunk]:
        """
        Split text into fixed-size chunks with overlap.

        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk

        Returns:
            List of chunks
        """
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            if chunk_text.strip():
                chunk_metadata = {
                    **metadata,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "start_index": start,
                    "end_index": end,
                    "chunking_strategy": "fixed",
                }

                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata=chunk_metadata,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

            start += self.chunk_size - self.chunk_overlap

        return chunks


class RecursiveChunker(Chunker):
    """
    Chunks text recursively by trying different separators.

    Tries to split by paragraphs first, then sentences, then words.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int):
        """
        Initialize the recursive chunker.

        Args:
            chunk_size: Target number of characters per chunk
            chunk_overlap: Number of characters to overlap between chunks

        Raises:
            ValueError: If chunk_size <= 0 or chunk_overlap >= chunk_size
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = [
            "\n\n",  # Paragraph breaks
            "\n",  # Line breaks
            ". ",  # Sentences
            " ",  # Words
            "",  # Characters
        ]

    def chunk(self, text: str, metadata: dict[str, Any]) -> list[Chunk]:
        """
        Split text into chunks recursively.

        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk

        Returns:
            List of chunks
        """
        chunks = self._split_text(text)
        doc_id = metadata.get("doc_id")

        result = []
        for i, chunk_text in enumerate(chunks):
            if chunk_text.strip():
                chunk_metadata = {
                    **metadata,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "chunking_strategy": "recursive",
                }

                result.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=f"c_{i}", # Unique within doc_id
                    text=chunk_text,
                    source=metadata.get("source"),
                    timestamp=datetime.now().isoformat()
                )
            )
            

        return result

    def _split_text(self, text: str) -> list[str]:
        """Recursively split text using different separators."""
        final_chunks = []

        # Try each separator in order
        for separator in self.separators:
            if separator == "":
                # Last resort: split by characters
                return self._split_by_chars(text)

            splits = text.split(separator) if separator else [text]

            current_chunk = []
            current_size = 0

            for split in splits:
                split_size = len(split)

                # If single split is too large, recursively split it
                if split_size > self.chunk_size:
                    if current_chunk:
                        final_chunks.append(separator.join(current_chunk))
                        current_chunk = []
                        current_size = 0

                    # Try next separator
                    next_sep_idx = self.separators.index(separator) + 1
                    if next_sep_idx < len(self.separators):
                        chunker = RecursiveChunker(self.chunk_size, self.chunk_overlap)
                        chunker.separators = self.separators[next_sep_idx:]
                        sub_chunks = chunker._split_text(split)
                        final_chunks.extend(sub_chunks)
                    continue

                # If adding this split exceeds chunk size, start new chunk
                if current_size + split_size > self.chunk_size and current_chunk:
                    final_chunks.append(separator.join(current_chunk))

                    # Handle overlap
                    if self.chunk_overlap > 0 and current_chunk:
                        overlap_text = separator.join(current_chunk)
                        overlap_text = overlap_text[-self.chunk_overlap :]
                        current_chunk = [overlap_text]
                        current_size = len(overlap_text)
                    else:
                        current_chunk = []
                        current_size = 0

                current_chunk.append(split)
                current_size += split_size + len(separator)

            # Add remaining chunk
            if current_chunk:
                final_chunks.append(separator.join(current_chunk))

            # If we successfully chunked everything, return
            if all(len(chunk) <= self.chunk_size * 1.1 for chunk in final_chunks):
                return final_chunks

        return final_chunks

    def _split_by_chars(self, text: str) -> list[str]:
        """Split text into character-based chunks as last resort."""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks


class SemanticChunker(Chunker):
    """
    Chunks text based on semantic similarity between sentences.

    This implementation uses consecutive sentence similarity to identify
    topic boundaries. Alternative: could compute similarity between grouped
    sentences (current chunk vs next sentence) for more context-aware chunking.

    This chunker:
    1. Splits text into sentences
    2. Computes embeddings for each sentence
    3. Finds natural breakpoints where semantic similarity drops
    4. Groups sentences into chunks based on these breakpoints
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        max_chunk_size: int,
        similarity_threshold: float = 0.5,
    ):
        """
        Initialize the semantic chunker.

        Args:
            embedding_model: EmbeddingModel instance for computing embeddings in batch
            max_chunk_size: Maximum number of characters per chunk
            similarity_threshold: Threshold for semantic similarity (0-1)
                Lower values create more chunks (stricter breakpoints)

        Raises:
            ValueError: If max_chunk_size <= 0 or similarity_threshold not in [0,1]
        """
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be greater than 0")
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")

        self.embedding_model = embedding_model
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold

    def chunk(self, text: str, metadata: dict[str, Any]) -> list[Chunk]:
        """
        Split text into semantically coherent chunks.

        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk

        Returns:
            List of semantically coherent chunks
        """
        # Split into sentences
        sentences = self._split_into_sentences(text)
        doc_id = metadata.get("doc_id")
        if not sentences:
            return []

        # If only one sentence, return as single chunk
        if len(sentences) == 1:
            chunk_metadata = {
                **metadata,
                "chunking_strategy": "semantic",
                "similarity_threshold": self.similarity_threshold,
            }
            return [
                Chunk(
                    text=sentences[0],
                    metadata=chunk_metadata,
                    chunk_index=0,
                )
            ]

        # Compute embeddings for all sentences in batch
        embeddings = self.embedding_model.embed_batch(sentences)

        # Calculate similarity between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # Find breakpoints where similarity drops below threshold
        breakpoints = [0]  # Start of first chunk
        current_chunk_size = len(sentences[0])

        for i, similarity in enumerate(similarities):
            sentence_length = len(sentences[i + 1])

            # Create breakpoint if:
            # 1. Similarity drops below threshold, OR
            # 2. Adding next sentence would exceed max_chunk_size
            if (
                similarity < self.similarity_threshold
                or current_chunk_size + sentence_length > self.max_chunk_size
            ):
                breakpoints.append(i + 1)
                current_chunk_size = sentence_length
            else:
                current_chunk_size += sentence_length

        # Add final breakpoint
        if breakpoints[-1] != len(sentences):
            breakpoints.append(len(sentences))

        # Create chunks from breakpoints
        chunks = []
        for i in range(len(breakpoints) - 1):
            start_idx = breakpoints[i]
            end_idx = breakpoints[i + 1]

            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = " ".join(chunk_sentences)

            # Note: Average similarity within chunks could be calculated here
            # for analysis purposes (e.g., to measure chunk coherence)

            chunk_metadata = {
                **metadata,
                "chunking_strategy": "semantic",
                "similarity_threshold": self.similarity_threshold,
                "num_sentences": end_idx - start_idx,
            }

            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=f"c_{i}", # Unique within doc_id
                    text=chunk_text,
                    source=metadata.get("source"),
                    timestamp=datetime.now().isoformat()
                )
            )
            
                
        

        return chunks

    def _split_into_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences using regex.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Pattern to split on sentence-ending punctuation followed by whitespace
        # Handles common abbreviations like "Dr.", "Mr.", "etc."
        pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+'

        sentences = re.split(pattern, text)

        # Clean and filter empty sentences
        sentences = [sent.strip() for sent in sentences if sent.strip()]

        return sentences