"""RAG (Retrieval Augmented Generation) pipeline."""

from langfuse import observe
from langfuse.openai import OpenAI
from openai import RateLimitError
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.rag.retrieval.types import Retriever, SearchResult
from src.rag.utils import is_reasoning_model


class RAGResponse(BaseModel):
    """Response from the RAG pipeline."""

    query: str
    answer: str
    sources: list[SearchResult]
    prompt: str | None = None


class RAGPipeline:
    """RAG pipeline combining retrieval and generation."""

    def __init__(
        self,
        retriever: Retriever,
        llm_client: OpenAI,
        model_name: str,
        temperature: float,
    ):
        """
        Initialize RAG pipeline.

        Args:
            retriever: Retriever instance (configured with its own parameters)
            llm_client: LLM client for generation
            model_name: Model name for generation
            temperature: Temperature for LLM generation (0.0-2.0)
        """
        self.retriever = retriever
        self.llm_client = llm_client
        self.model_name = model_name
        self.temperature = temperature

    @observe(name="rag-query")
    def query(self, query: str, include_prompt: bool = False) -> RAGResponse:
        """
        Process a query through the RAG pipeline.

        Args:
            query: User query
            include_prompt: If True, include the full prompt sent to the model in the response

        Returns:
            RAG response with answer and sources
        """
        # Retrieve relevant documents
        results = self.retriever.search(query)

        # Generate answer using LLM
        answer, prompt = self._generate_answer(query, results, include_prompt=include_prompt)

        return RAGResponse(
            query=query,
            answer=answer,
            sources=results,
            prompt=prompt,
        )

    @observe(name="rag-generate")
    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential_jitter(initial=1, max=60, jitter=5),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _generate_answer(self, query: str, sources: list[SearchResult], include_prompt: bool = False) -> tuple[str, str | None]:
        """
        Generate answer using LLM with retrieved context.

        Args:
            query: User query
            sources: Retrieved source documents
            include_prompt: If True, return the full prompt sent to the model along with the answer

        Returns:
            Tuple of (generated answer, full prompt string or None)
        """
        # Build context from sources
        context_parts = []
        for i, source in enumerate(sources, 1):
            context_parts.append(f"[{i}] {source.text}")

        context = "\n\n".join(context_parts)

        # Create prompt
        system_message = (
            "You are a helpful assistant that answers questions based on the provided context. "
            "Use only the information from the context to answer the question. "
            "If the context doesn't contain enough information to answer the question, ask the user to ser for the information using external sources "
        )

        user_message = f"""Context:
{context}

Question: {query}

Answer:"""

        # Generate response
        # Reasoning models (o1, o3, gpt-5) don't support temperature parameter
        call_params = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
        }

        if not is_reasoning_model(self.model_name):
            call_params["temperature"] = self.temperature

        response = self.llm_client.chat.completions.create(**call_params)

        answer = response.choices[0].message.content or ""

        if include_prompt:
            full_prompt = f"SYSTEM:\n{system_message}\n\nUSER:\n{user_message}"
            return answer, full_prompt
        return answer, None
