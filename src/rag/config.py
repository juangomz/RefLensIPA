"""Configuration management for the RAG Lab project."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from dotenv import load_dotenv
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Groq LLM Configuration
    groq_api_key: str = Field(..., description="Groq API key")
    groq_base_url: str = Field(
        "https://api.groq.com/openai/v1",
        description="Groq OpenAI-compatible base URL",
    )
    groq_model_name: str = Field(
        ..., description="Groq model name for RAG generation"
    )
    groq_judge_model_name: str | None = Field(
        None,
        description="Groq model name for evaluation judge (defaults to GROQ_MODEL_NAME)",
    )

    # Azure OpenAI Embeddings Configuration
    azure_openai_endpoint: str = Field(
        ..., description="Azure OpenAI endpoint URL for embeddings"
    )
    azure_openai_api_key: str = Field(
        ..., description="Azure OpenAI API key for embeddings"
    )
    azure_openai_api_version: str = Field(
        ..., description="Azure OpenAI API version for embeddings"
    )
    azure_openai_embedding_deployment_name: str = Field(
        ..., description="Azure OpenAI embedding deployment name"
    )

    # LangFuse Configuration
    langfuse_public_key: str = Field(..., description="LangFuse public key")
    langfuse_secret_key: str = Field(..., description="LangFuse secret key")
    langfuse_base_url: str = Field(..., description="LangFuse host URL")

    # ChromaDB Configuration
    chroma_persist_directory: str = Field(
        ..., description="ChromaDB persistence directory"
    )
    chroma_collection_name: str = Field(
        "kb_chunks", description="ChromaDB collection name"
    )

    # Retrieval Configuration
    retrieval_dense_top_k: int = Field(..., description="Number of documents to retrieve from dense retriever")
    retrieval_sparse_top_k: int = Field(..., description="Number of documents to retrieve from sparse retriever")
    retrieval_hybrid_final_top_k: int = Field(..., description="Number of final documents to return after hybrid fusion")
    rerank_top_k: int = Field(..., description="Number of documents after reranking")
    reranker_model_name: str = Field(
        ..., description="Cross-encoder model name for reranking"
    )
    hybrid_search_dense_weight: float = Field(
        ..., description="Weight for dense retrieval in hybrid search"
    )
    hybrid_search_sparse_weight: float = Field(
        ..., description="Weight for sparse retrieval in hybrid search"
    )

    # Ingestion Configuration
    ingestion_data_dir: str = Field(..., description="Directory containing raw documents")
    ingestion_file_pattern: str = Field(..., description="Glob pattern for matching files")
    ingestion_chunk_strategy: str = Field(
        ..., description="Chunking strategy: fixed, recursive, or semantic"
    )
    ingestion_chunk_size: int = Field(..., description="Target chunk size in characters")
    ingestion_chunk_overlap: int = Field(..., description="Overlap size in characters")
    ingestion_similarity_threshold: float = Field(
        ..., description="Similarity threshold for semantic chunking"
    )
    ingestion_save_processed: bool = Field(
        ..., description="Whether to save processed chunks to disk"
    )
    ingestion_processed_dir: str = Field(
        ..., description="Directory to save processed chunks"
    )

    # Generation Configuration
    generation_temperature: float = Field(
        ..., description="Temperature for LLM generation"
    )


# Global settings instance
settings = Settings()
