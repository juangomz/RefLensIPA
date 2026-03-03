from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import re

import chromadb
from chromadb.config import Settings
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from reflens.tool import Tool
from reflens.graph.neo4j_client import Neo4jClient

from langfuse import observe, get_client
from rag.config import settings
from rag.retrieval.fusion import reciprocal_rank_fusion
from rag.retrieval.types import SearchResult

# -------- Chroma tool --------

class VectorSearchParams(BaseModel):
    query: str = Field(..., description="User query text")
    k: int = Field(5, ge=1, le=20, description="Top-K results")
    where: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional Chroma metadata filter (where clause)",
    )

from rag.models.embedding_model import EmbeddingModel

def make_chroma_search_tool(persist_dir: str, embedding_model: EmbeddingModel, collection_name: str = "kb_chunks") -> Tool:
    persist_path = str(Path(persist_dir))
    client = chromadb.PersistentClient(
        path=persist_path,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(collection_name)

    # Build sparse index once from all chunks for lexical matching.
    all_docs = collection.get(include=["documents", "metadatas"])
    sparse_doc_ids = all_docs.get("ids", []) or []
    sparse_doc_texts = all_docs.get("documents", []) or []
    sparse_doc_metas = all_docs.get("metadatas", []) or []

    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", (text or "").lower())

    tokenized_corpus = [_tokenize(doc) for doc in sparse_doc_texts]
    sparse_index = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    def _meta_matches_where(meta: dict[str, Any], where: Optional[dict[str, Any]]) -> bool:
        if not where:
            return True
        if not isinstance(where, dict):
            return True
        for key, expected in where.items():
            if meta.get(key) != expected:
                return False
        return True
    
    @observe(name="vector_search", as_type="tool", capture_input=True, capture_output=True)
    def _search(args: dict[str, Any]) -> dict[str, Any]:
        p = VectorSearchParams(**args)

        dense_k = max(p.k * 3, settings.retrieval_dense_top_k, p.k)
        sparse_k = max(p.k * 3, settings.retrieval_sparse_top_k, p.k)

        # 1) Dense retrieval (semantic)
        q_emb = embedding_model.embed_batch([p.query], batch_size=1)[0]
        if hasattr(q_emb, "tolist"):
            q_emb = q_emb.tolist()

        dense_res = collection.query(
            query_embeddings=[q_emb],
            n_results=dense_k,
            where=p.where,
            include=["documents", "metadatas", "distances"],
        )

        dense_ids = dense_res["ids"][0] if dense_res.get("ids") else []
        dense_docs = dense_res["documents"][0] if dense_res.get("documents") else []
        dense_metas = dense_res["metadatas"][0] if dense_res.get("metadatas") else []
        dense_dists = dense_res["distances"][0] if dense_res.get("distances") else []

        dense_results: list[SearchResult] = []
        dense_result_map: dict[str, dict[str, Any]] = {}
        for i in range(len(dense_docs)):
            meta = dense_metas[i] or {}
            doc_id = dense_ids[i]
            dense_results.append(
                SearchResult(
                    doc_id=doc_id,
                    text=dense_docs[i],
                    metadata=meta,
                    score=float(dense_dists[i]),
                )
            )
            dense_result_map[doc_id] = {
                # Canonical chunk id comes from Chroma record id.
                "chunk_id": doc_id,
                "chunk_id_meta": meta.get("chunk_id"),
                "doc_id": meta.get("doc_id"),
                "source": meta.get("source"),
                "timestamp": meta.get("timestamp"),
                "text": dense_docs[i],
                "distance": dense_dists[i],
                "meta": meta,
                "retriever": "dense",
            }

        # 2) Sparse retrieval (BM25)
        sparse_results: list[SearchResult] = []
        sparse_result_map: dict[str, dict[str, Any]] = {}
        if sparse_index is not None and sparse_doc_texts:
            query_tokens = _tokenize(p.query)
            bm25_scores = sparse_index.get_scores(query_tokens)
            top_indices = bm25_scores.argsort()[-sparse_k:][::-1]

            for idx in top_indices:
                meta = sparse_doc_metas[idx] or {}
                if not _meta_matches_where(meta, p.where):
                    continue
                doc_id = sparse_doc_ids[idx]
                score = float(bm25_scores[idx])
                sparse_results.append(
                    SearchResult(
                        doc_id=doc_id,
                        text=sparse_doc_texts[idx],
                        metadata=meta,
                        score=score,
                    )
                )
                sparse_result_map[doc_id] = {
                    # Canonical chunk id comes from Chroma record id.
                    "chunk_id": doc_id,
                    "chunk_id_meta": meta.get("chunk_id"),
                    "doc_id": meta.get("doc_id"),
                    "source": meta.get("source"),
                    "timestamp": meta.get("timestamp"),
                    "text": sparse_doc_texts[idx],
                    "bm25_score": score,
                    "meta": meta,
                    "retriever": "sparse",
                }

        # 3) Hybrid fusion (RRF)
        fused = reciprocal_rank_fusion(
            result_lists=[dense_results, sparse_results],
            weights=[settings.hybrid_search_dense_weight, settings.hybrid_search_sparse_weight],
        )[: p.k]

        out = []
        for r in fused:
            payload = dense_result_map.get(r.doc_id) or sparse_result_map.get(r.doc_id)
            if payload is None:
                payload = {
                    "chunk_id": r.doc_id,
                    "chunk_id_meta": r.metadata.get("chunk_id"),
                    "doc_id": r.metadata.get("doc_id"),
                    "source": r.metadata.get("source"),
                    "timestamp": r.metadata.get("timestamp"),
                    "text": r.text,
                    "meta": r.metadata,
                }
            payload["score"] = r.score
            payload["retrieval_mode"] = "hybrid"
            out.append(payload)

        print(
            "[CHROMA_HYBRID] query=",
            p.query,
            "k=",
            p.k,
            "dense_hits=",
            len(dense_results),
            "sparse_hits=",
            len(sparse_results),
            "returned=",
            len(out),
            flush=True,
        )

        return {
            "results": out,
            "k": p.k,
            "query": p.query,   # útil para debug en Langfuse
            "mode": "hybrid",
        }

    return Tool(
        name="vector_search",
        description="Hybrid retrieval over chunks (semantic + lexical BM25) fused with RRF.",
        parameters=VectorSearchParams,
        func=_search,
    )


# -------- Neo4j tool --------

class GraphQueryParams(BaseModel):
    cypher: str = Field(..., description="Cypher query to run in Neo4j")
    params: Optional[dict[str, Any]] = Field(default=None, description="Cypher parameters dict")

def make_neo4j_query_tool(neo4j: Neo4jClient) -> Tool:
    @observe(name="neo4j_query",as_type="tool" ,capture_input=True, capture_output=True)
    def _query(args: dict[str, Any]) -> dict[str, Any]:
        p = GraphQueryParams(**args)

        rows = neo4j.run(p.cypher, p.params)

        return {
            "rows": rows,
        }
        
    return Tool(
        name="graph_query",
        description="Run a Cypher query on the knowledge graph (Neo4j) and return rows.",
        parameters=GraphQueryParams,
        func=_query,
    )
