from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import chromadb
from pydantic import BaseModel, Field

from src.reflens.tool import Tool
from src.reflens.graph.neo4j_client import Neo4jClient


# -------- Chroma tool --------

class VectorSearchParams(BaseModel):
    query: str = Field(..., description="User query text")
    k: int = Field(5, ge=1, le=20, description="Top-K results")
    where: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional Chroma metadata filter (where clause)",
    )

def make_chroma_search_tool(persist_dir: str, collection_name: str = "kb_chunks") -> Tool:
    persist_path = str(Path(persist_dir))

    client = chromadb.PersistentClient(path=persist_path)
    collection = client.get_or_create_collection(collection_name)

    def _search(args: dict[str, Any]) -> dict[str, Any]:
        p = VectorSearchParams(**args)
        res = collection.query(
            query_texts=[p.query],
            n_results=p.k,
            where=p.where,
            include=["documents", "metadatas", "distances"],
        )

        out = []
        docs = res["documents"][0] if res.get("documents") else []
        metas = res["metadatas"][0] if res.get("metadatas") else []
        dists = res["distances"][0] if res.get("distances") else []

        for i in range(len(docs)):
            meta = metas[i] or {}
            out.append({
                "chunk_id": meta.get("chunk_id"),  # IMPORTANT: guardarlo en metadata al upsert
                "doc_id": meta.get("doc_id"),
                "source": meta.get("source"),
                "timestamp": meta.get("timestamp"),
                "text": docs[i],
                "distance": dists[i],
                "meta": meta,
            })

        return {"results": out, "k": p.k}

    return Tool(
        name="vector_search",
        description="Semantic search over chunks using Chroma.",
        parameters=VectorSearchParams,
        func=_search,
    )


# -------- Neo4j tool --------

class GraphQueryParams(BaseModel):
    cypher: str = Field(..., description="Cypher query to run in Neo4j")
    params: Optional[dict[str, Any]] = Field(default=None, description="Cypher parameters dict")

def make_neo4j_query_tool(neo4j: Neo4jClient) -> Tool:
    def _query(args: dict[str, Any]) -> dict[str, Any]:
        p = GraphQueryParams(**args)
        rows = neo4j.run(p.cypher, p.params)
        return {"rows": rows}

    return Tool(
        name="graph_query",
        description="Run a Cypher query on the knowledge graph (Neo4j) and return rows.",
        parameters=GraphQueryParams,
        func=_query,
    )