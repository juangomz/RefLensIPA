from __future__ import annotations

from src.reflens.agent import Agent
from src.reflens.graph.neo4j_client import Neo4jClient
from src.reflens.query.tools import make_chroma_search_tool, make_neo4j_query_tool


QUERY_AGENT_SYSTEM = """You are QueryAgent for a hybrid RAG + Knowledge Graph system.

You have two tools:
- vector_search(query, k, where): returns top-k chunks (text + metadata + distance).
- graph_query(cypher, params): query Neo4j.

Your job:
1) Use vector_search to retrieve evidence chunks relevant to the user's question.
2) Use graph_query to retrieve structured facts: entities, relations, and evidence links (MENTIONED_IN) relevant to the question.
3) Answer using BOTH: cite supporting chunks via chunk_id/source/timestamp when possible.
4) If graph query returns nothing useful, rely on vector_search evidence.
5) Do not hallucinate Cypher schema: use labels Entity, Chunk and relations REL, MENTIONED_IN.
Return a concise answer plus a short 'Sources' list of chunk_ids.
"""

class QueryAgent(Agent):
    def __init__(
        self,
        chroma_persist_dir: str,
        chroma_collection: str = "kb_chunks",
        model: str | None = None,
    ):
        neo4j = Neo4jClient.from_env()
        tools = [
            make_chroma_search_tool(chroma_persist_dir, chroma_collection),
            make_neo4j_query_tool(neo4j),
        ]
        super().__init__(model=model, tools=tools, system_prompt=QUERY_AGENT_SYSTEM)
        self._neo4j = neo4j  # keep for close()

    def close(self):
        self._neo4j.close()