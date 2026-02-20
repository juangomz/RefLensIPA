from __future__ import annotations

from reflens.agent import Agent
from reflens.graph.neo4j_client import Neo4jClient
from reflens.query.tools import make_chroma_search_tool, make_neo4j_query_tool
from rag.config import settings
from rag.models.embedding_model import EmbeddingModel

QUERY_AGENT_SYSTEM = """You are QueryAgent for a hybrid RAG + Knowledge Graph system.

Tools:
- vector_search({query, k, where}) -> {results:[{chunk_id, doc_id, source, timestamp, text, distance, meta}], k}
- graph_query({cypher, params}) -> {rows:[...]}

Graph schema constraints (IMPORTANT):
- Do NOT assume any properties exist except:
  - Entity: name
  - Chunk: chunk_id (if present), otherwise use id
- Relationship types: use ONLY :REL and :MENTIONED_IN.
- If you need to know available properties, FIRST run:
  MATCH (e:Entity) RETURN keys(e) AS props LIMIT 1
  MATCH (c:Chunk)  RETURN keys(c) AS props LIMIT 1
- Never use e.description or c.timestamp/c.source unless you've verified they exist via keys().

Your job:
1) You MUST call vector_search at least once for every question, before any graph_query. If vector_search returns 1+ results, include at least 2 chunk_ids in Sources.
2) Use graph_query to fetch structured facts and evidence links relevant to the question:
   - Entities and their outgoing/incoming :REL edges
   - Evidence via (Entity)-[:MENTIONED_IN]->(Chunk)
3) Answer concisely and include a short Sources list with chunk_id/source when available.
"""

class QueryAgent(Agent):
    def __init__(
        self,
        chroma_persist_dir: str,
        chroma_collection: str = "kb_chunks",
        model: str | None = None,
    ):
        neo4j = Neo4jClient.from_env()

        # ✅ crea el embedding model igual que en ingesta
        embedding_model = EmbeddingModel(
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
            deployment_name=settings.azure_openai_embedding_deployment_name,
        )


        tools = [
            make_chroma_search_tool(chroma_persist_dir, embedding_model, chroma_collection),
            make_neo4j_query_tool(neo4j),
        ]
        super().__init__(model=model, tools=tools, system_prompt=QUERY_AGENT_SYSTEM)
        self._neo4j = neo4j

    def close(self):
        self._neo4j.close()