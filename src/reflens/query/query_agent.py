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

Graph schema (authoritative):
- Labels:
  - (:Entity) properties: id, name, canonical_name, aliases, type, confidence, updated_at
  - (:Chunk)  properties may include: chunk_id, doc_id, source, timestamp, text (do not assume others)
- Relationship types: use ONLY :REL and :MENTIONED_IN.

Rules:
1) You MUST call vector_search at least once for every question BEFORE any graph_query.
2) For questions of the form "Who is X?", "What is X?", "Tell me about X":
   - You MUST run a graph entity lookup using this exact Cypher template (no schema introspection first):
     MATCH (e:Entity)
     WHERE (e.name IS NOT NULL AND toLower(e.name) CONTAINS toLower($q))
        OR (e.canonical_name IS NOT NULL AND toLower(e.canonical_name) CONTAINS toLower($q))
        OR (e.aliases IS NOT NULL AND any(a IN e.aliases WHERE toLower(a) CONTAINS toLower($q)))
     RETURN e.name AS name, e.canonical_name AS canonical_name, e.type AS type, e.aliases AS aliases, e.id AS id
     LIMIT 20
   - Use params {"q": "<X>"}.
3) NEVER run keys()/schema introspection unless:
   - a graph_query fails with an error, OR
   - the lookup query returns 0 rows AND you need to adapt.
4) After finding an entity, you MAY fetch relations and evidence:
   - Relations:
     MATCH (e:Entity {id: $id})-[r:REL]->(e2:Entity)
     RETURN type(r) AS rel, e2.name AS target
     LIMIT 25
   - Evidence:
     MATCH (e:Entity {id: $id})-[:MENTIONED_IN]->(c:Chunk)
     RETURN c.chunk_id AS chunk_id, c.doc_id AS doc_id, c.source AS source
     LIMIT 10

Output:
- Answer concisely.
- Include Sources: list chunk_id + source from vector_search and any Chunk results from the graph.
"""

class QueryAgent(Agent):
    def __init__(
        self,
        chroma_persist_dir: str,
        chroma_collection: str = "kb_chunks",
        model: str | None = None,
    ):
        neo4j = Neo4jClient.from_env()

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