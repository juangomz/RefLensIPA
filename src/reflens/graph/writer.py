from __future__ import annotations

from datetime import datetime, timezone

from reflens.graph.schemas import NormalizedGraph
from reflens.graph.neo4j_client import Neo4jClient


def _now_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


class GraphWriter:
    def __init__(self, neo4j: Neo4jClient):
        self.neo4j = neo4j

    def write(self, ng: NormalizedGraph) -> dict:
        """
        Writes a NormalizedGraph to Neo4j idempotently.
        Returns metrics for observability.
        """
        t0 = _now_ms()

        chunk = ng.chunk.model_dump()
        entities = [e.model_dump() for e in ng.entities]
        relations = [r.model_dump() for r in ng.relations]

        with self.neo4j.driver.session(database=self.neo4j.config.database) as session:
            # Upsert chunk
            session.run(
                """
                MERGE (c:Chunk {doc_id:$doc_id, chunk_id:$chunk_id})
                SET c.text = $text,
                    c.source = $source,
                    c.timestamp = $timestamp,
                    c.topic = $topic,
                    c.embedding = $embedding,
                    c.updated_at = $updated_at
                """,
                doc_id=chunk["doc_id"],
                chunk_id=chunk["chunk_id"],
                text=chunk["text"],
                source=chunk["source"],
                timestamp=chunk["timestamp"],
                topic=chunk.get("topic"),
                embedding=chunk.get("embedding"),  # <---
                updated_at=_now_ms(),
            )

            # Upsert entities (UNWIND batch)
            session.run(
                """
                UNWIND $entities AS e
                MERGE (n:Entity {id: e.id})
                SET n.name = e.name,
                    n.canonical_name = e.canonical_name,
                    n.type = e.type,
                    n.aliases = e.aliases,
                    n.confidence = e.confidence,
                    n.updated_at = $updated_at
                """,
                entities=entities,
                updated_at=_now_ms(),
            )

            # Upsert relations (UNWIND batch)
            session.run(
                """
                UNWIND $relations AS r
                MATCH (a:Entity {id: r.source_id})
                MATCH (b:Entity {id: r.target_id})
                MERGE (a)-[rel:REL {id: r.id}]->(b)
                SET rel.type = r.type,
                    rel.confidence = r.confidence,
                    rel.evidence = r.evidence,
                    rel.updated_at = $updated_at
                """,
                relations=relations,
                updated_at=_now_ms(),
            )

            # Evidence: link involved entities to the chunk
            entity_ids = [e["id"] for e in entities]
            session.run(
                """
                UNWIND $entity_ids AS eid
                MATCH (e:Entity {id: eid})
                MATCH (c:Chunk {doc_id:$doc_id, chunk_id:$chunk_id})
                MERGE (e)-[m:MENTIONED_IN]->(c)
                SET m.updated_at = $updated_at
                """,
                entity_ids=entity_ids,
                doc_id=chunk["doc_id"],
                chunk_id=chunk["chunk_id"],
                updated_at=_now_ms(),
            )

        t1 = _now_ms()
        return {
            "entities": len(entities),
            "relations": len(relations),
            "latency_ms": t1 - t0,
            "doc_id": chunk["doc_id"],
            "chunk_id": chunk["chunk_id"],
        }