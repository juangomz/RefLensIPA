from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver

load_dotenv()


@dataclass
class Neo4jConfig:
    uri: str
    user: str
    password: str
    database: str = "neo4j"  # default DB name in Neo4j


class Neo4jClient:
    def __init__(self, config: Neo4jConfig):
        self.config = config
        self.driver: Driver = GraphDatabase.driver(
            config.uri,
            auth=(config.user, config.password),
        )

    @classmethod
    def from_env(cls) -> "Neo4jClient":
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        database = os.getenv("NEO4J_DATABASE", "neo4j")
        return cls(Neo4jConfig(uri=uri, user=user, password=password, database=database))

    def close(self) -> None:
        self.driver.close()

    def create_constraints(self) -> None:
        """
        Creates constraints for idempotent MERGE writes.
        Run once at startup.
        """
        queries = [
            # Unique entity id
            """
            CREATE CONSTRAINT entity_id IF NOT EXISTS
            FOR (e:Entity) REQUIRE e.id IS UNIQUE
            """,
            # Unique chunk composite key
            """
            CREATE CONSTRAINT chunk_key IF NOT EXISTS
            FOR (c:Chunk) REQUIRE (c.doc_id, c.chunk_id) IS UNIQUE
            """,
            # Unique relation id
            """
            CREATE CONSTRAINT rel_id IF NOT EXISTS
            FOR ()-[r:REL]-() REQUIRE r.id IS UNIQUE
            """,
        ]

        with self.driver.session(database=self.config.database) as session:
            for q in queries:
                session.run(q)
            
    from typing import Any, Optional

    def run(self, cypher: str, params: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
            """Run a Cypher query and return records as dicts."""
            with self.driver.session(database=self.config.database) as session:
                result = session.run(cypher, params or {})
                return [record.data() for record in result]