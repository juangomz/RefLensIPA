from __future__ import annotations

import time
from pathlib import Path

import fire

from lab4_agents.graph.chunks_loader import load_chunks_json
from lab4_agents.graph.extractor import extract_graph
from lab4_agents.graph.normalizer import normalize_graph
from lab4_agents.graph.neo4j_client import Neo4jClient
from lab4_agents.graph.writer import GraphWriter


def ingest(chunks_path: str, limit: int | None = None, skip: int = 0):
    """
    Ingest chunks from chunks.json into Neo4j:
      chunk -> LLM extract -> normalize -> MERGE into Neo4j

    Args:
      chunks_path: path to chunks.json
      limit: max chunks to process (for quick tests)
      skip: skip first N chunks (for debugging)
    """
    chunks = load_chunks_json(Path(chunks_path))
    chunks = chunks[skip : (skip + limit) if limit else None]

    neo4j = Neo4jClient.from_env()
    neo4j.create_constraints()
    writer = GraphWriter(neo4j)

    totals = {
        "chunks": 0,
        "entities": 0,
        "relations": 0,
        "latency_ms": 0,
        "failures": 0,
    }

    t0 = time.time()

    for idx, chunk in enumerate(chunks, start=1):
        try:
            extracted = extract_graph(chunk)
            normalized = normalize_graph(extracted, chunk)
            metrics = writer.write(normalized)

            totals["chunks"] += 1
            totals["entities"] += metrics["entities"]
            totals["relations"] += metrics["relations"]
            totals["latency_ms"] += metrics["latency_ms"]

            if idx % 10 == 0:
                print(f"[{idx}/{len(chunks)}] ok - totals so far: {totals}")

        except Exception as e:
            totals["failures"] += 1
            print(f"[{idx}/{len(chunks)}] ERROR doc={chunk.doc_id} chunk={chunk.chunk_id}: {e}")

    t1 = time.time()
    neo4j.close()

    totals["wall_time_s"] = round(t1 - t0, 2)
    print("DONE:", totals)
    return totals


if __name__ == "__main__":
    fire.Fire({"ingest": ingest})