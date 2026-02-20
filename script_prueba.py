from src.reflens.graph.schemas import Chunk
from src.reflens.graph.extractor import extract_graph
from src.reflens.graph.normalizer import normalize_graph
from src.reflens.graph.neo4j_client import Neo4jClient
from src.reflens.graph.writer import GraphWriter

chunk = Chunk(
    doc_id="doc1",
    chunk_id="c1",
    text="RefLens requires a RAG pipeline and uses Neo4j as graph database.",
    source="test.pdf",
    timestamp="2026-02-18",
    embedding=[0.1, -0.2, 0.3, 0.4]  # ejemplo
)

extracted = extract_graph(chunk)
normalized = normalize_graph(extracted, chunk)

neo4j = Neo4jClient.from_env()
neo4j.create_constraints()

writer = GraphWriter(neo4j)
metrics = writer.write(normalized)
res = neo4j.run("MATCH (n) RETURN count(n) AS total")
print(res)  # [{'total': ...}]
print(metrics)
neo4j.close()