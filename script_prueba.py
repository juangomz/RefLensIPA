# from src.reflens.graph.schemas import Chunk
# from src.reflens.graph.extractor import extract_graph
# from src.reflens.graph.normalizer import normalize_graph
# from src.reflens.graph.neo4j_client import Neo4jClient
# from src.reflens.graph.writer import GraphWriter

# chunk = Chunk(
#     doc_id="doc1",
#     chunk_id="c1",
#     text="RefLens requires a RAG pipeline and uses Neo4j as graph database.",
#     source="test.pdf",
#     timestamp="2026-02-18",
#     embedding=[0.1, -0.2, 0.3, 0.4]  # ejemplo
# )

# extracted = extract_graph(chunk)
# normalized = normalize_graph(extracted, chunk)

# neo4j = Neo4jClient.from_env()
# neo4j.create_constraints()

# writer = GraphWriter(neo4j)
# metrics = writer.write(normalized)
# res = neo4j.run("MATCH (n) RETURN count(n) AS total")
# print(res)  # [{'total': ...}]
# print(metrics)
# neo4j.close()

# from src.reflens.graph.neo4j_client import Neo4jClient
# from src.reflens.query.tools import make_neo4j_query_tool

# neo4j = Neo4jClient.from_env()


# graph_query = make_neo4j_query_tool(neo4j)

# print(type(graph_query))
# print(graph_query)
# print("attrs:", [a for a in dir(graph_query) if not a.startswith("_")])
# result = graph_query.execute({
#     "cypher": "MATCH (n) RETURN count(n) AS total",
#     "params": {}
# })
# print(result)

from src.reflens.query.query_agent import QueryAgent
from rag.config import settings
print("PERSIST DIR:", settings.chroma_persist_directory)
print("COLLECTION:", getattr(settings, "chroma_collection", None))

import chromadb
from rag.config import settings

client = chromadb.PersistentClient(path=settings.chroma_persist_directory)

print("Collections:", [c.name for c in client.list_collections()])

# prueba con tu colección esperada
name = getattr(settings, "chroma_collection", None) or "kb_chunks"
col = client.get_or_create_collection(name=name)

count = col.count()
print("Count in", name, "=", count)

query = QueryAgent(settings=settings.chroma_persist_directory)

print("\n--- VECTOR SEARCH ---")
vector_tool = query.tools["vector_search"]
v = vector_tool.execute({
    "query": "Napoleón",
    "k": 5,
    "where": None
})
print(v)

print("\n--- GRAPH QUERY ---")
graph_tool = query.tools["graph_query"]
g = graph_tool.execute({
    "cypher": """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($q)
        RETURN e.name AS name
        LIMIT 5
    """,
    "params": {"q": "Napoleón"}
})
print(g)