from reflens.query.query_agent import QueryAgent
from rag.config import settings
print("PERSIST DIR:", settings.chroma_persist_directory)
print("COLLECTION:", getattr(settings, "chroma_collection", None))

import chromadb
from chromadb.config import Settings
from rag.config import settings

client = chromadb.PersistentClient(
    path=settings.chroma_persist_directory,
    settings=Settings(anonymized_telemetry=False),
)

print("Collections:", [c.name for c in client.list_collections()])

# prueba con tu colección esperada
name = getattr(settings, "chroma_collection", None) or "kb_chunks"
col = client.get_or_create_collection(name=name)

count = col.count()
print("Count in", name, "=", count)

qa = QueryAgent(chroma_persist_dir=settings.chroma_persist_directory)

print("\n--- VECTOR SEARCH ---")
vector_tool = qa.tools["vector_search"]
v = vector_tool.execute({
    "query": "Napoleón",
    "k": 5,
    "where": None
})
print(v)

print("\n--- GRAPH QUERY ---")
graph_tool = qa.tools["graph_query"]
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