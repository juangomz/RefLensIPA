from src.rag.generation.rag import RAGPipeline

def get_galdos_knowledge_tool(rag_pipeline: RAGPipeline):
    """
    Factory function to create the tool for an agent.
    """
    
    def search_galdos_novels(query: str) -> str:
        """
        Queries a RAG pipeline containing novels by Pérez Galdós.
        Use this to answer factual or conceptual questions about the book 'Bailén'.
        """
        # We only return the text answer so the agent doesn't get overwhelmed 
        # with SearchResult objects or raw metadata.
        response = rag_pipeline.query(query)
        return response.answer

    return search_galdos_novels