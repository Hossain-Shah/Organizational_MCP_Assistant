from org_rag.graph_retriever import retrieve
from org_rag.llm_reasoner import reason

async def org_rag_agent(query: str) -> str:
    """
    Standalone agentic RAG for knowledge base
    """
    docs = retrieve.retrieve(query)

    if not docs:
        return "I couldn’t find relevant information."

    return await reason(query, docs)
