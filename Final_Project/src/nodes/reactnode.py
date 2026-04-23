"""LangGraph nodes for RAG workflow + ReAct Agent inside generate_content"""

from groq.types.chat.completion_create_params import Document
from langchain_community.tools import Tool
from typing import List, Optional
from src.state.rag_state import RAGState
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikidata.tool import WikidataQueryRun

class RAGNodes:
    """Contains the node functions for RAG workflow"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None
    
    def retrieve_docs(self, state:RAGState) -> RAGState:
        """Classic retriever node"""
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )
    
    def _build_tools(self)-> List[Tool]:
        """Build retriever + wikipedia tools"""

        def retriever_tool_fn(query:str)-> str:
            docs: List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No documents found."
            
            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata if hasattr(d, 'metadata') else {}
                title = meta.get('title') or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")
            return "\n\n".join(merged)
        
        retriever_tool = Tool(
            name="retriever",
            description="Fetch passages from indexed vectorstore",
            func=retriever_tool_fn
        )
        wiki=WikidataQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=3, lang='en')
        )
        wiki_tool = Tool(
            name="wikipedia",
            description="Search general world knowledge on wikipedia",
            func=wiki.run
        )

        return [retriever_tool,wiki_tool]

        # For now, let's use a simple chain as the 'agent' implementation is incomplete
        pass

    def generate_answer(self, state: RAGState) -> RAGState:
        """Node to generate the final answer"""
        context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])
        prompt = f"Question: {state.question}\n\nContext:\n{context}\n\nAnswer:"
        response = self.llm.invoke(prompt)
        
        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=response.content
        )
