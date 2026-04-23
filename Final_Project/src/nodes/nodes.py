"""LangGraph nodes for RAG workflow"""

from src.state.rag_state import RAGState

class RAGNodes:
    """
    Initialize RAG Nodes
    
    Args:
        retriever: Document retriever instance
        llm: Language model instance
    """
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """
        Retrieve relevant documents node

        Args:
            state: Current RAG state
        
        Returns:
            Updated RAG state with retrieved documents
        """
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question = state.question,
            retrieved_docs=docs
        )


    # Generate answer from retrieved documents
    def generate_answer(self, state:RAGState) -> RAGState:
        """
        Generate answer using ReACT agent with retriever + wikipedia.

        Args:
            state: Current RAG state with retrieved documents
        
        Returns:
            Updated RAG state with retrieved documents
        """
        # Combine retrieved documents into context
        context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])

        # Create prompt
        prompt = f"""Answer the following question based on the context
        Context:
        {context}

        Question: {state.question}
        """

        # Generate response
        response = self.nodes.llm.invoke(prompt)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=response.content
        )