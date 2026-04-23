"""Graph builder defintion for langGraph"""
from src.state.rag_state import RAGState
from langgraph.graph import StateGraph, END
from src.nodes.reactnode import RAGNodes

# Your graph building logic here
class GraphBuilder:
    """Builds and manages the LangGraph workflow."""
    def __init__(self, retriever, llm):
        """
        Initialize the graph builder

        Args:
            retriever: Document retriever instance
            llm: Language model instance
        """

        self.nodes = RAGNodes(retriever, llm)
        self.graph = None
    
    def build(self):
        """
        Build the RAG workflow graph

        Returns:
            Complied graph instance
        """
        # Create state graph
        builder = StateGraph(RAGState)

        # Add nodes
        builder.add_node("retriever", self.nodes.retrieve_docs)
        builder.add_node("responder", self.nodes.generate_answer)

        # Set entry point
        builder.set_entry_point("retriever")

        # Add edges
        builder.add_edge("retriever", "responder")
        builder.add_edge("responder", END)

        # Compile graph
        self.graph = builder.compile()
        return self.graph 
    
    def run(self, question:str) -> dict:
        """
        Run the RAG Workflow

        Args:
            question: User question
        
        Returns:
            Final state with answer
        """
        if self.graph is None:
            self.build()
        
        initial_state = RAGState(question=question)
        return self.graph.invoke(initial_state)