"""Vector Store for document embedding and retrieval"""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from typing import List


class VectorStore:
    """Message vector store application"""
    def __init__(self):
        self.embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = None
        self.retriever = None
    
    def create_retriever(self, documents:List[Document]):
        """
            Create vector store from documents

            Args:
                documents: List of documents to embed    
        """
        self.vectorstore = FAISS.from_documents(documents,self.embedding)
        self.retriever = self.vectorstore.as_retriever()
    
    def get_retriever(self):
        """

        Get the retriever instance

        Returns:
            Retriever instance
        """
        if self.retriever is None:
            raise ValueError("Vector store is not initialized. Call create_vectorstore first.")
        return self.retriever
    
    
    def retrieve(self, query: str, k:int = 4) -> List[Document]:
        """
            Retrieve  relevant documents for a query

            Args:
                query: Search query
                k: Number of documents to retrieve
            
            Returns:
                List of relevant documents
        """
        if self.retriever is None:
            raise ValueError("Vector store is not initialized. Call create_vectorstore first.")
        return self.retriever.invoke(query)