"""Dcoument processing module for loading and splitting documents"""

from typing import List
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Union
from pathlib import Path
from langchain_community.document_loaders import (WebBaseLoader, PyPDFLoader, TextLoader, PyPDFDirectoryLoader)

class DocumentProcessor:
    "Handle document loading and processing"

    def __init__(self, chunk_size: int=500, chunk_overlap: int=50):
        """
        Initialize DocumentProcessor.

        Args:
            chunk_size (int): Size of text chunks
            chunk_overlap (int): Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    
    def load_from_url(self, url:str) -> List[Document]:
        """
        Load document from URLs
        """
        loader = WebBaseLoader(url)
        return loader.load()
    
    def load_from_pdf_dir(self, directory:Union[str, Path]) -> List[Document]:
        """Load Documents from all PDF's inside a directory"""
        loader = PyPDFDirectoryLoader(str(directory))
        return loader.load()
    
    def load_from_txt(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a TXT file"""
        loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()

    def load_from_pdf(self, file_path:Union[str, Path]) -> List[Document]:
        """Load document(s) from a PDF file"""
        loader = PyPDFDirectoryLoader(str("data"))
        return loader.load()
    
    def load_documents(self, sources: List[str]) -> List[Document]:
        """
        Load documents from URLs, PDF folder paths, or TXT file 

        Args:
            sources: List of URLs, PDF folder paths, or TXT file paths

        Returns:
            List: List of loaded documents
        """ 
        docs: List[Document] = []
        for src in sources:
            if src.startswith("http://") or src.startswith("https://"):
                docs.extend(self.load_from_url(src))
            else:
                path = Path(src)
                if path.is_dir():
                    docs.extend(self.load_from_pdf_dir(path))
                elif path.is_file():
                    if path.suffix.lower() == ".txt":
                        docs.extend(self.load_from_txt(path))
                    elif path.suffix.lower() == ".pdf":
                        docs.extend(self.load_from_pdf(path))
                else:
                    print(f"Warning: Source {src} not found or unsupported.")
        
        # Optional: Load from default 'data' directory if explicitly desired, 
        # but let's keep it clean and only load what's in sources.
        return docs
    
    def split_documents(self, documents:List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents List of documents to split

        Returns:
            List: List of chunks
        """
        return self.splitter.split_documents(documents)


    def process_url(self, urls:List[str])-> List[Document]:
        """
        Complete pipeline to load and split documents

        Args:
            urls List of URLs to process
        Returns:
            List: List of processed document chunks
        """
        docs = self.load_documents(urls)
        return self.split_documents(docs)
        
