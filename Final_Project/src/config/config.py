import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class for RAG system"""

    # API Key
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Model Configuration
    MODEL_NAME = "llama-3.3-70b-versatile"

    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    #Default URLs
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-diffusion-video"
    ]

    @classmethod
    def get_llm(cls):
        """Initialize and return the LLM Model"""
        os.environ["GROQ_API_KEY"] = cls.GROQ_API_KEY
        return ChatGroq(model=cls.MODEL_NAME)