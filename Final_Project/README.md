# 🔍 Agentic RAG Document Search

A powerful Retrieval-Augmented Generation (RAG) system built with **LangGraph**, **Groq**, and **Streamlit**. This system allows you to search through web content and local PDF documents using a structured agentic workflow.

## 🚀 Features

- **Multi-Source Ingestion**: Automatically scrapes web articles and processes local PDF documents from the `data/` directory.
- **Agentic Workflow**: Uses **LangGraph** to manage the state and flow between document retrieval and answer generation.
- **High-Performance LLM**: Powered by Groq's `llama-3.3-70b-versatile` for lightning-fast inference.
- **Local Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` for efficient, local vectorization.
- **Interactive UI**: Clean and modern Streamlit interface with real-time status updates and search history.
- **Optimized Processing**: Smart document splitting and redundant load prevention.

## 🏗️ Architecture

The system follows a modular design:
1. **Document Ingestion**: Loads content from URLs and local PDFs.
2. **Vector Store**: Indexes document chunks using FAISS for fast similarity search.
3. **Graph Workflow**: A LangGraph state machine with two primary nodes:
   - `retriever`: Fetches relevant context from the vector store.
   - `responder`: Generates a final answer using the LLM based on retrieved context.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Final_Project
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

## 💻 Usage

1. **Add Documents**:
   - Place any PDF files you want to search in the `data/` directory.
   - (Optional) Update `src/config/config.py` to add default URLs for scraping.

2. **Run the App**:
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Search**:
   - Enter your question in the search bar.
   - View answers along with the specific source document chunks used to generate them.

## ⚙️ Configuration

You can customize the system in `src/config/config.py`:
- `CHUNK_SIZE`: Control the size of document fragments (default: 500).
- `CHUNK_OVERLAP`: Control the overlap between chunks (default: 50).
- `MODEL_NAME`: Change the Groq model being used.

## 📝 Project Structure

```text
Final_Project/
├── data/               # Local PDF storage
├── src/
│   ├── config/         # Configuration settings
│   ├── document_ingestion/ # Loading and splitting logic
│   ├── vectorstore/    # FAISS implementation
│   ├── graph_builder/  # LangGraph orchestration
│   ├── nodes/          # Graph node functions
│   └── state/          # State definitions
├── streamlit_app.py    # Main UI application
└── README.md           # You are here
```

---
Built with ❤️ using LangChain, LangGraph, and Streamlit.
