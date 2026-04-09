# Import necessary modules for time tracking, environment handling, LangChain, and LangGraph
from __future__ import annotations
import time
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents import Document
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# Load environment variables from .env file
load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")

# ================ CONFIG ==================== #
# Configuration for embeddings, model selection, and retrieval parameters
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIM = 384

LLM_MODEL = "qwen/qwen3-32b"
LLM_TEMPERATURE = 0

RETRIEVE_TOP_K = 4
CACHE_TOP_K = 3

# Distance threshold for semantic cache hits (lower is more strict)
CACHE_DISTANCE_THRESHOLD = 0.45

# Optional TTL for cache entries (seconds). 0 = disabled.
CACHE_TTL_SEC = 0

### Cache Variable
# Model_Cache = {}

### Function to implement simple LLM caching: 
### 1. Checks if the query exists in Model_Cache (Cache Hit).
### 2. If hit, returns the cached response and prints execution time.
### 3. If miss, invokes the LLM, measures execution time, stores response in cache, and returns it.
# def cache_model(query):
#     start_time=time.time()
#     if Model_Cache.get(query):
#         print("**Cache Hit**")
#         end_time=time.time()
#         elapsed_time=end_time-start_time
#         print(f"EXECUTION TIME: {elapsed_time:.2f} seconds")
#         return Model_Cache.get(query)
#     else:
#         print("***CACHE MISS - EXECUTING MODEL***")
#         start_time=time.time()
#         response=llm.invoke(query)
#         end_time=time.time()
#         elapsed_time=end_time - start_time
#         print(f"EXECUTION TIME: {elapsed_time:.2f} seconds")
#         Model_Cache[query]=response
#         return response

###ADVANCED CAG
# ================== STATE ============= #
# Define the structure of the application's state throughout the graph
class RAGState(TypedDict):
    question:str
    normalized_question:str
    context_docs: List[Document]
    answer: Optional[str]
    citations: List[str]
    cache_hit: bool

# ================= GLOBAL ============== #
# Initialize shared components: Embeddings and LLM
EMBED = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
LLM = ChatGroq(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

# ----- QA CACHE (EMPTY, SAFE INIT) ----- #
# Initialize the FAISS index for semantic caching of question-answer pairs
qa_index = faiss.IndexFlatL2(VECTOR_DIM)
QA_CACHE = FAISS(
    embedding_function=EMBED,
    index=qa_index,
    docstore=InMemoryDocstore({}),
    index_to_docstore_id={}
)

# ----------------- RAG STORE (demo only) ----------------- #
# Mock document store for retrieval purposes
RAG_STORE = FAISS.from_texts(
    texts=[
        "LangGraph lets you compose stateful LLM workflows as graphs.",
        "In langGraph, nodes can be cached; node caching memoizes outputs keyed by inputs for a TTL.",
        "Retrieval-Agumented Generation (RAG) retrieves external context and injects it into prompts.",
        "Semantic caching reuses prior answers when new questions are semantically similar."
    ],
    embedding=EMBED
)

# ============= NODES ============ #
def normalize_query(state: RAGState) -> RAGState:
    q = (state['question'] or "").strip()
    state["normalized_question"] = q.lower()
    return state

def semantic_cache_lookup(state: RAGState) -> RAGState:
    """
    Performs a semantic lookup in the FAISS-based QA cache.
    1. Normalizes context and checks for a valid query.
    2. Performs similarity search in the vector store.
    3. Validates hits against distance thresholds and TTL.
    4. Updates state with cached answer if a match is found.
    """
    # 1. Access normalized query and reset cache hit flag for this turn
    q = state["normalized_question"]
    state["cache_hit"] = False 

    # 2. Early exit if the question is empty
    if not q:
        return state
    
    # 3. Guard: Ensure FAISS index is initialized and contains entries to prevent crashes
    if getattr(QA_CACHE, "index", None) is None or QA_CACHE.index.ntotal == 0:
        return state
    
    # 4. Search: Look for the 'CACHE_TOP_K' most similar previously answered questions
    hits=QA_CACHE.similarity_search_with_score(q, k=CACHE_TOP_K)
    if not hits:
        return state
    
    # 5. Result: Get the best match and its L2 distance (lower score = higher similarity)
    best_doc, dist = hits[0]
    
    # 6. TTL Check: If CACHE_TTL_SEC is set, ensure the cache entry hasn't expired
    if CACHE_TTL_SEC > 0:
        ts = best_doc.metadata.get("ts")
        if ts is None or (time.time() - float(ts)) > CACHE_TTL_SEC:
            return state
    
    # 7. Threshold Check: Only use the result if similarity distance is below threshold
    if dist <= CACHE_DISTANCE_THRESHOLD:
        cache_answer = best_doc.metadata.get("answer")
        if cache_answer:
            # 8. Hit: Populate the state with cached data and set citations
            state["answer"] = cache_answer
            state["citations"] = ["(cache)"]
            state["cache_hit"] = True
    
    return state


def respond_from_cache(state:RAGState) -> RAGState:
    """Placeholder node for returning cached response"""
    return state

def retrieve(state:RAGState) -> RAGState:
    """Fetch relevant documents from the vector store based on the user's question"""
    q = state["normalized_question"]
    docs = RAG_STORE.similarity_search(q, k=RETRIEVE_TOP_K)
    state["context_docs"] = docs
    return state

def generate(state:RAGState) -> RAGState:
    """Construct prompt and invoke model to generate a final answer with context citations"""
    q = state["question"]
    docs = state.get("context_docs",[])
    # Create context string with document markers
    ctx = "\n\n".join([f"[doc-{i}] {d.page_content}" for i, d in enumerate(docs, start=1)])

    system = (
        "You are a precise RAG assistant. Use the context when helpful."
        "Cite with [doc-i] markers if you use a fact from the context."
    )

    user = f"Question: {q}\n\nContext:\n{ctx}\n\nWrite a concise answer with citations"

    # Invoke model with system instructions and user query
    resp = LLM.invoke([{"role":"system", "content":system},
            {"role":"user", 'content':user}])
    
    state["answer"] = resp.content
    # Update citations list for traceability
    state["citations"] = [f"[doc-{i}]" for i in range(1, len(docs) + 1)]
    return state

def cache_write(state:RAGState) -> RAGState:
    """Store the newly generated answer back into the semantic cache for future reuse"""
    q = state["normalized_question"]
    a = state.get("answer")
    if not q or not a:
        return state
    
    # Save text and metadata (answer, timestamp) to QA_CACHE
    QA_CACHE.add_texts(
        texts=[q],
        metadatas=[{
            "answer":a,
            "ts":time.time(),
        }]
    )
    return state


# ===================== GRAPH WRITING ==================== #
# Initialize the graph with the defined state
graph = StateGraph(RAGState)

# Add all nodes to the graph
graph.add_node("normalize_query", normalize_query)
graph.add_node("semantic_cache_lookup", semantic_cache_lookup)
graph.add_node("respond_from_cache", respond_from_cache)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_node("cache_write", cache_write)

# Establish execution flow and edges
graph.set_entry_point("normalize_query")
graph.add_edge("normalize_query", "semantic_cache_lookup")

def branch(state:RAGState) -> str:
    """Router logic: decide between cached response and fresh retrieval"""
    return "respond_from_cache" if state.get("cache_hit") else "retrieve"

# Add conditional routing based on cache hit result
graph.add_conditional_edges(
    "semantic_cache_lookup",
    branch,
    {
        "respond_from_cache":"respond_from_cache",
        "retrieve":"retrieve"
    }
)

# Finalize flow paths to END
graph.add_edge("respond_from_cache", END)
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "cache_write")
graph.add_edge("cache_write", END)

# Compile graph with persistent memory
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# === DEMO === #
if __name__ == "__main__":
    thread_cfg = {"configurable":{"thread_id":"demo-user-1"}}
    
    q1 = "What is LangGraph?"
    out1 = app.invoke({"question":q1, "context_docs":[], "citations":[]}, thread_cfg)

    print("Answer:", out1["answer"])
    print("Citations:", out1.get("citations"))
    print("Cache Hit?:", out1.get("cache_hit"))