import os
from typing import List
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables (API Keys)
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize the LLM (Large Language Model)
# Using a specific Groq model for fast inference
llm = ChatGroq(model_name="moonshotai/kimi-k2-instruct-0905")

#---------------------------------------
# 1. Load and Embed Documents (Vector DB)
#---------------------------------------
# Load research notes from a local text file
docs = TextLoader("../research_notes.txt", encoding="utf-8").load()

# Split documents into smaller chunks for better retrieval accuracy
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)

# Use HuggingFace embeddings to convert text chunks into numerical vectors
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store the vector embeddings in a FAISS vector database for fast similarity search
vectorestore = FAISS.from_documents(chunks, embeddings)
retriever = vectorestore.as_retriever()

#---------------------------------------
# 2. Define Agent State
#---------------------------------------
# The state object manages data that flows through the LangGraph nodes
class IterativeRAGState(BaseModel):
    question: str              # Original user question
    redefined_question: str = "" # Improved query if initial retrieval was insufficient
    retrieved_docs: List[Document] = [] # List of documents found by retriever
    answer: str = ""           # Generated answer from the LLM
    verified: bool = False     # Flag indicating if the answer passed reflection
    attempts: int = 0          # Counter for iterative loops

#---------------------------------------
# 3. Define Graph Nodes (Functions)
#---------------------------------------

# Node: Retrieve Documents
def retrieve_docs(state: IterativeRAGState) -> IterativeRAGState:
    """Retrieves relevant documents based on the current (or redefined) question."""
    query = state.redefined_question or state.question
    docs = retriever.invoke(query)
    # Update state with retrieved documents
    return state.model_copy(update={"retrieved_docs": docs})

# Node: Generate Answer
def generate_answer(state: IterativeRAGState) -> IterativeRAGState:
    """Generates an answer using the retrieved context."""
    context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])
    prompt = f"""Use the following context to answer the question:
    Context:
    {context}
    
    Question:
    {state.question}
    """
    response = llm.invoke(prompt).content.strip()
    # Update answer and increment attempt counter
    return state.model_copy(update={"answer": response, "attempts": state.attempts + 1})

# Node: Reflect and Verify
def reflect_on_answer(state: IterativeRAGState) -> IterativeRAGState:
    """Evaluates if the generated answer is complete and accurate."""
    prompt = f"""
        Evaluate whether the answer below is actually sufficient and complete.

        Question: {state.question}
        Answer: {state.answer}

        Respond 'YES' if it's complete, otherwise 'NO' with feedback.
    """
    feedback = llm.invoke(prompt).content.lower()
    verified = 'yes' in feedback
    return state.model_copy(update={"verified": verified})

# Node: Refine Query
def refine_query(state: IterativeRAGState) -> IterativeRAGState:
    """If the answer is insufficient, refines the search query for better retrieval."""
    prompt = f"""
        The answer appears incomplete. Suggest a better version of the query that would help retrieve more relevant context.

        Original Question: {state.question}
        Current Answer: {state.answer}
    """
    new_query = llm.invoke(prompt).content.strip()
    return state.model_copy(update={"redefined_question": new_query})

#---------------------------------------
# 4. Build LangGraph (Workflow)
#---------------------------------------

# Initialize the graph with our state definition
builder = StateGraph(IterativeRAGState)

# Add our processing nodes to the graph
builder.add_node("retrieve", retrieve_docs)
builder.add_node("answer", generate_answer)
builder.add_node("reflect", reflect_on_answer)
builder.add_node("refine", refine_query)

# Define the workflow edges
builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "answer")
builder.add_edge("answer", "reflect")

# Use conditional edges for the 'reflect' node:
# If verified OR too many attempts -> END
# Else -> refine query then retrieve again
builder.add_conditional_edges(
    "reflect",
    lambda s: END if s.verified or s.attempts >= 2 else "refine"
)

# Route the refined query back to the retrieval node for another iteration
builder.add_edge('refine', 'retrieve')

# Compile the graph into an executable app
graph = builder.compile()

#---------------------------------------
# 5. Execute the Workflow
#---------------------------------------
if __name__ == '__main__':
    user_query = "agent loops with transformer-based systems?"
    initial_state = IterativeRAGState(question=user_query)
    
    # Run the graph and get the final result
    final_result = graph.invoke(initial_state)
    
    print("\n" + "="*50)
    print("Final Answer:\n", final_result['answer'])
    print("-" * 50)
    print(f"Verified: {final_result['verified']}")
    print(f"Attempts: {final_result['attempts']}")
    print("="*50)