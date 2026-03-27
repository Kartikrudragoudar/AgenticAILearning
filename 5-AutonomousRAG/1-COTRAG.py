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

load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

#----------------------------
# 1. Prepare VectoreStore
#----------------------------

docs = TextLoader("../research_notes.txt", encoding="utf-8").load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorestore = FAISS.from_documents(chunks, embeddings)
retriever = vectorestore.as_retriever()
llm = ChatGroq(model_name="openai/gpt-oss-120b")

#-------------------------------------
# 2. LangGraph State Definition
#-------------------------------------

class RAGCoTState(BaseModel):
    question: str
    sub_steps: List[str] = []
    retrieved_docs: List[Document] = []
    answer: str = ""


#-------------------------
# 3. Nodes
#-------------------------

# a. Plan Sub-Questions
def plan_steps(state:RAGCoTState) -> RAGCoTState:
    prompt = f"Break the question into 2-3 reasoning steps: \n\n{state.question}"
    result = llm.invoke(prompt).content
    subqs = [line.strip("- ") for line in result.split("\n") if line.strip()]

    return state.model_copy(update={"sub_steps":subqs})

#b. Retrieve for each step
def retrieve_per_step(state:RAGCoTState)->RAGCoTState:
    all_docs = []
    for sub in state.sub_steps:
        docs = retriever.invoke(sub)
        all_docs.extend(docs)
    return state.model_copy(update={"retrieved_docs":all_docs})

#c. Generate Final Answer
def generate_answer(state:RAGCoTState) -> RAGCoTState:
    context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])
    prompt = f"""
    You are answering a complex question using reasoning and retrieved documents.

    Question: {state.question}

    Relevant Information
    {context}

    Now synthesize a well-reasoned final answer.
    """

    result = llm.invoke(prompt).content.strip()
    return state.model_copy(update={"answer":result})


#-------------------------------------
# 4. LangGraph Graph
#------------------------------------

builder = StateGraph(RAGCoTState)
builder.add_node("Planner", plan_steps)
builder.add_node("retriever", retrieve_per_step)
builder.add_node("responder", generate_answer)

builder.set_entry_point("Planner")
builder.add_edge("Planner", "retriever")
builder.add_edge("retriever", "responder")
builder.add_edge("responder", END)


graph = builder.compile()


if __name__ == "__main__":
    query = "What are the additional experiments in transfromer evalutaion?"
    state = RAGCoTState(question=query)
    final = graph.invoke(state)
    
    print("\n Reasoning Steps:", final["sub_steps"])
    print("\n Final Answer:\n", final["answer"])