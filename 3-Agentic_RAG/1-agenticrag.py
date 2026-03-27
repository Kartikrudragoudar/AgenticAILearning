from typing import List
from typing_extensions import TypedDict
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
load_dotenv()

#-----------------------
# Document Processing
#-----------------------

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
]

loaders = [WebBaseLoader(url) for url in urls]
docs = []
for loader in loaders:
    docs.extend(loader.load())

## Recursive character text splitter an vectorstore
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",)
vector_store=FAISS.from_documents(split_docs, embeddings)
retriever = vector_store.as_retriever()

#-----------------------------
# 2. Define RAG State
#-----------------------------

class RAGState(TypedDict):
    question: str
    retrieved_docs: List[Document]
    answer: str


#------------------------------------------------
# 3.  LangGraph Nodes
#------------------------------------------------

llm = ChatGroq(model="qwen/qwen3-32b")

def retrieve_docs(state: RAGState) -> dict:
    docs = retriever.invoke(state["question"])
    return {"retrieved_docs": docs}

def generate_answer(state: RAGState) -> dict:
    context = "\n\n".join([doc.page_content for doc in state["retrieved_docs"]])
    prompt = f'Answer the question based on the context.\n\nContext:\n{context}\n\nQuestion: {state["question"]}'
    response = llm.invoke(prompt)
    return {"answer": response.content}

#-----------------------------------
# 4. Build the Graph
#-----------------------------------

builder = StateGraph(RAGState)

builder.add_node("retriever", retrieve_docs)
builder.add_node("responder", generate_answer)

builder.set_entry_point("retriever")
builder.add_edge("retriever", "responder")
builder.add_edge("responder", END)

graph = builder.compile()

if __name__ == "__main__":
    user_question = "What is the concept of agent loop in autonomous agents?"
    inital_state = RAGState(question=user_question)
    final_state = graph.invoke(inital_state)

    print("\n Final Answer:\n", final_state["answer"])