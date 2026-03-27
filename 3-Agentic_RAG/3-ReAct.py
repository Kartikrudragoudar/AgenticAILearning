import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader, ArxivLoader, TextLoader
from langchain_core.tools import Tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from typing import Annotated, Sequence, TypedDict
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()

llm = ChatGroq(model="qwen/qwen3-32b")
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

### Generic function to create a retrieval tool
def make_retriever_tool_from_text(file,name, desc):
    docs = TextLoader(file, encoding="utf-8").load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    vs = FAISS.from_documents(chunks, embedding=embedding)
    retriever = vs.as_retriever()


    def tool_func(query:str)-> str:
        print(f"Using tool: {name}")
        results = retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in results)
    return Tool(name=name, description=desc, func=tool_func)

# Wikipedia Tool
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
wiki_tool = Tool(
    name="Wikipedia",
    description="Use this tool to fetch general world knownledge from wikipedia",
    func=wikipedia.invoke
)

# ArXiv Tool
def arxiv_search(query:str)->str:
    print("Searching ArXiv....")
    results = ArxivLoader(query).load()
    return "\n\n".join(doc.page_content[:1000] for doc in results[:2] or "No Papers Found.")

arxiv_tool = Tool(
    name="ArxivSearch",
    description="Use this tool to fetch recent academic papers on technical topics",
    func=arxiv_search
)


internal_tool_1 = make_retriever_tool_from_text(
    "research_notes.txt",
    "InternalResearchNotes",
    "Search internal research notes for experimental results and agent designs"
)

tools = [wiki_tool, arxiv_tool, internal_tool_1]

react_node = create_agent(llm, tools=tools)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

builder = StateGraph(AgentState)
builder.add_node("agentic_rag", react_node)
builder.set_entry_point("agentic_rag")
builder.add_edge("agentic_rag", END)

graph = builder.compile()

if __name__ == '__main__':
    query = "What does our internal research notes say about transformer variants, and what does ArXiv suggest recently?"
    state = {"messages":[HumanMessage(content=query)]}
    result = graph.invoke(state)

    print("\n Final Answer:\n", result["messages"][-1].content)
