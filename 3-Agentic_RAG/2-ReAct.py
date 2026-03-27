import os
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
load_dotenv()

#---------------------------------
# 1. Create Retriever Tool
#---------------------------------

#Load content from blog
docs = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/").load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding)
retriever = vectorstore.as_retriever()

def retriever_tool_func(query:str) ->  str:
    print("Using RAGRetriever tool")
    docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs])

retriever_tool = Tool(
    name="RAGRetriever",
    description="Use this tool to fetch relevant knowledge base info",
    func=retriever_tool_func
)

#Wikipedia tool
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

llm = ChatGroq(model="qwen/qwen3-32b")

#----------------------------
# 2. Define the Agent Node
#----------------------------
tools = [retriever_tool, wiki_tool]

#create the native Langgraph react agent
react_node = create_agent(llm, tools=tools)


#-----------------------------------
# 3. LangGraph Agent State
#-----------------------------------

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


#--------------------------------
# 4. Build LangGraph Graph
#--------------------------------

builder = StateGraph(AgentState)

builder.add_node('react_agent', react_node)
builder.set_entry_point("react_agent")
builder.add_edge("react_agent", END)

graph = builder.compile()

#------------------------------
#5. Run the ReAct Agent
#------------------------------

if __name__ == "__main__":
    user_query = "What is an agent loop and how does wikipedia describe autonomous agents?"
    state = {"messages": [HumanMessage(content=user_query)]}
    result = graph.invoke(state)

    print("\n Final Answer:\n", result["messages"][-1].content)