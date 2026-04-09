import bs4
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langgraph.graph import END, StateGraph, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver # persistent memory (checkpointer) component
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")

llm = ChatGroq(model="qwen/qwen3-32b")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

## Document Ingestion And Processing
# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=["http://lilianweng.github.io/posts/2023-06-23-agent/"],
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title","post-header")
        )
    ),
)
docs = loader.load()

## chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

## Vector Store
vectorstore = FAISS.from_documents(
    documents=all_splits,
    embedding=embeddings
)

@tool()
def retrieve(query:str):
    """Retrieve the information related to the query"""
    retrieved_docs = vectorstore.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized

#Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond"""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessageState appends messages to state instead of overwriting
    return {"messages":[response]}

# Step 2: Execute the retrieval
tools = ToolNode([retrieve])

#Step 3: Generate a response  using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]
    
    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks."
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise "
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # RUN
    response = llm.invoke(prompt)
    return {"messages":[response]}

#Build Graph 
graph_builder = StateGraph(MessagesState)
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END:END, "tools":"tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

# Initialize memory saver to maintain persistent state (checkpoints) across conversations
memory = MemorySaver()
# Compile the graph with the checkpointer to enable persistent memory
graph = graph_builder.compile(checkpointer=memory)
# Configuration with a unique thread_id to identify and retrieve a specific conversation's memory
config = {"configurable":{"thread_id":"abcd123"}}

input_message = "What is task decompostion?"
for step in graph.stream(
    {"messages":[{"role":"user", "content":input_message}]},
    stream_mode = "values",
    config = config
):
    step["messages"][-1].pretty_print()

### Conversation History
chat_history = graph.get_state(config=config)
for message in chat_history.values["messages"]:
    message.pretty_print()