# Import necessary libraries for web scraping, environment variables, LangChain, and LangGraph
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

# Load environment variables from .env file
load_dotenv()

# Set API keys for Groq and Tavily (if needed) for model and search access
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")

# Initialize LLM with Groq (Qwen model) and embedding model from HuggingFace
llm = ChatGroq(model="qwen/qwen3-32b")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

## Document Ingestion And Processing
# Load and chunk contents of the blog
# Load documents from a specific blog URL using WebBaseLoader and filter content
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
# Split documents into smaller chunks for better retrieval performance
# chunk_size defines the max length of each chunk, chunk_overlap ensures context continuity
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

## Vector Store
# Create a FAISS vector store from the documents and embeddings for similarity search
vectorstore = FAISS.from_documents(
    documents=all_splits,
    embedding=embeddings
)

@tool()
def retrieve(query:str):
    """Retrieve the information related to the query"""
    # Perform similarity search in the vector store for the top 2 relevant documents
    retrieved_docs = vectorstore.similarity_search(query, k=2)
    # Format and serialize the retrieved documents for the LLM's context
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized

#Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond"""
    # Bind the retrieve tool to the LLM so it knows it can use it
    llm_with_tools = llm.bind_tools([retrieve])
    # Invoke the model with the current conversation history
    response = llm_with_tools.invoke(state["messages"])
    # Return the response to be appended to the state's message list
    return {"messages":[response]}

# Step 2: Execute the retrieval
tools = ToolNode([retrieve])

# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Isolate relevant tool messages from the state for the current generation
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]
    
    # Extract content from the tool messages to use as context
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    
    # Construct the personalized system message with context information
    system_message_content = (
        "You are an assistant for question-answering tasks."
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise "
        "\n\n"
        f"{docs_content}"
    )
    
    # Filter the conversation messages to exclude any historical tool calls/outputs
    conversation_messages = [
        message for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    
    # Combine the system message and current conversation history for the final prompt
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Execute the final model call to generate a response for the user
    response = llm.invoke(prompt)
    return {"messages":[response]}

#Build Graph 
graph_builder = StateGraph(MessagesState)
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

# Define the flow: Start with query_or_respond node
graph_builder.set_entry_point("query_or_respond")

# Conditional edges to decide whether to call tools or end the conversation
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END:END, "tools":"tools"},
)

# Route from tools to generate and then to end
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

# Initialize memory saver to maintain persistent state (checkpoints) across conversations
memory = MemorySaver()
# Compile the graph with the checkpointer to enable persistent memory
graph = graph_builder.compile(checkpointer=memory)
# Configuration with a unique thread_id to identify and retrieve a specific conversation's memory
config = {"configurable":{"thread_id":"abcd123"}}

# Execute the graph with a user question and stream the results using the defined config (memory)
input_message = "What is task decompostion?"
for step in graph.stream(
    {"messages":[{"role":"user", "content":input_message}]},
    stream_mode = "values",
    config = config
):
    # Print the last message in each step of the stream
    step["messages"][-1].pretty_print()

### Conversation History
# Retrieve and print the full conversation history from the persistent state
chat_history = graph.get_state(config=config)
for message in chat_history.values["messages"]:
    message.pretty_print()