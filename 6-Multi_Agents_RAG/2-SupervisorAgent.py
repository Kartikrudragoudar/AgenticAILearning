"""
2-SupervisorAgent.py
====================
Multi-Agent Supervisor System using LangGraph + LangChain + Groq.

This program demonstrates a Supervisor Agent pattern where:
  - A "research_agent" handles knowledge retrieval (RAG) from internal notes.
  - A "math_agent" handles arithmetic using explicit tool calls (add, multiply, divide).
  - A "supervisor" orchestrates both agents, ensuring each task is delegated to the right specialist.

Key Design Decisions:
  - The Supervisor uses a strict checklist prompt to FORCE sequential handoffs.
  - The Research Agent is explicitly told to NEVER do math (says 'MATH_REQUIRED' instead).
  - The Math Agent is told to ALWAYS use its tools, never answer from internal knowledge.

Dependencies (install via pip):
  pip install langgraph langgraph-supervisor langchain langchain-groq langchain-tavily
  pip install langchain-community langchain-huggingface langchain-text-splitters faiss-cpu
  pip install python-dotenv

Environment Variables (set in .env file):
  GROQ_API_KEY     - Your Groq Cloud API key
  TAVILY_API_KEY   - Your Tavily search API key
"""

from typing import Literal

# langgraph.graph: Provides StateGraph for building the agent workflow graph
from langgraph.graph import MessagesState

# langchain_core.messages: Message types used across agents for communication
from langchain_core.messages import BaseMessage

# langgraph.types: Command is used to route control flow between agent nodes
from langgraph.types import Command

# langchain.agents.create_agent: Creates tool-calling agents compatible with langgraph-supervisor
# NOTE: If you see deprecation warnings, this is still the stable import for langgraph v1.x
from langchain.agents import create_agent

import os
from dotenv import load_dotenv

# langchain_groq: Provides the ChatGroq LLM wrapper for Groq Cloud's ultra-fast inference
from langchain_groq import ChatGroq

# langchain_tavily: Web search tool powered by Tavily API (used by the research agent)
from langchain_tavily import TavilySearch

# langchain_core.tools.Tool: Used to wrap custom Python functions as LangChain-compatible tools
from langchain_core.tools import Tool

# langgraph.graph: END and START constants define the entry/exit points of the state graph
from langgraph.graph import StateGraph, END, START

# TextLoader: Loads plain text files as LangChain Document objects
from langchain_community.document_loaders import TextLoader

# RecursiveCharacterTextSplitter: Splits documents into smaller chunks for embedding
from langchain_text_splitters import RecursiveCharacterTextSplitter

# FAISS: Facebook's vector store for fast similarity search over document embeddings
from langchain_community.vectorstores import FAISS

# HuggingFaceEmbeddings: Generates text embeddings using HuggingFace sentence-transformers
from langchain_huggingface import HuggingFaceEmbeddings

# langgraph_supervisor: High-level API to create a Supervisor agent that coordinates sub-agents
from langgraph_supervisor import create_supervisor

# ---------------------------------------------------------------------------
# PATH SETUP: Compute an absolute path to 'research_notes.txt' so the script
# works regardless of which directory you run it from (e.g., project root vs. subfolder).
# ---------------------------------------------------------------------------
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
notes_path = os.path.join(base_path, "research_notes.txt")

# ---------------------------------------------------------------------------
# ENVIRONMENT: Load API keys from .env file into the environment
# ---------------------------------------------------------------------------
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# ---------------------------------------------------------------------------
# LLM SETUP: Initialize the Groq-hosted LLM.
# 'qwen/qwen3-32b' is the current model. You can also use 'llama-3.3-70b-versatile'.
# NOTE: Free-tier Groq has a 100,000 TPD (tokens per day) limit.
# ---------------------------------------------------------------------------
llm = ChatGroq(model="qwen/qwen3-32b")

# Tavily web search tool (used by the research agent for live web lookups)
web_search = TavilySearch(max_results=5)


# ===========================================================================
# RAG RETRIEVER TOOL
# ===========================================================================
# This function creates a retrieval tool from a text file using:
#   1. TextLoader      → loads the file
#   2. TextSplitter    → chunks it into ~500-char pieces
#   3. FAISS           → builds a vector index from those chunks
#   4. Tool wrapper    → exposes the retriever as a callable LangChain Tool
# ===========================================================================
def make_retriever_tool_from_text(file, name, desc):
    docs = TextLoader(file, encoding="utf-8").load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=200
    ).split_documents(docs)
    vs = FAISS.from_documents(
        chunks,
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    retriever = vs.as_retriever()

    def tool_func(query: str) -> str:
        """Retrieve relevant chunks from the research notes."""
        print(f"Using tool: {name}")
        results = retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in results)

    return Tool(name=name, description=desc, func=tool_func)


# Create the retrieval tool instance from the research notes file
internal_tool_1 = make_retriever_tool_from_text(
    notes_path,
    name="research_notes_tool",
    desc="Useful for answering questions about the internal research notes on AI agents."
)


# ===========================================================================
# HELPER: Routing Logic
# ===========================================================================
# Determines whether the workflow should continue to the next agent or stop.
# If an agent's last message contains 'FINAL ANSWER', the graph terminates.
# ===========================================================================
def get_next_node(last_message: BaseMessage, goto: str):
    if "FINAL ANSWER" in last_message.content:
        return END
    return goto


# ===========================================================================
# HELPER: System Prompt Builder
# ===========================================================================
# Creates a collaborative system prompt for agents. The `suffix` parameter
# customizes the role-specific instructions for each agent.
# ===========================================================================
def make_system_prompt(suffix: str) -> str:
    return (
        "You are helpful AI Assistant, collaborating with other assistants."
        "Use the provided tools to progress towards answering the question."
        "If you are unable to fully answer, that's OK, another assistant with different tools"
        "Will help where you left off. Execute what you can make progress."
        "If you or any of ther other assistants have the final answer or delievrable"
        "Prefix your response with FINAL ANSWER so the team knows to stop."
        f"\n{suffix}"
    )


# ===========================================================================
# BLOG GENERATOR WORKFLOW (currently unused, kept for reference)
# ===========================================================================
# These agents and nodes form a Research → Blog generation pipeline.
# They are NOT used by the Supervisor workflow below, but are kept here
# as a template for a two-agent content-generation graph.
# ===========================================================================

### Research agent for blog workflow
research_agent = create_agent(
    model=llm,
    tools=[internal_tool_1],
    system_prompt=make_system_prompt(
        "You can only research. Use the tool that you are binded with. "
        "You are working with a content writer colleague."
    )
)

## Research Node: Invokes the research agent and routes to blog_generator or END
def research_node(state: MessagesState) -> Command[Literal["blog_generator", END]]:
    result = research_agent.invoke(state)
    last_message = result["messages"][-1]
    goto = get_next_node(last_message, "blog_generator")
    last_message.name = "researcher"
    return Command(
        update={"messages": [last_message]},
        goto=goto,
    )

## Blog write agent: Takes research output and writes a detailed blog post
blog_agent = create_agent(
    model=llm,
    tools=[],
    system_prompt=make_system_prompt(
        "You can only write a detailed blog. You are working with a researcher colleague."
    )
)

## Blog Node: Invokes the blog agent and routes back to researcher or END
def blog_node(state: MessagesState) -> Command[Literal["researcher", END]]:
    result = blog_agent.invoke(state)
    last_message = result["messages"][-1]
    goto = get_next_node(last_message, "researcher")
    last_message.name = "blog_generator"
    return Command(
        update={"messages": [last_message]},
        goto=goto
    )


# ===========================================================================
# SUPERVISOR WORKFLOW — The main multi-agent system
# ===========================================================================
# This is the active workflow. It uses a Supervisor to coordinate:
#   1. research_agent  → retrieves info from internal notes
#   2. math_agent      → performs calculations using add/multiply/divide tools
# ===========================================================================

# --- RESEARCH AGENT ---
# Handles all knowledge retrieval tasks. Equipped with:
#   - web_search: Live web search via Tavily API
#   - internal_tool_1: RAG retriever over research_notes.txt
# STRICT RULE: This agent must NEVER perform calculations.
research_agent = create_agent(
    model=llm,
    tools=[web_search, internal_tool_1],
    system_prompt=(
        "You are a Research Expert. You ONLY look up facts using your tools.\n"
        "STRICT RULE: Do NOT perform any math. If a calculation is requested, "
        "simply provide the research and say: 'MATH_REQUIRED'. End your turn."
    ),
    name="research_agent",
)

# --- MATH TOOLS ---
# These are simple arithmetic functions wrapped as LangChain tools.
# Each function prints a log line so you can verify tool usage in the terminal.

def add(a: float, b: float):
    """Add two numbers"""
    print(f"\n--- [Internal Tool] Calling add({a}, {b}) ---")
    return a + b

def multiply(a: float, b: float):
    """Multiply two numbers"""
    print(f"\n--- [Internal Tool] Calling multiply({a}, {b}) ---")
    return a * b

def divide(a: float, b: float):
    """Divide two numbers"""
    print(f"\n--- [Internal Tool] Calling divide({a}, {b}) ---")
    return a / b  # Uses floating-point division for accuracy

# --- MATH AGENT ---
# Handles all arithmetic tasks. Equipped with: add, multiply, divide.
# STRICT RULE: Must ALWAYS use tools, never compute from memory.
math_agent = create_agent(
    model=llm,
    tools=[add, multiply, divide],
    system_prompt=(
        "You are a calculation expert. You MUST use your tools for EVERY number.\n"
        "Do NOT answer from memory. Just provide the final calculated number."
    ),
    name="math_agent",
)

# --- SUPERVISOR AGENT ---
# Orchestrates the research_agent and math_agent using a strict two-step checklist.
# The supervisor is FORBIDDEN from providing a final answer until both agents
# have completed their tasks. This prevents the LLM from "being lazy" and
# skipping the math_agent by answering simple math questions internally.
supervisor = create_supervisor(
    model=llm,
    agents=[research_agent, math_agent],
    system_prompt=(
        "You are a strict supervisor managing two tools:\n"
        "STEP 1: Call research_agent for notes.\n"
        "STEP 2: Once research_agent says 'MATH_REQUIRED', you MUST call math_agent.\n"
        "FORBIDDEN: You are not allowed to finish the job until math_agent has used its tools."
    ),
    add_handoff_back_messages=True,  # Adds context messages when agents hand back control
    output_mode="full_history"       # Returns the full message history for inspection
).compile()


# ===========================================================================
# EXECUTION: Run the Supervisor with a combined research + math query
# ===========================================================================
result = supervisor.invoke({
    "messages": "Check my internal research notes for transformer evaluation highlights"
                "AND calculate 2+2 using your tools."
})

# --- OUTPUT SECTION ---

# 1. Print the final answer from the last message in the conversation
print("\n--- FINAL ANSWER ---\n")
print(result["messages"][-1].content)

# 2. Print all tool interactions to verify that agents used their tools
# This loop inspects every message in the conversation history:
#   - AI messages with tool_calls show which agent requested which tool
#   - Tool messages show the results returned by each tool
print("\n--- TOOLS CALLED ---\n")
for m in result["messages"]:
    if m.type == "ai" and hasattr(m, "tool_calls") and m.tool_calls:
        for tc in m.tool_calls:
            print(f"Agent '{getattr(m, 'name', 'Unknown')}' requested tool: {tc['name']}")
    elif m.type == "tool":
        print(f"Tool Result from tool: {getattr(m, 'name', 'N/A')} (ID: {m.tool_call_id if hasattr(m, 'tool_call_id') else 'N/A'})")


# ===========================================================================
# BLOG GENERATOR GRAPH (Commented Out — activate to use the blog workflow)
# ===========================================================================
# ## Graph Builder
# workflow = StateGraph(MessagesState)
# workflow.add_node("researcher", research_node)
# workflow.add_node("blog_generator", blog_node)
# workflow.add_edge(START, "researcher")
# graph = workflow.compile()
#
# ## Final execution
# print("Calling Graph...")
# final_state = graph.invoke({
#     "messages": [HumanMessage(content="Write a detailed blog about transformer evaluation")]
# })
# print("\n--- FINAL RESULT ---\n")
# print(final_state["messages"][-1].content)