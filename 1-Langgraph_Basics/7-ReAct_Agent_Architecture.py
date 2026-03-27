from langgraph.graph import StateGraph
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
import os
from langchain_core.messages import AIMessage, HumanMessage, AnyMessage
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import START, END
from langgraph.prebuilt import ToolNode, tools_condition  
from langgraph.checkpoint.memory import MemorySaver
load_dotenv()


os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"]="ReAct_Agent_Architecture"


tavily = TavilySearch()

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)
arxiv=ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

api_wrapper_wikipedia = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper_wikipedia)

###Custom Functions
def multiply(a:int, b:int):
    """Multiply a and b.

    Args:
        a (int): First number
        b (int): Second number

    Returns:
        int: Product of a and b
    """
    return a*b

def add(a:int, b:int):
    """Add a and b.

    Args:
        a (int): First number
        b (int): Second number

    Returns:
        int: Sum of a and b
    """
    return a+b

def divide(a:int, b:int):
    """Divide a by b.

    Args:
        a (int): First number
        b (int): Second number

    Returns:
        int: Quotient of a and b
    """
    return a/b

tools = [arxiv, wiki, multiply, add, divide, tavily]

llm=ChatGroq(model="qwen/qwen3-32b")
llm_with_tools=llm.bind_tools(tools)
llm_with_tools.invoke([HumanMessage(content=f"What is the 2+2")]).tool_calls

##State Schema
class State(TypedDict):
    messages: Annotated[list[AnyMessage],add_messages]

###Node definition
def tool_calling_llm(state:State):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}

###Build Graph
builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node('tools', ToolNode(tools))

builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    #If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "tool_calling_llm")
memory=MemorySaver()
graph_memory=builder.compile(checkpointer=memory)

#View
config={"configurable":{"thread_id":"1"}}
messages=graph_memory.invoke({"messages":HumanMessage(content="Add 5+5 and then multiply by 13")}, config=config)
for m in messages['messages']:
    m.pretty_print()

###Specify the thread
messages=[HumanMessage(content="First multiply previous result with 56 and then result of the calculation should be added to  plus 18")]
state_after=graph_memory.invoke({"messages":messages},config=config)
print("\n--- SECOND RUN MESSAGES ---")
for m in state_after['messages']:
    m.pretty_print()