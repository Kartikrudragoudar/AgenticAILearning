from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages import AnyMessage
from typing import Annotated
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
load_dotenv()


def add(a:int, b:int)-> int:
    """Add a and b
    Args:
        a(int): first int
        b (int): second int
    Returns:
        int
    """
    return a + b

tools=[add]

# for message in messages:
#     message.pretty_print()


### Binding tool with llm
llm = ChatGroq(model_name="qwen/qwen3-32b")
llm_with_tools = llm.bind_tools([add])
response = llm_with_tools.invoke([HumanMessage(content=f'What is 2 plus 2', name="Kartik")])


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    name:str

###Reducers with add_messages is to append instead of override
intial_messages = [AIMessage(content=f"Please tell me how can I help", name="LLMModel")]
intial_messages.append(HumanMessage(content=f"I want to learn coding", name="Kartik"))
ai_message=AIMessage(content=f"Which Programming Language you want to learn", name="LLMModel")
add_messages(intial_messages, ai_message)

###chatbot node functionality
def llm_tool(state:State):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}

builder=StateGraph(State)
builder.add_node('llm_tool', llm_tool)
builder.add_edge(START, 'llm_tool')
builder.add_edge('llm_tool', END)


builder=StateGraph(State)

## Add nodes
builder.add_node("llm_tool", llm_tool)
builder.add_node("tools", ToolNode(tools=tools))

##Add edges
builder.add_edge(START, "llm_tool")
builder.add_conditional_edges(
    "llm_tool",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is not a tool call -> tools_condition routes to END
    tools_condition
)

builder.add_edge("tools", "llm_tool")

graph = builder.compile()

#invocation
messages=graph.invoke({"messages":"What is machine learning?"})
for message in messages["messages"]:
    message.pretty_print()
