##Implementing a simple chatbot using LangGraph
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()

##Reducers 
from typing import Annotated
from langgraph.graph.message import add_messages 

class State(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatGroq(model="qwen/qwen3-32b")

##We will start with creating nodes
def superbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph = StateGraph(State)
graph.add_node("superbot", superbot)

##Edges
graph.add_edge(START, "superbot")
graph.add_edge("superbot", END)

graph_builder=graph.compile()

##Invocation
# response = graph_builder.invoke({"messages":"Hi, My Name is Kartik and I Like Cricket"})
# print(response)


#Streaming The Responses
for event in graph_builder.stream({"messages":"Hi, My Name is Kartik and I Like Cricket"}, stream_mode="values"):
    print(event)