from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from typing import Literal

class State(BaseModel):
    name:str

##node function
def example_node(state:State):
    return {"name":"Hello"}

builder=StateGraph(State)
builder.add_node("example_node",example_node)
builder.add_edge(START, "example_node")
builder.add_edge("example_node", END)

graph=builder.compile()
res=graph.invoke({"name":"kartik"})
print(res)