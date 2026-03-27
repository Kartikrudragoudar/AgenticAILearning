from numpy.random.mtrand import random
from typing_extensions import TypedDict
from typing import Literal
import random 
from langgraph.graph import StateGraph, START, END

class TypedDictState(TypedDict):
    name:str
    game:Literal["cricket", "football"]


def decide_play(state:TypedDictState)->['cricket', 'football']:
    if random.random() < 0.5:
        return "cricket"
    else:
        return "football"

def play_game(state:TypedDictState):
    print("---Play Game node has been called---")
    return {"name":state["name"] + " want to play "}

def cricket(state:TypedDictState):
    print("--Cricket node has been called--")
    return {"name":state["name"] + " cricket","game":"cricket"}

def football(state:TypedDictState):
    print("--Football node has been called--")
    return {"name":state["name"] + " football","game":"football"}

builder=StateGraph(TypedDictState)
builder.add_node("play_game", play_game)
builder.add_node("cricket", cricket)
builder.add_node("football", football)


#Flow of the graph
builder.add_edge(START, "play_game")
builder.add_conditional_edges("play_game", decide_play)
builder.add_edge("cricket", END)
builder.add_edge("football", END)

graph = builder.compile()

res=graph.invoke({"name":"Kartik"})
print(res)
