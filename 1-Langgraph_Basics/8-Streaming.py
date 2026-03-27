from distro import version
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
import asyncio
load_dotenv()


class State(TypedDict):
    messages:Annotated[list, add_messages]

llm=ChatGroq(model='qwen/qwen3-32b')

memory=MemorySaver()

def superbot(state:State):
    return {"messages":[llm.invoke(state['messages'])]}

graph=StateGraph(State)

graph.add_node('superbot',superbot)
graph.add_edge(START,'superbot')
graph.add_edge('superbot', END)

graph_builder=graph.compile(checkpointer=memory)

config={"configurable":{"thread_id":"1"}}
for chunk in graph_builder.stream({'messages':'Hi, my name is kartik and I like cricket'}, config,stream_mode="updates"):
    print(chunk)

print("------------------------------------------------------------------------------")

for chunk in graph_builder.stream({'messages':'Hi, my name is kartik and I like cricket'}, config,stream_mode="values"):
    print(chunk)

print("-------------------------------------------------------------------------------")

config = {"configurable":{"thread_id":"3"}}

async def stream_events():
    async for event in graph_builder.astream_events({"messages":["Hi My name is Kartik and I like to play cricket"]},config,version='v2'):
        print(event)

asyncio.run(stream_events())