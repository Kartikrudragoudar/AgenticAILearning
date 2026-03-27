import os
from typing import Literal
from langgraph.graph import MessagesState
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.types import Command
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END, START
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")

llm = ChatGroq(model="openai/gpt-oss-120b")

tavily_tool = TavilySearch(max_results=5)


#Generate Function to create a retrieval tool
def make_retriever_tool_from_text(file, name,desc):
    docs = TextLoader(file, encoding="utf-8").load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200).split_documents(docs)
    vs = FAISS.from_documents(chunks, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    retriever = vs.as_retriever()

    def tool_func(query:str) -> str:
        print(f"Using tool: {name}")
        results = retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in results)
    
    return Tool(name=name, description=desc, func=tool_func)

internal_tool_1 = make_retriever_tool_from_text(
    "../research_notes.txt", 
    name="research_notes_tool", 
    desc="Useful for answering questions about the internal research notes on AI agents."
)

def get_next_node(last_message:BaseMessage, goto:str):
    if "FINAL ANSWER" in last_message.content:
        #Any agent decided the work is done
        return END
    return goto


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

### Reasearch agent and node
research_agent = create_agent(
    model=llm,
    tools=[internal_tool_1],
    system_prompt=make_system_prompt("You can only research. Use the tool that you are binded with. You are working with a content writer colleague.")
)

## Research Node
def research_node(state: MessagesState) -> Command[Literal["blog_generator", END]]:
    result = research_agent.invoke(state)
    last_message = result["messages"][-1]
    goto = get_next_node(last_message, "blog_generator")

    # Mark the message as coming from the researcher
    last_message.name = "researcher"

    return Command(
        update={
            "messages": [last_message]
        },
        goto=goto,
    )


## Blog write agent
blog_agent = create_agent(
    model=llm,
    tools=[],
    system_prompt=make_system_prompt(
        "You can only write a detailed blog. You are working with a researcher colleague."
    )
)

def blog_node(state: MessagesState) -> Command[Literal["researcher", END]]:
    result = blog_agent.invoke(state)
    last_message = result["messages"][-1]
    goto = get_next_node(last_message, "researcher")

    # Mark the message as coming from the blog generator
    last_message.name = "blog_generator"

    return Command(
        update={
            "messages": [last_message],
        },
        goto=goto
    )


## Graph Builder
workflow = StateGraph(MessagesState)
workflow.add_node("researcher", research_node)
workflow.add_node("blog_generator", blog_node)
workflow.add_edge(START, "researcher")
graph = workflow.compile()

# Final execution
print("Calling Graph...")
final_state = graph.invoke({"messages": [HumanMessage(content="Write a detailed blog about transformer evaluation")]})
print("\n--- FINAL RESULT ---\n")
print(final_state["messages"][-1].content)