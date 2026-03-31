import os
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, BaseMessage, trim_messages
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import Tool, tool
from langchain.agents import create_agent
from dotenv import load_dotenv
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Literal, Dict, Callable, Optional
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_experimental.utilities import PythonREPL

# Set User Agent for LangChain tools
os.environ["USER_AGENT"] = "MyAgenticApp/1.0"

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
notes_path = os.path.join(base_path, "research_notes.txt")

load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")

llm = ChatGroq(model="qwen/qwen3-32b")

_TEMP_DICTIONARY = TemporaryDirectory()
WORKING_DIRECTORY = Path(_TEMP_DICTIONARY.name)


tavily_tool = TavilySearch(max_results=5)

# ===========================================================================
# RAG RETRIEVER TOOL FACTORY
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

# Tool: Scrape content from specific URLs.
@tool
def scrape_webpages(urls: List[str]) -> str:
    """Scrape the content of multiple webpages from provided URLs and return them as formatted text."""
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return "\n\n".join(
        [
            f'<Document name="{doc.metadata.get("title","")}" >{doc.page_content}</Document>' 
            for doc in docs
        ]
    )

@tool
def read_document(
    file_name:Annotated[str, "File path to read the document from."],
    start: Annotated[Optional[int],"The start line. Default is 0."] = None,
    end: Annotated[Optional[int], "The end line. Default is None"] = None,
) -> str:
    """Read the specified document."""
    with (WORKING_DIRECTORY / file_name).open('r') as file:
        lines = file.readlines()
    if start is None:
        start = 0
    return "\n".join(lines[start:end])


# Tool: Generate a numbered outline from a list of points.
@tool
def create_outline(
    points:Annotated[List[str], "List of main points or sections."],
    file_name:Annotated[str, "File path to save the outline."]
)-> Annotated[str, "Path of the saved outline file."]:
    """Convert a list of bullet points into a numbered outline and save it to a file."""
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        for i, point in enumerate(points):
            file.write(f"{i + 1}.{point}\n")
    return f"Outline saved to {file_name}"

# Tool: Create and save a document with given content.
@tool
def write_document(
    content: Annotated[str, "Text context to be written into the document."],
    file_name:Annotated[str, "File path to save the document."]
) -> Annotated[str, "Path of the saved document file."]:
    """Write the provided string content into a new document file at the specified path."""
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.write(content)
    return f"Document saved to {file_name}"

# Tool: Insert text at specific line numbers in an existing document.
@tool
def edit_document(
    file_name:Annotated[str, "Path of the document to be edited"],
    inserts: Annotated[
        Dict[int, str],
        "Dictionary where key is the line number (1-indexed and value is the text to be inserted at that line.",
    ], 
) -> Annotated[str, "Path of the edited document file."]:
    """Edit an existing document by inserting text at specific line numbers (1-indexed)."""

    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()
    
    sorted_inserts = sorted(inserts.items())

    for line_number, text in sorted_inserts:
        if 1 <= line_number <= len(lines) + 1:
            lines.insert(line_number - 1, text + "\n")
        else:
            return f"Error: Line number {line_number} is out of range."

    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.writelines(lines)
    
    return f"Document edited and saved to {file_name}"

repl = PythonREPL()


# Tool: Execute Python code in a REPL environment.
@tool
def python_repl_tool(
    code: Annotated[str, "The Python code to execute to generate your chart"],
):
    """Use this to execute python code. If you want to see the output of a value,
    You should print it out with `print(....)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error. {repr(e)}"
    return f"Successfully executed:\n ```python\n{code}\n```\nStdout: {result}"

# Define the State for the graph
class State(MessagesState):
    next: str

# Helper to keep message history manageable
trimmer = trim_messages(
    max_tokens=65, # Keep approx last 10-15 messages
    strategy="last",
    token_counter=len, # Simple count-based trimming
    include_system=True,
    allow_partial=False,
    start_on="human",
)

def make_supervisor_node(llm: BaseChatModel, members: list[str]) -> Callable:
    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the "
        f"following workers: {members}. Given the following user request, "
        "respond with the worker to act next. Each worker will perform a "
        "task and respond with their results and status. When finished, "
        "respond with FINISH."
    )
    
    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""
        next: Literal[*options] # Use str for dynamic options in TypedDict

    def supervisor_node(state: State) -> Command[Literal["__end__"]]:
        """An LLM-based router node"""
        # Trim messages before sending to supervisor to save tokens
        trimmed_messages = trimmer.invoke(state["messages"])
        messages = [
            {"role": "system", "content": system_prompt},
        ] + trimmed_messages
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        if goto == "FINISH":
            goto = END
        
        return Command(goto=goto, update={"next": goto})

    return supervisor_node

# --- Agent and Node Definitions ---

# Search Node
search_agent = create_agent(llm, tools=[tavily_tool, internal_tool_1])

def search_node(state: State) -> Command[Literal["supervisor"]]:
    result = search_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="search")
            ]
        },
        goto="supervisor"
    )

# Web Scraper Node
web_scraper_agent = create_agent(llm, tools=[scrape_webpages])

def web_scraper_node(state: State) -> Command[Literal["supervisor"]]:
    result = web_scraper_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="web_scraper")
            ]
        },
        goto="supervisor"
    )

research_supervisor_node = make_supervisor_node(llm, ["search", "web_scraper"])

research_builder = StateGraph(State)
research_builder.add_node("supervisor", research_supervisor_node)
research_builder.add_node("search", search_node)
research_builder.add_node("web_scraper", web_scraper_node)

research_builder.add_edge(START, 'supervisor')
research_graph = research_builder.compile()


doc_writer = create_agent(
    llm,
    tools=[write_document, edit_document, read_document],
    system_prompt=(
        "You can read, write and edit documents based on note-taker's outlines. "
        "Don't ask follow-up questions."
    )
)

def doc_writing_node(state:State) -> Command[Literal["supervisor"]]:
    result = doc_writer.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="doc_writer")
            ]
        },
        #We Want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor"
    )

note_taking_agent = create_agent(
    llm, 
    tools=[create_outline, read_document],
    system_prompt=(
        "You can read documents and create outlines for the writer. "
        "Don't ask follow-up questions"
    ))

def note_taking_node(state:State)-> Command[Literal["supervisor"]]:
    result = note_taking_agent.invoke(state)
    return Command(
        update={
            "messages":[
                HumanMessage(content=result["messages"][-1].content, name="note_taker")
            ]
        },
        #We Want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor"
    )

chart_generating_agent = create_agent(
    llm, tools=[read_document, python_repl_tool]
)

def chart_generating_node(state:State) -> Command[Literal["supervisor"]]:
    result = chart_generating_agent.invoke(state)
    return Command(
        update={
            "messages":[
                HumanMessage(
                    content=result["messages"][-1].content, name="chart_generator"
                )
            ]
        },
        goto="supervisor"
    )

doc_writing_supervisor_node = make_supervisor_node(llm, ["doc_writer", "note_taker", "chart_generator"])

paper_writing_builder = StateGraph(State)
paper_writing_builder.add_node("supervisor", doc_writing_supervisor_node)
paper_writing_builder.add_node("doc_writer", doc_writing_node)
paper_writing_builder.add_node("note_taker", note_taking_node)
paper_writing_builder.add_node("chart_generator", chart_generating_node)

paper_writing_builder.add_edge(START, "supervisor")
paper_writing_graph = paper_writing_builder.compile()

teams_supervisor_node = make_supervisor_node(llm, ["research_team", "writing_team"])

def call_research_team(state:State) -> Command[Literal["supervisor"]]:
    # Only pass relevant context to the sub-graph
    trimmed_messages = trimmer.invoke(state["messages"])
    response = research_graph.invoke({"messages": trimmed_messages})
    return Command(
        update={
            "messages":[
                HumanMessage(
                    content=response["messages"][-1].content, name="research_team"
                )
            ]
       },
       goto="supervisor"
    )

def call_paper_writing_team(state:State)-> Command[Literal["supervisor"]]:
    # Only pass relevant context to the sub-graph
    trimmed_messages = trimmer.invoke(state["messages"])
    response = paper_writing_graph.invoke({"messages": trimmed_messages})
    return Command(
        update={
            "messages":[
                HumanMessage(
                    content=response["messages"][-1].content, name="writing_team"               )
            ]
        },
        goto="supervisor",
    )

super_builder = StateGraph(State)
super_builder.add_node("supervisor", teams_supervisor_node)
super_builder.add_node("research_team", call_research_team)
super_builder.add_node("writing_team", call_paper_writing_team)
super_builder.add_edge(START, "supervisor")
super_graph = super_builder.compile()

response  = super_graph.invoke(
    {
        "messages":[
            ("user","Check my internal research notes for transformer evaluation highlights.")
        ],
    }
)

print(response["messages"][-1].content)