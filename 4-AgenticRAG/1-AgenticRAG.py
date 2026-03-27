import os
os.environ["USER_AGENT"] = "AgenticRAG/1.0"
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_classic import hub
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools.retriever import create_retriever_tool
from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition


load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
urls = [
    "https://langchain-ai.github.io/langgraph/tutorials/introduction/",
    "https://langchain-ai.github.io/langgraph/tutorials/workflows/",
    "https://langchain-ai.github.io/langgraph/tutorials/how-tos/map-reduce/"
]

docs = [WebBaseLoader(url).load() for url in urls]
doc_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
doc_splits = text_splitter.split_documents(doc_list)

##Add all these text to vectordb
vectorstore = FAISS.from_documents(
    documents=doc_splits,
    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)
retriever = vectorstore.as_retriever()


### Retriever To Retriever Tools
retriever_tool = create_retriever_tool(
    retriever,
    "retriever_vector_db_blog",
    "Search and run information about Langgraph"
)

langchain_urls = [
    "https://python.langchain.com/docs/tutorials/",
    "https://python.langchain.com/docs/tutorials/chatbot/",
    "https://python.langchain.com/docs/tutorials/qa_chat_history"
]

docs = [WebBaseLoader(url).load() for url in langchain_urls]
doc_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
doc_splits = text_splitter.split_documents(doc_list)

###Add all these text to vectordb
vectorstorelangchain = FAISS.from_documents(
    documents=doc_splits,
    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)

retrieverlangchain = vectorstorelangchain.as_retriever()
retriever_tool_langchain = create_retriever_tool(
    retrieverlangchain,
    "retriever_vector_db_langchain",
    "Search and run information about Langchain"
)

tools = [retriever_tool, retriever_tool_langchain]


###LangGraph Workflow
class AgentState(TypedDict):
    #The add messages function defines how an update should be processed
    #Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]

llm = ChatGroq(model_name="qwen/qwen3-32b")

def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state
    Returns:
        dict: The updated state with the agent response appended to message
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    model = ChatGroq(model="qwen/qwen3-32b")
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    #We return a list, because this will be added to the existing list
    return {"messages": [response]}


###Edges
def grade_documents(state)-> Literal["generate", "rewrite"]:
    """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (messages): The current state
        
        Returns:
            str:A descision for whether the documents are relevant or not
    """
    print("-----CHECK RELEVANCE-----")

    #Data Model
    class grade(BaseModel):
        """Binary score for relevance check."""
        binary_score: str = Field(description="Relevance score 'yes' or 'no' ")
    
    # LLM
    model = ChatGroq(model="qwen/qwen3-32b")
    
    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    #Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing the relevance of a retrieved document to a user question. \n
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.
        """,
        input_variables=["context", "question"],
    )

    #Chain
    chain = prompt | llm_with_tool

    messages = state['messages']
    last_message = messages[-1]
    
    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"context": docs, "question": question})
    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT-----")
        return "generate"
    else:
        print("---DECISION: DOCS NOT RELEVANT-----")
        return "rewrite"


def generate(state):
    """
    Generate answer
    
    Args: 
        state (messages); The current state
    
    Returns:
        dict: The updated message
    """
    print("---GENERATE---")
    messages = state['messages']
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    #Prompt
    prompt = hub.pull("rlm/rag-prompt")

    #LLM
    llm = ChatGroq(model="qwen/qwen3-32b")

    #Chain
    rag_chain = prompt | llm | StrOutputParser()

    #Run
    response = rag_chain.invoke({"context":docs, "question":question})
    return {"messages":[response]}

def rewrite(state):
    """
        Transform the query to produce a better query

        Args:   
            state (messages): The current state
        
        Returns:
            dict: The updated state with pre-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f"""
    Look at the input  and try to reason about the underlying semantic intent / meaning. \n
    Here is the initial question:
    \n ------------- \n
    {question}
    Formulate an improved question:
    """,
        )
    ]

    #Grader
    model = ChatGroq(model="qwen/qwen3-32b")
    response = model.invoke(msg)
    return {"messages": [response]}


#Define new graph
workflow = StateGraph(AgentState)

#Define the nodes we will cycle between
workflow.add_node("agent", agent) # agent
retrieve = ToolNode([retriever_tool, retriever_tool_langchain])
workflow.add_node("retrieve", retrieve) # retrieval
workflow.add_node("rewrite", rewrite) #Re-writing the question
workflow.add_node(
    "generate", generate
) #Generating a response after we know the documents are relevant
#Call agent node to decide to retrieve or not
workflow.add_edge(START, "agent")


#Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    #Assess agent decision
    tools_condition,
    {
        #Translate the condition outputs to nodes in our graph
        "tools":"retrieve",
        END:END,
    },

)

#Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    #Assess agent descision
    grade_documents,
)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

#Compile
graph = workflow.compile()

if __name__ == '__main__':
    query = "What is Langgraph?"
    state = {"messages": [HumanMessage(content=query)]}
    result = graph.invoke(state)
    print("\n Final Answer:\n", result["messages"][-1].content)