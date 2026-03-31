import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List
from typing_extensions import TypedDict
from langchain_classic import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph, START


load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")

llm = ChatGroq(model="qwen/qwen3-32b")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


urls = [
    'https://lilianweng.github.io/posts/2023-06-23-agent/',
    'https://lilianweng.github.io/posts/2023-03-15-prompt-engineering',
    'https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/'
]

# Load
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

vectorstore=FAISS.from_documents(
    documents=doc_splits,
    embedding=embeddings
)

retriever=vectorstore.as_retriever()


##Data Model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant, 'yes' or 'no'")


structured_llm_grader = llm.with_structured_output(GradeDocuments)

#prompt
system = """
            You are a grader assessing relevance of a retrieved document to a user question. \n
            If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        """

grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Retrieved document : \n\n {document} \n\n User question: {question}"),
    ]

)

retriever_grader = grade_prompt | structured_llm_grader
question = "agent memory"
docs = retriever.invoke(question)
doc_txt = docs[1].page_content

##prompt
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


#Chain
rag_chain = prompt | llm | StrOutputParser()

#Run
generation = rag_chain.invoke({"context":docs, "question":question})


### Question Re-writer
system = """
        You a question re-writer that converts an input question to a better version that is optimized \n
        for web search. Look at the input and try to reason about the underlying semantic intent / meaning.
"""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            'human',
            "Here is the initial question: \n\n {question} \n Formulate an improved question."
        ),
    ]
)

question_re_writer = re_write_prompt | llm | StrOutputParser()
question_re_writer.invoke({"question":question})

web_search_tool = TavilySearch(k=3)


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM Generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]

def retrieve(state):
    """
        Retrieve documents

        Args:
            state (dict): The current graph state
        
        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("----RETRIEVE----")
    question = state["question"]
    
    #Retrieval 
    documents = retriever.invoke(question)
    return {"documents":documents, "question":question}

def generate(state):
    """
        Generate Answer

        Args:
            state (dict): The current graph state
        
        Returns:
            state (dict): New key added to state, generation, that contains LLM Generation
    """
    print("----GENERATE----")
    question = state["question"]
    documents = state["documents"]
    
    #RAG Generation
    generation = rag_chain.invoke({"context":documents, "question":question})
    return {"generation":generation, "documents":documents, "question":question}


def grade_documents(state):
    """ 
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state
    
    Returns:
        state (dict): New key added to state, web_search, that contains whether to add search
    """
    print("----GRADE_DOCUMENTS----")
    question = state["question"]
    documents = state["documents"]
    
    #Score each doc
    filtered_docs = []
    web_search = "No"

    for d in documents:
        score = retriever_grader.invoke(
            {"question":question, "document":d.page_content}
        )
        grade  = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "YES"
            continue
    
    return {"documents":filtered_docs, "web_search":web_search} 

def transform_query(state):
    """
        Transform the question to be more suitable for web search.

        Args:
            state (dict): The current graph state
        
        Returns:
            state (dict): New key added to state, question, that contains the transformed question
    """
    print("----TRANSFORM_QUERY----")
    question = state["question"]
    
    #Transform the question
    better_question = question_re_writer.invoke({"question":question})
    return {"question":better_question, "documents":state["documents"]}

def web_search(state):
    """
    Web Search based on the re-phrased question.

    Args:
        State(dict), the current graph state
    
    Returns:
        state(dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state['documents']

    #Web Search
    docs = web_search_tool.invoke({"query":question})
    web_results = "\n".join([d["content"] if isinstance(d, dict) else d for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {"documents":documents, "question":question}


##Edges

def decide_to_generate(state):
    """
        Determine whether to generate an answer, or re-generate a question.
    
    Args:
        state(dict): The current graph state
    Returns:
        str: Binary decision for next node to call    
    """
    print("---ASSESS GRADED DOCUMENTS---")
    state['question']
    web_search=state["web_search"]
    state["documents"]

    if web_search == "YES":
        # ALl documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "----DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        #We have relevant documents, generate answer
        print("----DECISION: GENERATE---")
        return "generate"

workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node('generate', generate)
workflow.add_node('transform_query', transform_query)
workflow.add_node('web_search_node', web_search)

# Build graph
workflow.add_edge(START, 'retrieve')
workflow.add_edge('retrieve', 'grade_documents')
workflow.add_conditional_edges(
    'grade_documents',
    decide_to_generate,
    {
        "transform_query":"transform_query",
        "generate":"generate",
    }
)
workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "generate")
workflow.add_edge("generate", END)

# Complile
app = workflow.compile()

# Run
inputs = {"question":"What is the capital of France?"}
print(app.invoke(inputs))