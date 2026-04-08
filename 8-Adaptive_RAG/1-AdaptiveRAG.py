import os
from langgraph.graph import END, StateGraph, START
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from typing import Literal, List
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langchain_classic import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily import TavilySearch
load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")

llm = ChatGroq(model="openai/gpt-oss-120b")
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

# Data Model for structured output routing
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    # Define available routing options: internal vectorstore or external web_search
    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

# Attaches the schema to the LLM to ensure it returns data in the RouteQuery format
Structured_llm_router = llm.with_structured_output(RouteQuery)

# System logic defining the criteria for choosing between vectorstore and web search
System="""You are an expert at routing a user question to a vectorstore pr websearch.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use web-search."""

# Templates the messages to properly format the input for the router LLM
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",System),
        ("human", "Question: {question}")
    ]
)

# Combines the prompt and structured LLM into a single callable router chain
question_router = route_prompt | Structured_llm_router

# print(
#     question_router.invoke({
#         "question":"What are the types of agents memory?"
#     })
# )

## Retrieval Grader
# Data Model for structured output of relevance scores
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    # Defines 'yes' or 'no' relevance field for the LLM output
    binary_score: str = Field(
        description="Documents are relevant to question, 'yes' or 'no'"
    )

# Configures the LLM to output structured relevance grade data
Structured_llm_grader = llm.with_structured_output(GradeDocuments)

# System instruction defining how the LLM should judge document relevance
System2 = """You are a grader assessing relevance of a retrieved document to a user question. \n
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""

# Format for providing both the document and question to the grader
grade_prompt=ChatPromptTemplate.from_messages(
    [
        ("system",System2),
        ("human", "Retrieved document: \n\n {document} \n\n User Question: {question}")
    ]
)

# Chains the grading prompt and the structured LLM into a reusable document grader
retrieval_grader = grade_prompt | Structured_llm_grader
question = "agent memory"
docs = retriever.invoke(question)
doc_txt = docs[1].page_content


#Generate
# Pulls a standard RAG prompt template from the LangChain Hub
prompt = hub.pull("rlm/rag-prompt")

#Post-Processing
# Utility function to merge retrieved document contents into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


#Chain
# Constructs the core RAG sequence: prompt + LLM + string parser
rag_chain = prompt | llm | StrOutputParser()

# Passes ONLY the text content of the retrieved snippets to stay under the token limit
# Executes the RAG chain using the initial question and retrieved context
generation = rag_chain.invoke({"context": format_docs(docs), "question": question})
# print(generation)

# Schema for binary hallucination checks (grounded or not)
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

Structured_llm_grader2 = llm.with_structured_output(GradeHallucinations)

system3 = """You are a grader assessing whether an LLM generation is grounded in or supported by a set of retrieved facts. \n 
Give a binary score 'yes' or 'no' means that the answer is grounded in or supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system3),
        ("human", "Set of facts: \n\n{documents} \n\n LLM generation: {generation}"),
    ]
)

# Chain to verify if the LLM generation is strictly supported by facts
hallucination_grader = hallucination_prompt | Structured_llm_grader2
hallucination_grader.invoke({
    "documents":format_docs(docs),
    "generation":generation
})


# Answer Grader

#Data Model
# Schema for binary answer quality checks (addresses question or not)
class GradeAnswer(BaseModel):
    """Binary score to assess answer to addresses question."""

    binary_score:str=Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

Structured_llm_grader3 = llm.with_structured_output(GradeAnswer)

System4="""You are a grader assessing whether an answer addresses / resolves a question \n
Give a binary score 'yes' or 'no'. Yes means that the answer resolves the question."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", System4),
        ("human", "Question: {question} \n\n LLM generation: {generation}"),
    ]
)

# Chain to verify if the final answer actually resolves the user's query
answer_grader = answer_prompt | Structured_llm_grader3
answer_grader.invoke({"question":question, "generation":generation})

System5 = """
You are a question re-writer that converts an input question to a better version that is optimized \n
for vectorestore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning.
"""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",System5),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question."
        )
    ]
)

# Chain to optimize search queries for better vectorstore retrieval
question_rewriter = re_write_prompt | llm | StrOutputParser()
# print(question_rewriter.invoke({"question":question}))

# External search tool for fallbacks when vectorstore data is insufficient
web_search_tool = TavilySearch(k=3)

# Defines the schema for data flowing through the LangGraph workflow
class GraphState(TypedDict):
    """
    Represents the state of our state

    Attributes:
        question: question
        generation: LLM Generation
        documents: list of documents
    """
    question: str
    generation: str
    documents: List[str]


# Node for fetching relevant documents from the vectorstore retriever
def retrieve(state):
    """
    Retrieve Documents
    
    Args:
        state (dict): The current graph state
    
    Returns:
        state(dict): New key added to state, documents, that contains retrieved documents
    """
    print("----RETRIEVE----")
    question = state["question"]
    
    #Retrieval
    documents = retriever.invoke(question)
    
    return {"documents": documents, "question":question}


# Node for synthesizing an answer using the RAG chain and context
def generate(state):
    """
    Generate Answer

    Args:
        state(dict): The current graph state
    
    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("----GENERATE----")
    question = state["question"]
    documents = state["documents"]
    
    #RAG Generation
    generation = rag_chain.invoke({"context":format_docs(documents), "question":question})
    return {"generation":generation, "documents":documents, "question":question}

# Node for filtering out documents that are not relevant to the query
def grade_documents(state):
    """
    Determines whether the retieved documents are relevant to the question.

    Args:
        state(dict): The current graph state
    
    Returns:
        state(dict): Updates documents key with only filtered relevant documents
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    #Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({
            "question":question, "document":d.page_content
        })
        grade = score.binary_score
        if grade == 'yes':
            print("--GRADE: DOCUMENT RELEVANT--")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT--")
            continue
    
    return {"documents":filtered_docs, "question":question}

# Node for re-phrasing the question to improve retrieval success
def transform_query(state):
    """
    Transform the question to produce a better question.

    Args:
        state (dict): The current graph state
    
    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state['question']
    documents = state['documents']
    
    #Re-write question
    better_question = question_rewriter.invoke({"question":question})
    return {"documents":documents, "question":better_question}

# Node for performing a web search as a fallback datasource
def web_search(state):
    """
        Web Search based on the re-phrased question

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results       
    """
    print("---WEB SEARCH---")
    question = state['question']

    #Web Search 
    docs = web_search_tool.invoke({"query":question})
    if isinstance(docs, str):
        web_results = docs
    elif isinstance(docs, list):
        # Extract content from list of dicts, documents, or strings
        results = []
        for d in docs:
            if isinstance(d, dict) and "content" in d:
                results.append(d["content"])
            elif hasattr(d, "page_content"):
                results.append(d.page_content)
            else:
                results.append(str(d))
        web_results = "\n".join(results)
    else:
        web_results = str(docs)
    web_results = Document(page_content=web_results)
    
    return {"documents":[web_results], "question":question}


#Edges
# Router function to decide between vectorstore or web search
def route_question(state):
    """
        Route question to web search or RAG.

    Args:
        state (dict): The current graph state
    
    Returns:
        str: Next Node to call
    """

    print("---ROUTE QUESTION---")
    question = state['question']
    source = question_router.invoke({"question":question})
    if source.datasource == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"

# Logical edge to decide between generation or query transformation
def decide_to_generate(state):
    """
        Determines whether to generate an answer, or  re-generate a question.

        Args:
            state (dict): The current graph state
        
        Returns:
            str: Binary decision for next node to call
    """
    print("---ASSES GRADED DOCUMENTS---")
    question = state['question']
    filtered_documents = state['documents']
    
    if not filtered_documents:
        #All documents have been filtered check_relevance
        #We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We Have Relevant Documents, Generate Answer
        print("---DECISION: GENERATE ANSWER---")
        return "generate"
    
# Final validation node to check for hallucinations and answer quality
def grade_generation_v_documents_and_question(state):
    """
        Determines whether the generation is grounded in the document and answers question.

        ARGS:
            state (dict): The current graph state
        
        Returns:
            str: Decision for next node to call
    """
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {
            "documents":format_docs(documents),
            "generation":generation
        }
    )
    grade = score.binary_score

    if grade == "yes":
        print("---GRADE: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({
            "question":question, "generation":generation
        })
        grade = score.binary_score
        if grade == "yes":
            print("---DESCISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DESCISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DESCISION: GENERATION NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

# Initializes and configures the individual nodes of the task graph
workflow = StateGraph(GraphState)
workflow.add_node("web_search", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

# Router logic determining whether to start with vectorstore retrieval or web search
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "web_search":"web_search",
        "vectorstore":"retrieve",
    }
)
# Linear transitions: from search to generation and retrieval to document grading
workflow.add_edge("web_search", 'generate')
workflow.add_edge("retrieve", "grade_documents")
# Validation logic choosing between answer generation or query transformation after document check
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query":"transform_query",
        "generate":"generate"
    }
)

# Final logic choosing to loop back for repairs (not supported/not useful) or finish (useful)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not_supported":"generate",
        "useful":END,
        "not_useful":"transform_query"
    }
)

app = workflow.compile()
response = app.invoke({"question":"What is machine Learning"})
print(response)