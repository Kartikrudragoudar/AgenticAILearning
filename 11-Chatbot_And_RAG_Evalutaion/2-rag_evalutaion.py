import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langsmith import traceable
from langsmith.client import Client
from typing_extensions import Annotated, TypedDict
load_dotenv()
client = Client()


os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"



urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

#Load documents from the URLs
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

#Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250,
    chunk_overlap=50
)

doc_splits = text_splitter.split_documents(docs_list)

vector_store = InMemoryVectorStore.from_documents(
    documents=doc_splits,
    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)

retriever = vector_store.as_retriever(k=6)

llm = ChatGroq(model="llama-3.3-70b-versatile")

### Add decorator
@traceable()
def rag_bot(question:str)->dict:
    docs = retriever.invoke(question)
    docs_string = " ".join(doc.page_content for doc in docs)
    instructions = f"""You are helpful assistant who is good at analyzing source information and answer
        Documents:
        {docs_string}"""

    ### llm invoke
    ai_msg=llm.invoke([
        {"role":"system", "content":instructions},
        {"role":"human", "content":question},
    ])
    return {"answer":ai_msg.content, "documents":docs}

# print(rag_bot("What is agents?")["response"])

examples = [
    {
        "inputs":{"question":"How does the ReAct agent use self-reflection?"},
        "outputs":{"answer":"ReAct integrates reasoning and acting, performing actions - such tools like Wikipedia search API - and then observing / reasoning about the tool outputs."}
    },
    {
        "inputs":{"question":"What are the types of biases that can arise with few-shot prompting?"},
        "outputs":{"answer":"The biases that can arise with few-shot prompting include (1) Majority label bias, (2) Recency bias, and (3) Common token bias."}
    },
    {
        "inputs":{"question":"What are five types of adversarial attacks?"},
        "outputs":{"answer":"Five types of adversarial attacks are (1) Token manipulation, (2) Gradient based attack, (3) Jailbreak prompting, (4) Human red-teaming, (5) Model red-teaming."}
    }
]

### create the dataset and examples in langsmith
dataset_name = "Simple RAG Test Evaluation"
if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(
        dataset_id=dataset.id,
        examples=examples
    )
else:
    dataset = client.read_dataset(dataset_name=dataset_name)


## Correctness Output Schema
class CorrectnessGrade(TypedDict):
    explanation:Annotated[str, "Explain you reasoning for the score"]
    correct:Annotated[bool, "True if the anwer is correct, False otherwise"]

## correctness prompt
correctness_instructions = """You are a teacher grading quiz.

You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer.
(2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the  ground truth answer.

Correctness:
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.

Avoid simply stating the correct answer at the outset."""

llm2 = ChatGroq(model="llama-3.3-70b-versatile").with_structured_output(CorrectnessGrade)

def correctness(inputs:dict, outputs:dict, reference_outputs:dict) -> bool:
    """An evaluator for RAG answer accuracy"""
    answers = f"""\
QUESTION: {inputs['question']}
GROUND TRUTH ANSWER: {reference_outputs['answer']}
STUDENT ANSWER: {outputs["answer"]}"""
    
    #Run evaluator
    grade = llm2.invoke([
        {"role":"system", "content":correctness_instructions},
        {"role":"human", "content":answers}
    ])
    return grade["correct"]

#Grade output schema
class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant:Annotated[bool, ..., "Provide the score on whether the answer address the question"]

#Grade prompt
relevance_instructions = """You are a teacher grading a quiz.
You will be given a QUESTION and a STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.
Avoid simply stating the correct answer at the outset.
"""

llm3 = ChatGroq(model="llama-3.3-70b-versatile").with_structured_output(RelevanceGrade)

def relevance(inputs:dict, outputs:dict, reference_outputs:dict) -> bool:
    """A simple evaluator for RAG answer helpfulness"""
    answer = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = llm3.invoke([
        {"role":"system", "content":relevance_instructions},
        {"role":"user", "content":answer}
    ])
    return grade["relevant"]



class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[bool, ..., "Provide the score on if the answer hallucinations from the documents"]

#Grade prompt
grounded_instructions = """You are a teacher grading a quiz.

Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS.
(2) Ensure the STUDENT ANSWER does not contain \"hallucinated\" information outside the scope of the FACTS.

Grounded:
A grounded value of True means that the student's answer meets all of the criteria.
A grounded value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 
Avoid simply stating the correct answer at the outset.
"""
llm4 = ChatGroq(model="llama-3.3-70b-versatile").with_structured_output(GroundedGrade)

#Evaluator
def groundedness(inputs: dict, outputs: dict)-> bool:
    """A simple evaluator for RAG answer groundedness"""
    doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
    answer = f"FACTS: {doc_string}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = llm4.invoke([{"role":"system", "content":grounded_instructions}, {"role":"user", "content":answer}])
    return grade["grounded"]


### Retrieval Relevance: Retrieved docs vs input
class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "True if the retrieved documents are relevant to the question, False otherwise"]

# Grade prompt
retrieval_relevance_instructions = """You are a teacher grading a quiz. 
        You will be given a QUESTION and a set of FACTS provided by the student. 

        Here is the grade criteria to follow:
        (1) You goal is to identify FACTS that are completely unrelated to the QUESTION
        (2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
        (3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

        Relevance:
        A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
        A relevance value of False means that the FACTS are completely unrelated to the QUESTION.
        Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.
        Avoid simply stating the correct answer at the outset.
    """


llm5 = ChatGroq(model="llama-3.3-70b-versatile").with_structured_output(RetrievalRelevanceGrade)

def retrieval_relevance(inputs: dict, outputs: dict) -> bool:
    """An evaluator for document relevance"""
    doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
    answer = f"FACTS: {doc_string}\nQUESTION: {inputs["question"]}"

    grade = llm5.invoke([
        {"role":"system", "content":retrieval_relevance_instructions},
        {"role":"user", "content":answer}
    ])
    return grade["relevant"]

def target(inputs:dict)-> dict:
    return rag_bot(inputs["question"])

experimental_results = client.evaluate(
    target,
    data=dataset_name,
    evaluators=[correctness, groundedness, relevance, retrieval_relevance],
    experiment_prefix="rag-doc-relevance",
    metadata={"version":"LCEL context, GROQ-MODEL-PREVIEW"},
)

# Explore results locally as a dataframe if you have pandas installed 
experimental_results.to_pandas()