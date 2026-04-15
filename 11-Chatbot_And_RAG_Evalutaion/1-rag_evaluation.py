import os
from dotenv import load_dotenv
from langsmith.client import Client
from langchain_groq import ChatGroq
from langsmith import wrappers
from groq import Groq
load_dotenv()
client = Client()


os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"


### Create the datapoints - these are your test data

dataset_name = "Simple Chatbots Evaluation"
if not client.has_dataset(dataset_name=dataset_name):
    dataset=client.create_dataset(dataset_name)

    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
            "inputs": {"question": "What is LangChain?"},
            "outputs": {"answer": "LangChain is a framework designed to simplify the creation of applications using large language models (LLMs)."}
            },
            {
                "inputs": {"question": "Who developed LangChain?"},
                "outputs": {"answer": "LangChain was launched by Harrison Chase in 2022."}
            },
            {
                "inputs": {"question": "What is OpenAI?"},
                "outputs": {"answer": "OpenAI is an AI research and deployment company."}
            },
            {
                "inputs": {"question": "What is Google?"},
                "outputs": {"answer": "Google is an American multinational technology company."}
            }
        ]
    )


### Define Metrics (LLM As A Judge)
evaluator_llm = ChatGroq(model="qwen/qwen3-32b")
eval_instructions = "You are an expert professor grading a student's answer."

def correctness(inputs:dict, outputs:dict, reference_outputs:dict)-> bool:
    user_content= f"""You are grading the following questions:
        {inputs['question']}
            Here is the real answer:
        {reference_outputs['answer']}
            You are grading the following predicted answer:
        {outputs['response']}
            Respond with CORRECT or INCORRECT
        Grade:
    """

    response = evaluator_llm.invoke([
        ("system", eval_instructions),
        ("human", user_content)
    ])
    return "CORRECT" in response.content.strip().upper()


### Concisions - checks whether the actual output is less than 2x the length of the expected result.
def concision(outputs: dict, reference_outputs:dict) -> bool:
    return int(len(outputs['response']) < 2 * len(reference_outputs['answer']))


###RUN EVALUATIONS
default_instructions = "Respond to the users questions in a short, concise answer (one shot sentence)."

def my_app(question:str, model:str="qwen/qwen3-32b", instructions:str=default_instructions):
    return evaluator_llm.invoke([
        ("system", instructions),
        ("human", question)
    ]).content

### Call my_app for every datapoints
def ls_target(inputs:str)-> dict:
    return {"response":my_app(inputs["question"], model="qwen/qwen3-32b")}

### Run our evalutaion
experimen_results = client.evaluate(
    ls_target,
    data=dataset_name,
    evaluators=[correctness, concision],
    experiment_prefix="qwen/qwen3-32b-chatbot-eval"
)