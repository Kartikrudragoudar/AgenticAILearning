import os
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama-3.3-70b-versatile")

uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
database = os.getenv("NEO4J_DATABASE")


graph = Neo4jGraph(
    url=uri,
    username=username,
    password=password,
    database=database,
)

### Dataset Movie
movie_query = """
LOAD CSV WITH HEADERS FROM
'https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/movies/movies_small.csv' AS row

MERGE(m:Movie{id:row.movieId})
SET m.released = date(row.released),
    m.title = row.title,
    m.imdbRating = toFloat(row.imdbRating)

FOREACH (director in split(row.director, '|') |
    MERGE (p:Person {name:trim(director)})
    MERGE (p)-[:DIRECTED]->(m))
FOREACH (actor in split(row.actors, '|') |
    MERGE (p:Person {name:trim(actor)})
    MERGE (p)-[:ACTED_IN]->(m))
FOREACH (genre in split(row.genres, '|') |
    MERGE (g:Genre {name:trim(genre)})
    MERGE (m)-[:IN_GENRE]->(g))
"""

graph.query(movie_query)
graph.refresh_schema()

chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, exclude_types=["Genre"], verbose=True, allow_dangerous_requests=True)

examples = [
    {
        "question":"How many artists are there?",
        "query":"MATCH (a:Person)-[:ACTED_IN]->(:Movie) RETURN count(DISTINCT a)"
    },
    {
        "question":"Which actors played in the movie Casino?",
        "query":"MATCH (m:Movie {{title: 'Casino'}})<-[:ACTED_IN]-(a) RETURN a.name"
    },
    {
        "question":"How many movies has Tom Hanks acted in?",
        "query":"MATCH (a:Person {{name: 'Tom Hanks'}}) -[:ACTED_IN]->(m:Movie) RETURN count(m)"
    },
    {
        "question":"List all the genres of the movie Schindler's List",
        "query":"MATCH (m:Movie {{title: 'Schindler\\\\'s List'}})-[:IN_GENRE]->(g:Genre) RETURN g.name"
    },
    {
        "question":"Which actors have worked in movies from both the comedy and action genres?",
        "query":"MATCH (a:Person)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g1:Genre), (a)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g2:Genre) WHERE g1.name = 'Comedy' AND g2.name = 'Action' RETURN DISTINCT a.name"
    },
    {
        "question":"Which directors have made movies with at least three different actors named 'John'?",
        "query":"MATCH (d:Person)-[:DIRECTED]->(m:Movie)<-[:ACTED_IN]-(a:Person) WHERE a.name STARTS WITH 'John' WITH d, COUNT(DISTINCT a) AS JohnsCount WHERE JohnsCount >= 3 RETURN d.name"
    },
    {
        "question":"Identify movies where directors also played a role in the film.",
        "query":"MATCH (p:Person)-[:DIRECTED]->(m:Movie), (p)-[:ACTED_IN]->(m) RETURN m.title, p.name"
    },
    {
        "question":"Find the actor with the highest number of movies in the database.",
        "query":"MATCH (a:Actor)-[:ACTED_IN]->(m:Movie) RETURN a.name, COUNT(m) AS movieCount ORDER BY movieCount DESC LIMIT 1"
    }
]

example_prompt = PromptTemplate.from_template(
    "User input:{question}\nCypher Query:{query}"
)

prompt = FewShotPromptTemplate(
    examples=examples[:5],
    example_prompt=example_prompt,
    prefix='You are Neo4j expert. Given an input question, Create a syntactically very accurate Cypher Query',
    suffix="User input: {question}\nCypher query: ",
    input_variables=["question","schema"]
)

print(prompt.format(question="How many artists are there?", schema="foo"))
