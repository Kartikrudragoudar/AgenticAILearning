import os
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
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
chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, verbose=True, allow_dangerous_requests=True)
response = chain.invoke({"query":"Who were the actors of the movie Casino?"})
print(response)
