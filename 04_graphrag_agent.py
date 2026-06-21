import os
from langchain_neo4j import Neo4jGraph
from langchain_neo4j import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENROUTER_API_KEY

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is not set.")

# 1. Initialize the Neo4j Graph connection for LangChain
print("Connecting to Neo4j Graph...")
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD
)

# Refresh the schema so LangChain knows what nodes and relationships exist
graph.refresh_schema()
print("Graph schema refreshed successfully.")

# 2. Configure the LLM for Text-to-Cypher and Response Synthesis
# Using Kimi by Moonshot via OpenRouter
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    model="moonshotai/kimi-k2",
    max_tokens=1000,
    temperature=0.0
)

# 3. Create a custom Cypher generation prompt
cypher_generation_template = """Task: Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}

Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any formatting like ```cypher, just the plain query.
Make sure the Cypher query is read-only (MATCH and RETURN only).
For symptoms, use case-insensitive matching where possible or exact match if preferred. 

Example 1:
User: "I have a fever, cough, and shortness of breath."
Cypher: MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom) WHERE toLower(s.name) IN ['fever', 'cough', 'shortness of breath'] RETURN d.name, collect(s.name) AS matched_symptoms, count(s) AS match_count ORDER BY match_count DESC LIMIT 5

Example 2:
User: "What medications treat Malaria?"
Cypher: MATCH (d:Disease)-[:TREATED_WITH]->(m:Medication) WHERE toLower(d.name) = 'malaria' RETURN m.name

The question is:
{question}
"""
cypher_prompt = PromptTemplate(
    input_variables=["schema", "question"],
    template=cypher_generation_template
)

# 4. Initialize the GraphCypherQAChain
# We pass the LLM for both cypher generation and answering, but you can separate them if needed.
chain = GraphCypherQAChain.from_llm(
    cypher_llm=llm,
    qa_llm=llm,
    graph=graph,
    cypher_prompt=cypher_prompt,
    verbose=True,
    return_intermediate_steps=True,
    allow_dangerous_requests=True
)

def ask_medigraph(query: str):
    """
    Takes a natural language query, generates a Cypher query, 
    retrieves data from Neo4j, and synthesizes a response.
    """
    print(f"\nUser Query: {query}")
    try:
        response = chain.invoke({"query": query})
        
        print("\n=== Intermediate Steps ===")
        for step in response.get("intermediate_steps", []):
            if "query" in step:
                print(f"Generated Cypher: {step['query']}")
            if "context" in step:
                print(f"Retrieved Context: {step['context']}")
                
        print("\n=== Final Answer ===")
        print(response.get("result", "No answer could be generated."))
        return response
    except Exception as e:
        print(f"Error during GraphRAG execution: {e}")

if __name__ == "__main__":
    # Test queries
    test_queries = [
        "I have a runny nose, sneezing, and an itchy throat. What disease could this be, and what medications are recommended?"
    ]
    
    for q in test_queries:
        ask_medigraph(q)
