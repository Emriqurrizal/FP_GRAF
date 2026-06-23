import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENROUTER_API_KEY

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is not set.")

# 1. Initialize LLM
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    model="moonshotai/kimi-k2",
    max_tokens=1000,
    temperature=0.0
)

# 2. Initialize Neo4j Driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def get_graph_schema():
    """Retrieve graph schema dynamically from Neo4j."""
    schema_parts = []
    with driver.session() as session:
        # Node labels and properties
        node_result = session.run("""
            CALL db.schema.nodeTypeProperties()
            YIELD nodeLabels, propertyName, propertyTypes
            RETURN nodeLabels, collect(propertyName) AS properties
        """)
        schema_parts.append("Node Labels & Properties:")
        for r in node_result:
            schema_parts.append(f"  :{r['nodeLabels']} -> {r['properties']}")

        # Relationship patterns
        pattern_result = session.run("""
            MATCH (a)-[r]->(b)
            RETURN DISTINCT labels(a)[0] AS from_label, type(r) AS rel_type, labels(b)[0] AS to_label
        """)
        schema_parts.append("\nRelationship Patterns:")
        for r in pattern_result:
            schema_parts.append(f"  (:{r['from_label']})-[:{r['rel_type']}]->(:{r['to_label']})")
            
    return "\n".join(schema_parts)


# 3. Create the Text-to-Cypher Prompt (LangChain Template)
TEXT_TO_CYPHER_PROMPT = """You are a Neo4j Cypher expert. Generate a READ-ONLY Cypher query based on the user's question and the graph schema below.

GRAPH SCHEMA:
{schema}

RULES:
1. Output ONLY the Cypher query, nothing else. No explanations, no markdown formatting.
2. Use only MATCH and RETURN (read-only). Never use CREATE, DELETE, SET, MERGE, or DETACH.
3. Use toLower() for case-insensitive string matching.
4. Always LIMIT results (max 10) unless the user asks for all.
5. For symptom-based questions, match symptoms and count overlaps to rank diseases.
6. Node property for names is always .name (lowercase).
7. Disease names in the database are lowercase.

EXAMPLES:
User: "Penyakit apa yang punya gejala fever dan headache?"
Cypher: MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom) WHERE toLower(s.name) IN ['fever', 'headache'] WITH d, collect(s.name) AS matched, count(s) AS cnt RETURN d.name AS disease, matched AS matched_symptoms, cnt AS match_count ORDER BY cnt DESC LIMIT 10

User: "Obat untuk malaria?"
Cypher: MATCH (d:Disease)-[:TREATED_WITH]->(m:Medication) WHERE toLower(d.name) = 'malaria' RETURN d.name AS disease, collect(m.name) AS medications

User: "Rekomendasi diet dan olahraga untuk asthma?"
Cypher: MATCH (d:Disease) WHERE toLower(d.name) = 'asthma' OPTIONAL MATCH (d)-[:RECOMMENDED_DIET]->(dt:Diet) OPTIONAL MATCH (d)-[:RECOMMENDED_WORKOUT]->(w:Workout) RETURN d.name AS disease, collect(DISTINCT dt.name) AS diets, collect(DISTINCT w.name) AS workouts

NOW GENERATE CYPHER FOR THIS QUESTION:
{question}"""

prompt = PromptTemplate.from_template(TEXT_TO_CYPHER_PROMPT)
text_to_cypher_chain = prompt | llm | StrOutputParser()

def text_to_cypher(question: str):
    """Translate natural language question to Cypher using LangChain and execute."""
    print(f"\n{'='*60}")
    print(f" Pertanyaan: {question}")
    print(f"{'='*60}")

    schema = get_graph_schema()
    
    # Generate Cypher via LangChain
    cypher_query = text_to_cypher_chain.invoke({
        "schema": schema, 
        "question": question
    }).strip()
    
    # Clean up markdown if LLM accidentally adds it
    cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()

    print(f"\n Generated Cypher:\n   {cypher_query}")

    # Execute on Neo4j
    try:
        with driver.session() as session:
            result = session.run(cypher_query)
            records = [record.data() for record in result]

        print(f"\n Results ({len(records)} rows):")
        for i, record in enumerate(records, 1):
            print(f"   [{i}] {record}")

        return {"query": cypher_query, "results": records}

    except Exception as e:
        print(f"\nError executing Cypher: {e}")
        return {"query": cypher_query, "error": str(e)}

if __name__ == "__main__":
    print("DEMO: LLM Text-to-Cypher (LangChain Version)\n")

    demo_questions = [
        "Penyakit apa saja yang memiliki gejala fever dan headache?",
        "Obat apa yang direkomendasikan untuk common cold?",
        "Berikan rekomendasi diet dan olahraga untuk diabetes"
    ]

    try:
        for q in demo_questions:
            text_to_cypher(q)
            print()
    finally:
        driver.close()
