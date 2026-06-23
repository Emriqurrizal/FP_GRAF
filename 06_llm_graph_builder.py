import os
import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from neo4j import GraphDatabase
from config import OPENROUTER_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is not set.")

# Initialize LLM
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    model="moonshotai/kimi-k2", 
    max_tokens=2000,
    temperature=0.0
)

# Initialize Neo4j Driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

GRAPH_BUILDER_PROMPT = """You are a medical knowledge graph builder. Extract entities and relationships from the given medical text.

OUTPUT FORMAT (strict JSON, no markdown):
{{
  "entities": [
    {{"type": "Disease", "name": "disease name in lowercase"}},
    {{"type": "Symptom", "name": "symptom name in lowercase"}},
    {{"type": "Medication", "name": "medication name in lowercase"}},
    {{"type": "Precaution", "name": "precaution in lowercase"}},
    {{"type": "Diet", "name": "diet recommendation in lowercase"}},
    {{"type": "Workout", "name": "exercise recommendation in lowercase"}}
  ],
  "relationships": [
    {{"from": "entity name", "from_type": "Disease", "relation": "HAS_SYMPTOM", "to": "entity name", "to_type": "Symptom"}},
    {{"from": "entity name", "from_type": "Disease", "relation": "TREATED_WITH", "to": "entity name", "to_type": "Medication"}},
    {{"from": "entity name", "from_type": "Disease", "relation": "HAS_PRECAUTION", "to": "entity name", "to_type": "Precaution"}},
    {{"from": "entity name", "from_type": "Disease", "relation": "RECOMMENDED_DIET", "to": "entity name", "to_type": "Diet"}},
    {{"from": "entity name", "from_type": "Disease", "relation": "RECOMMENDED_WORKOUT", "to": "entity name", "to_type": "Workout"}}
  ]
}}

RULES:
1. Extract ALL entities mentioned in the text.
2. Only use the entity types and relationship types listed above.
3. All names should be lowercase.
4. Output valid JSON only, no explanation.

MEDICAL TEXT:
{text}"""

def extract_entities_from_text(raw_text):
    """Extract entities and relationships from medical text using LLM."""
    print(f"\n{'='*60}")
    print(f"LLM Graph Builder — Entity Extraction")
    print(f"{'='*60}")
    print(f"\n Input Text:\n   {raw_text[:200]}{'...' if len(raw_text) > 200 else ''}\n")

    prompt = PromptTemplate.from_template(GRAPH_BUILDER_PROMPT)
    chain = prompt | llm
    
    # Run the LLM
    response = chain.invoke({"text": raw_text}).content

    # Parse JSON from response
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            extracted = json.loads(json_match.group())
        else:
            extracted = json.loads(response)

        entities = extracted.get("entities", [])
        relationships = extracted.get("relationships", [])

        print(f"  Extracted {len(entities)} entities:")
        for e in entities:
            print(f"     • [{e['type']}] {e['name']}")

        print(f"\n  Extracted {len(relationships)} relationships:")
        for r in relationships:
            print(f"     • ({r['from']}) -[{r['relation']}]-> ({r['to']})")

        return extracted

    except Exception as e:
        print(f"  Error parsing LLM response: {e}")
        print(f"  Raw response: {response[:300]}")
        return {"entities": [], "relationships": []}

def populate_graph_from_extraction(extracted):
    """Insert extracted entities and relationships into Neo4j."""
    print(f"\n{'='*60}")
    print(f" Populating Neo4j from Extracted Data")
    print(f"{'='*60}\n")

    entities = extracted.get("entities", [])
    relationships = extracted.get("relationships", [])

    valid_types = {"Disease", "Symptom", "Medication", "Precaution", "Workout", "Diet"}
    valid_rels = {"HAS_SYMPTOM", "TREATED_WITH", "HAS_PRECAUTION", "RECOMMENDED_WORKOUT", "RECOMMENDED_DIET"}

    with driver.session() as session:
        # Insert entities
        for entity in entities:
            etype = entity.get("type")
            ename = entity.get("name", "").strip().lower()
            if etype in valid_types and ename:
                session.run(f"MERGE (n:{etype} {{name: $name}})", name=ename)
                print(f"  MERGE :{etype} -> {ename}")

        # Insert relationships
        for rel in relationships:
            from_type = rel.get("from_type")
            to_type = rel.get("to_type")
            relation = rel.get("relation")
            from_name = rel.get("from", "").strip().lower()
            to_name = rel.get("to", "").strip().lower()

            if relation in valid_rels and from_type in valid_types and to_type in valid_types:
                query = f"""
                    MATCH (a:{from_type} {{name: $from_name}})
                    MATCH (b:{to_type} {{name: $to_name}})
                    MERGE (a)-[:{relation}]->(b)
                """
                session.run(query, from_name=from_name, to_name=to_name)
                print(f"  ({from_name}) -[{relation}]-> ({to_name})")

    print(f"\nGraph berhasil di-update dengan data baru!")

if __name__ == "__main__":
    print("DEMO: LLM Graph Builder\n")

    demo_texts = [
        """
        Dengue fever is a mosquito-borne tropical disease caused by the dengue virus.
        Symptoms include high fever, severe headache, pain behind the eyes, joint and muscle pain,
        fatigue, nausea, vomiting, and skin rash. Treatment involves acetaminophen for pain relief,
        oral rehydration salts, and IV fluids in severe cases. Patients should rest, drink plenty
        of fluids, avoid aspirin, and use mosquito repellent. A diet rich in papaya leaf extract,
        coconut water, and vitamin C foods is recommended. Light stretching and walking are
        suggested as recovery exercises.
        """,
        """
        Migraine is a neurological condition characterized by intense, recurring headaches often
        on one side of the head. Common symptoms include throbbing pain, sensitivity to light and
        sound, nausea, and visual disturbances called aura. Medications include triptans like
        sumatriptan, NSAIDs such as ibuprofen, and preventive drugs like topiramate. Precautions
        include avoiding triggers like stress and bright lights, maintaining regular sleep, and
        keeping a headache diary. Recommended foods include magnesium-rich foods, omega-3 fatty
        acids, and ginger tea. Yoga and aerobic exercise can help reduce frequency.
        """
    ]

    try:
        for text in demo_texts:
            extracted = extract_entities_from_text(text)
            populate_graph_from_extraction(extracted)
            print("\n" + "─" * 60)
    finally:
        driver.close()
