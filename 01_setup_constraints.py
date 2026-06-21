from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

print("Connecting ke Neo4j")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

try:
    driver.verify_connectivity()
    print(" Koneksi ke Neo4j berhasil!\n")
except Exception as e:
    print(f" Koneksi gagal: {e}")
    exit(1)

constraints = [
    ("Disease",     "CREATE CONSTRAINT disease_name IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE"),
    ("Symptom",     "CREATE CONSTRAINT symptom_name IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE"),
    ("Medication",  "CREATE CONSTRAINT med_name IF NOT EXISTS FOR (m:Medication) REQUIRE m.name IS UNIQUE"),
    ("Precaution",  "CREATE CONSTRAINT prec_name IF NOT EXISTS FOR (p:Precaution) REQUIRE p.name IS UNIQUE"),
    ("Workout",     "CREATE CONSTRAINT workout_name IF NOT EXISTS FOR (w:Workout) REQUIRE w.name IS UNIQUE"),
    ("Diet",        "CREATE CONSTRAINT diet_name IF NOT EXISTS FOR (dt:Diet) REQUIRE dt.name IS UNIQUE"),
]

print("Membuat constraints")
with driver.session() as session:
    for label, query in constraints:
        try:
            session.run(query)
            print(f"  Selesai membuat constraint node :{label}")
        except Exception as e:
            print(f"  Gagal  :{label} — {e}")

driver.close()
print("\n Setup done.")
