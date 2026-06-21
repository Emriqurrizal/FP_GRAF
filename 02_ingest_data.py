import pandas as pd
import ast
import os
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, DATA_DIR

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
driver.verify_connectivity()
print("Terhubung ke Neo4j\n")

print("Loading Diseases & Symptoms")

df_main = pd.read_csv(os.path.join(DATA_DIR, "Diseases_and_Symptoms_dataset.csv"))
symptom_cols = [c for c in df_main.columns if c != "diseases"]

disease_symptoms = {}
for disease, group in df_main.groupby("diseases"):
    symptom_freq = group[symptom_cols].mean()
    active_symptoms = symptom_freq[symptom_freq > 0.1].index.tolist()
    disease_symptoms[disease] = active_symptoms

print(f"  → {len(disease_symptoms)} penyakit unik")
print(f"  → {len(symptom_cols)} gejala tersedia")

def ingest_diseases_and_symptoms(tx, disease_name, symptoms):
    tx.run("MERGE (d:Disease {name: $name})", name=disease_name)
    for symptom in symptoms:
        tx.run("""
            MERGE (s:Symptom {name: $symptom})
            WITH s
            MATCH (d:Disease {name: $disease})
            MERGE (d)-[:HAS_SYMPTOM]->(s)
        """, symptom=symptom, disease=disease_name)

with driver.session() as session:
    for i, (disease, symptoms) in enumerate(disease_symptoms.items(), 1):
        session.execute_write(ingest_diseases_and_symptoms, disease, symptoms)
        print(f"  [{i:>3}/{len(disease_symptoms)}] {disease} ({len(symptoms)} gejala)")

print(f"\nStep 1 done.\n")

print("STEP 2: Loading Disease Descriptions")

df_desc = pd.read_csv(os.path.join(DATA_DIR, "description.csv"))

def ingest_description(tx, disease, description):
    tx.run("""
        MERGE (d:Disease {name: $name})
        SET d.description = $desc
    """, name=disease, desc=description)

with driver.session() as session:
    matched = 0
    for _, row in df_desc.iterrows():
        disease_name = str(row["Disease"]).strip().lower()
        session.execute_write(ingest_description, disease_name, str(row["Description"]))
        matched += 1
    print(f"  ✅ {matched} deskripsi di-update")

print("STEP 3: Loading Medications")

df_meds = pd.read_csv(os.path.join(DATA_DIR, "medications.csv"))

def parse_list_string(s):
    """Parse string list seperti "['item1', 'item2']" """
    try:
        return ast.literal_eval(s)
    except:
        return [x.strip().strip("'\"") for x in str(s).strip("[]").split(",")]

def ingest_medications(tx, disease, medications):
    for med in medications:
        med = med.strip()
        if not med:
            continue
        tx.run("""
            MERGE (m:Medication {name: $med})
            WITH m
            MERGE (d:Disease {name: $disease})
            MERGE (d)-[:TREATED_WITH]->(m)
        """, med=med, disease=disease)

with driver.session() as session:
    for _, row in df_meds.iterrows():
        disease_name = str(row["Disease"]).strip().lower()
        meds = parse_list_string(row["Medication"])
        session.execute_write(ingest_medications, disease_name, meds)
    print(f" {len(df_meds)} penyakit + obat di-ingest")

print("STEP 4: Loading Precautions")

df_prec = pd.read_csv(os.path.join(DATA_DIR, "precautions.csv"))

def ingest_precautions(tx, disease, precautions):
    for prec in precautions:
        prec = prec.strip()
        if not prec:
            continue
        tx.run("""
            MERGE (p:Precaution {name: $prec})
            WITH p
            MERGE (d:Disease {name: $disease})
            MERGE (d)-[:HAS_PRECAUTION]->(p)
        """, prec=prec, disease=disease)

with driver.session() as session:
    for _, row in df_prec.iterrows():
        disease_name = str(row["Disease"]).strip().lower()
        precs = [str(row[f"Precaution_{i}"]).strip()
                 for i in range(1, 5)
                 if pd.notna(row.get(f"Precaution_{i}"))]
        session.execute_write(ingest_precautions, disease_name, precs)
    print(f"  {len(df_prec)} penyakit + precautions di-ingest")

print("\n" + "=" * 55)
print("🏋️ STEP 5: Loading Workouts...")
print("=" * 55)

df_workout = pd.read_csv(os.path.join(DATA_DIR, "workout.csv"))

def ingest_workouts(tx, disease, workouts):
    for w in workouts:
        w_name = w.split(":")[0].strip() if ":" in w else w.strip()
        if not w_name:
            continue
        tx.run("""
            MERGE (w:Workout {name: $workout})
            WITH w
            MERGE (d:Disease {name: $disease})
            MERGE (d)-[:RECOMMENDED_WORKOUT]->(w)
        """, workout=w_name, disease=disease)

with driver.session() as session:
    for _, row in df_workout.iterrows():
        disease_name = str(row["Disease"]).strip().lower()
        workouts = parse_list_string(row["Workouts"])
        session.execute_write(ingest_workouts, disease_name, workouts)
    print(f"  {len(df_workout)} penyakit + workouts di-ingest")

print("STEP 6: Loading Diets")

df_diet = pd.read_csv(os.path.join(DATA_DIR, "diets.csv"))

def ingest_diets(tx, disease, diets):
    for d_item in diets:
        d_item = d_item.strip()
        if not d_item:
            continue
        tx.run("""
            MERGE (dt:Diet {name: $diet})
            WITH dt
            MERGE (d:Disease {name: $disease})
            MERGE (d)-[:RECOMMENDED_DIET]->(dt)
        """, diet=d_item, disease=disease)

with driver.session() as session:
    for _, row in df_diet.iterrows():
        disease_name = str(row["Disease"]).strip().lower()
        diets = parse_list_string(row["Diet"])
        session.execute_write(ingest_diets, disease_name, diets)
    print(f"  {len(df_diet)} penyakit + diets di-ingest")

print("Counting nodes & relationships")

with driver.session() as session:
    result = session.run("""
        MATCH (n)
        RETURN labels(n)[0] AS NodeType, COUNT(n) AS Total
        ORDER BY Total DESC
    """)
    for r in result:
        print(f"  {r['NodeType']:>15}: {r['Total']} nodes")

    result2 = session.run("MATCH ()-[r]->() RETURN TYPE(r) AS RelType, COUNT(r) AS Total ORDER BY Total DESC")
    print()
    for r in result2:
        print(f"  {r['RelType']:>25}: {r['Total']} relasi")

driver.close()
print(" Done, run di http://localhost:7474")
