from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def run_query(tx, query):
    result = tx.run(query)
    return [record.data() for record in result]

def run_fastrp_embeddings():
    """Generate FastRP node embeddings based on graph structure."""
    print(f"\n{'='*60}")
    print(" FastRP Node Embeddings")
    print(f"{'='*60}\n")
    
    # Compute FastRP embeddings in memory
    fastrp_mutate = """
    CALL gds.fastRP.mutate('medigraph', {
        embeddingDimension: 64,
        randomSeed: 42,
        mutateProperty: 'embedding'
    })
    YIELD nodePropertiesWritten
    """
    with driver.session() as session:
        try:
            result = session.execute_write(run_query, fastrp_mutate)
            print(f" FastRP computed: {result[0]['nodePropertiesWritten']} node embeddings generated")
            
            # Write embeddings to database permanently
            fastrp_write = """
            CALL gds.graph.nodeProperties.write('medigraph', ['embedding'])
            YIELD propertiesWritten
            """
            result2 = session.execute_write(run_query, fastrp_write)
            print(f" Embeddings saved to database: {result2[0]['propertiesWritten']} properties written")
            
        except Exception as e:
            if "already exists" in str(e):
                print(" FastRP Embeddings already exist in the in-memory graph! Skipping re-computation...")
            else:
                print(f" Error computing FastRP: {e}")
                print(" (Have you run `04_gds_analytics.py` first to project 'medigraph'?)")
                return

def run_knn_similarity():
    """Find most similar diseases using KNN based on FastRP embeddings."""
    print(f"\n{'='*60}")
    print(" KNN — Top Similar Disease Pairs")
    print(f"{'='*60}\n")

    knn_query = """
    CALL gds.knn.stream('medigraph', {
        topK: 3,
        nodeProperties: ['embedding'],
        nodeLabels: ['Disease'],
        randomSeed: 42,
        concurrency: 1
    })
    YIELD node1, node2, similarity
    WITH gds.util.asNode(node1) AS n1, gds.util.asNode(node2) AS n2, similarity
    WHERE 'Disease' IN labels(n1) AND 'Disease' IN labels(n2)
    RETURN n1.name AS disease_1, n2.name AS disease_2,
           round(similarity * 10000) / 10000 AS similarity_score
    ORDER BY similarity_score DESC
    LIMIT 15
    """
    with driver.session() as session:
        try:
            results = session.execute_read(run_query, knn_query)
            print(f" {'Rank':<6} {'Disease 1':<25} {'Disease 2':<25} {'Similarity':<10}")
            print(f" {'-'*66}")
            for i, r in enumerate(results, 1):
                score = r['similarity_score']
                bar = '#' * int(score * 20)
                print(f" {i:<6} {r['disease_1']:<25} {r['disease_2']:<25} {score:.4f} {bar}")
        except Exception as e:
            print(f" Error running KNN: {e}")

def run_shared_symptoms_similarity():
    """Calculate similarity purely based on shared symptoms using Cypher."""
    print(f"\n{'='*60}")
    print(" Node Similarity — Diseases with Most Shared Symptoms")
    print(f"{'='*60}\n")
    
    similarity_query = """
    MATCH (d1:Disease)-[:HAS_SYMPTOM]->(s:Symptom)<-[:HAS_SYMPTOM]-(d2:Disease)
    WHERE elementId(d1) < elementId(d2)
    WITH d1.name AS disease_1, d2.name AS disease_2,
         count(s) AS shared_symptoms,
         collect(s.name)[0..5] AS sample_symptoms
    ORDER BY shared_symptoms DESC
    LIMIT 10
    RETURN disease_1, disease_2, shared_symptoms, sample_symptoms
    """
    with driver.session() as session:
        results = session.execute_read(run_query, similarity_query)
        for i, r in enumerate(results, 1):
            symptoms_str = ", ".join(r["sample_symptoms"])
            print(f"  [{i}] {r['disease_1']} <-> {r['disease_2']}")
            print(f"      Shared: {r['shared_symptoms']} symptoms (e.g., {symptoms_str})\n")
            
    print("Insight: Diseases with high similarity and many shared symptoms")
    print("         can be misdiagnosed for one another — important for differential diagnosis.")

if __name__ == "__main__":
    try:
        run_fastrp_embeddings()
        run_knn_similarity()
        run_shared_symptoms_similarity()
    finally:
        driver.close()
