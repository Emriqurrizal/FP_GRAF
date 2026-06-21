from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def run_query(tx, query):
    result = tx.run(query)
    return [record.data() for record in result]

def setup_gds_graph():
    """Projects the current Neo4j graph into an in-memory graph for GDS algorithms."""
    print("Setting up GDS in-memory graph 'medigraph'...")
    
    # Drop if exists
    try:
        with driver.session() as session:
            session.execute_write(lambda tx: tx.run("CALL gds.graph.drop('medigraph', false)"))
    except Exception:
        pass
    
    projection_query = """
    CALL gds.graph.project(
        'medigraph',
        ['Disease', 'Symptom', 'Medication'],
        {
            HAS_SYMPTOM: {orientation: 'UNDIRECTED'},
            TREATED_WITH: {orientation: 'UNDIRECTED'}
        }
    )
    YIELD graphName, nodeProjection, nodeCount, relationshipCount
    """
    with driver.session() as session:
        result = session.execute_write(run_query, projection_query)
        print(f"Graph projected: {result[0]}")

def run_pagerank():
    """Find the most significant symptoms and diseases using PageRank."""
    print("\n--- Running PageRank ---")
    query = """
    CALL gds.pageRank.stream('medigraph')
    YIELD nodeId, score
    RETURN gds.util.asNode(nodeId).name AS name, labels(gds.util.asNode(nodeId))[0] AS label, score
    ORDER BY score DESC, name ASC
    LIMIT 10
    """
    with driver.session() as session:
        results = session.execute_read(run_query, query)
        for r in results:
            print(f"{r['label']} - {r['name']}: {r['score']:.4f}")

def run_community_detection():
    """Cluster diseases and symptoms using the Louvain method."""
    print("\n--- Running Community Detection (Louvain) ---")
    query = """
    CALL gds.louvain.stream('medigraph')
    YIELD nodeId, communityId
    WITH gds.util.asNode(nodeId) AS n, communityId, labels(gds.util.asNode(nodeId))[0] AS label
    WHERE label = 'Disease'
    RETURN communityId, collect(n.name)[0..5] AS diseases, count(n) AS size
    ORDER BY size DESC
    LIMIT 5
    """
    with driver.session() as session:
        results = session.execute_read(run_query, query)
        for r in results:
            print(f"Community {r['communityId']} (Size: {r['size']}): {', '.join(r['diseases'])}")

def run_fastrp():
    """Generate FastRP node embeddings for similarity searches."""
    print("\n--- Running FastRP Embeddings ---")
    
    # We mutate the in-memory graph to store embeddings
    mutate_query = """
    CALL gds.fastRP.mutate('medigraph', {
        embeddingDimension: 64,
        randomSeed: 42,
        mutateProperty: 'embedding'
    })
    YIELD nodePropertiesWritten
    """
    
    # Then write to the database
    write_query = """
    CALL gds.graph.nodeProperties.write('medigraph', ['embedding'])
    YIELD propertiesWritten
    """
    with driver.session() as session:
        mut_res = session.execute_write(run_query, mutate_query)
        print(f"Computed FastRP Embeddings (Properties written in memory: {mut_res[0]['nodePropertiesWritten']})")
        
        write_res = session.execute_write(run_query, write_query)
        print(f"Saved FastRP Embeddings to database (Properties written: {write_res[0]['propertiesWritten']})")

if __name__ == "__main__":
    try:
        setup_gds_graph()
        run_pagerank()
        run_community_detection()
        run_fastrp()
    except Exception as e:
        print(f"Error running GDS algorithms: {e}")
        print("Make sure you have the Neo4j Graph Data Science (GDS) plugin installed and enabled in your Neo4j instance.")
    finally:
        driver.close()
