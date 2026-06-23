from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def run_query(tx, query, **kwargs):
    result = tx.run(query, **kwargs)
    return [record.data() for record in result]

def setup_gds_graph():
    """Projects the current Neo4j graph into an in-memory graph for GDS algorithms."""
    print(" Setting up GDS in-memory graph 'medigraph'...\n")
    
    # Drop if exists
    try:
        with driver.session() as session:
            session.execute_write(lambda tx: tx.run("CALL gds.graph.drop('medigraph', false)"))
            print("  Dropped existing 'medigraph' projection")
    except Exception:
        pass
    
    projection_query = """
    CALL gds.graph.project(
        'medigraph',
        ['Disease', 'Symptom', 'Medication', 'Precaution', 'Workout', 'Diet'],
        {
            HAS_SYMPTOM: {orientation: 'UNDIRECTED'},
            TREATED_WITH: {orientation: 'UNDIRECTED'},
            HAS_PRECAUTION: {orientation: 'UNDIRECTED'},
            RECOMMENDED_WORKOUT: {orientation: 'UNDIRECTED'},
            RECOMMENDED_DIET: {orientation: 'UNDIRECTED'}
        }
    )
    YIELD graphName, nodeCount, relationshipCount
    """
    with driver.session() as session:
        result = session.execute_write(run_query, projection_query)
        info = result[0]
        print(f"  Graph '{info['graphName']}' projected:")
        print(f"   -> {info['nodeCount']} nodes")
        print(f"   -> {info['relationshipCount']} relationships")

def run_pagerank():
    """Find the most significant nodes using PageRank."""
    print(f"\n{'='*60}")
    print(" PageRank — Top 15 Most Important Nodes")
    print(f"{'='*60}\n")
    
    query = """
    CALL gds.pageRank.stream('medigraph')
    YIELD nodeId, score
    RETURN gds.util.asNode(nodeId).name AS name,
           labels(gds.util.asNode(nodeId))[0] AS label,
           score
    ORDER BY score DESC
    LIMIT 15
    """
    with driver.session() as session:
        results = session.execute_read(run_query, query)
        print(f" {'Rank':<6} {'Type':<12} {'Name':<35} {'Score':<10}")
        print(f" {'-'*63}")
        for i, r in enumerate(results, 1):
            print(f" {i:<6} {r['label']:<12} {r['name']:<35} {r['score']:.6f}")
            
    print("\nInsight: Nodes with the highest PageRank are the most 'central'")
    print("         in the medical network — mostly connected to other entities.")

def run_community_detection():
    """Cluster nodes using the Louvain method."""
    print(f"\n{'='*60}")
    print(" Community Detection (Louvain) — Top 10 Communities")
    print(f"{'='*60}\n")
    
    query = """
    CALL gds.louvain.stream('medigraph')
    YIELD nodeId, communityId
    WITH gds.util.asNode(nodeId) AS n, communityId, labels(gds.util.asNode(nodeId))[0] AS label
    WHERE label = 'Disease'
    RETURN communityId,
           collect(n.name)[0..5] AS sample_diseases,
           count(n) AS size
    ORDER BY size DESC
    LIMIT 10
    """
    with driver.session() as session:
        results = session.execute_read(run_query, query)
        for r in results:
            diseases_str = ", ".join(r["sample_diseases"])
            print(f" Community {r['communityId']} (Size: {r['size']} diseases)")
            print(f"   Examples: {diseases_str}\n")

    print("Insight: Diseases in the same community tend to share")
    print("         similar symptoms, medications, or treatments.")

def run_shortest_path_and_degree():
    """Find shortest path between 2 diseases and calculate degree centrality."""
    print(f"\n{'='*60}")
    print(" Shortest Path Between Diseases")
    print(f"{'='*60}\n")
    
    # Grab 2 random diseases
    with driver.session() as session:
        sample = session.run("""
            MATCH (d:Disease)
            WITH d, rand() AS r
            ORDER BY r
            LIMIT 2
            RETURN collect(d.name) AS diseases
        """)
        sample_diseases = sample.single()["diseases"]
        
    if len(sample_diseases) < 2:
        print(" Not enough diseases to find a path.")
        return
        
    disease_a = sample_diseases[0]
    disease_b = sample_diseases[1]
    
    print(f" Finding shortest path between:")
    print(f"  Disease A: {disease_a}")
    print(f"  Disease B: {disease_b}\n")
    
    shortest_path_query = """
    MATCH (a:Disease {name: $disease_a}), (b:Disease {name: $disease_b})
    MATCH path = shortestPath((a)-[*]-(b))
    RETURN [node IN nodes(path) | coalesce(node.name, 'unknown')] AS path_nodes,
           [rel IN relationships(path) | type(rel)] AS path_rels,
           length(path) AS path_length
    """
    with driver.session() as session:
        result = session.run(shortest_path_query, disease_a=disease_a, disease_b=disease_b)
        record = result.single()
        
        if record:
            print(f"  Path Length: {record['path_length']} hops\n")
            print(f"  Path:")
            nodes = record["path_nodes"]
            rels = record["path_rels"]
            path_str = nodes[0]
            for i, rel in enumerate(rels):
                path_str += f" --[{rel}]--> {nodes[i+1]}"
            print(f"    {path_str}")
        else:
            print("  No path found between the two diseases")

    # Degree Centrality
    print(f"\n{'='*60}")
    print(" Degree Centrality — Top 10 Most Connected Diseases")
    print(f"{'='*60}\n")
    
    degree_query = """
    MATCH (d:Disease)
    OPTIONAL MATCH (d)-[r]-()
    WITH d.name AS disease, count(r) AS degree
    ORDER BY degree DESC
    LIMIT 10
    RETURN disease, degree
    """
    with driver.session() as session:
        results = session.execute_read(run_query, degree_query)
        print(f" {'Rank':<6} {'Disease':<35} {'Connections':<12}")
        print(f" {'-'*53}")
        for i, r in enumerate(results, 1):
            bar = '#' * min(int(r['degree'] / 2), 30)
            print(f" {i:<6} {r['disease']:<35} {r['degree']:<5} {bar}")

if __name__ == "__main__":
    try:
        setup_gds_graph()
        run_pagerank()
        run_community_detection()
        run_shortest_path_and_degree()
    except Exception as e:
        print(f"Error running GDS algorithms: {e}")
        print("Make sure you have the Neo4j Graph Data Science (GDS) plugin installed and enabled.")
    finally:
        driver.close()
