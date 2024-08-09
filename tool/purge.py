from neo4j import GraphDatabase

# Define correct URI and AUTH arguments (no AUTH by default)
URI = "bolt://localhost:7687"
AUTH = ("", "")

with GraphDatabase.driver(URI, auth=AUTH) as client:
    # Check the connection
    client.verify_connectivity()

    # Create a user in the database
    records, summary, keys = client.execute_query(
        "MATCH (n) DETACH DELETE n;",
        database_="memgraph",
    )
    print(summary.query)
