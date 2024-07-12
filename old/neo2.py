from neo4j import GraphDatabase
import networkx as nx
from networkx.drawing.nx_agraph import write_dot

driver = GraphDatabase.driver('bolt://localhost:7687', auth=("", ""))

query = """
MATCH (n)-[r]->(c) RETURN *
"""

query2 = """
MATCH path1=(individual:Officer)-->(:Country), path2=(individual:Officer)-->(:Entity)-->(:Country)
WHERE individual.name =~ '.*Babis.*'
RETURN path1, path2;
"""

results = driver.session().run(query2)

G = nx.MultiDiGraph()

nodes = list(results.graph()._nodes.values())
for node in nodes:
    assert len(node._labels)
    label = list(node._labels)[0]
    name = node._properties.get("name", "?")
    G.add_node(node.id, xlabel=label, label=name, properties=node._properties)

rels = list(results.graph()._relationships.values())
for rel in rels:
    label = rel.type
    G.add_edge(rel.start_node.id, rel.end_node.id, key=rel.id, label=label, type=rel.type, properties=rel._properties)
print("G", G)
write_dot(G, "out2.dot")
