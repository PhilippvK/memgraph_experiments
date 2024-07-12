from neo4j import GraphDatabase
import networkx as nx
from networkx.drawing.nx_agraph import write_dot
import matplotlib.pyplot as plt

driver = GraphDatabase.driver('bolt://localhost:7687', auth=("", ""))

query = """
MATCH (n)-[r]->(c) RETURN *
"""

results = driver.session().run(query)

G = nx.MultiDiGraph()

nodes = list(results.graph()._nodes.values())
print("nodes", nodes)
for node in nodes:
    print("node", node)
    if len(node._labels) > 0:
        label = list(node._labels)[0]
    else:
        label = "?!"
    name = node._properties.get("name", "?")
    G.add_node(node.id, xlabel=label, label=name, properties=node._properties)

rels = list(results.graph()._relationships.values())
for rel in rels:
    label = rel.type
    # G.add_edge(rel.start_node.element_id, rel.end_node.element_id, key=rel.element_id, label=label, type=rel.type, properties=rel._properties)
    G.add_edge(rel.start_node.id, rel.end_node.id, key=rel.id, label=label, type=rel.type, properties=rel._properties)
print("G", G, dir(G))
write_dot(G, "out2.dot")

# print("G[]", G.nodes[2595], dir(G.nodes[2595]))
# print("G[]2", G.edges, dir(G.edges))

def filter_graph(G):
    view = nx.subgraph_view(G, filter_node=lambda node: G.nodes[node]["properties"].get("basic_block") == "%bb.7" and "%bb" not in G.nodes[node].get("label"))
    G_ = G.subgraph([node for node in view.nodes])
    # G__ = nx.subgraph_view(G_, filter_edge=lambda n1, n2: G_[n1][n2]["type"] == "DFG")
    return G_

G_ = filter_graph(G)
print("G_", G_)
write_dot(G_, "out3.dot")

def algo(G):
    print("algo")
    print("G", G, dir(G))
    if G.number_of_nodes() == 0:
        return
    mapping = dict(zip(G.nodes.keys(), range(len(G.nodes))))
    G = nx.relabel_nodes(G, mapping)
    topo = list(reversed(list(nx.topological_sort(G))))
    print("topo", topo)
    invalid = [False] * len(topo)
    fanout = [0] * len(topo)
    processed = [False] * len(topo)
    fanout_org = [0] * len(topo)
    for node in G.nodes:
        print("node", node)
        print("node_data", G.nodes[node])
        od = G.out_degree(node)
        is_output = G.out_degree(node) == 0
        if is_output:
            od += 1
        print("od", od)
        fanout_org[topo.index(node)] = od
        is_input = G.nodes[node].get("label") == "Const"  # TODO: add to db
        load_instrs = ["LB", "LBU", "LH", "LHU", "LW"]
        is_load = G.nodes[node].get("label") in load_instrs # TODO: add to db
        is_invalid = is_input or is_load
        if is_invalid:
            invalid[topo.index(node)] = True
    max_misos = []
    for node in topo:
        print("node", node)
        if processed[topo.index(node)]:
            continue
        fanout = fanout_org.copy()
        processed[topo.index(node)] = True
        if invalid[topo.index(node)]:
            continue
        max_miso = [False] * len(topo)
        max_miso[topo.index(node)] = True
        def generate_max_miso(node, count):
            ins = G.in_edges(node)
            print("ins", ins)
            for src, dest in ins:
                print("src", src)
                print("dest", dest)
                fanout[topo.index(src)] -= 1
                if not invalid[topo.index(src)] and fanout[topo.index(src)] == 0:
                    max_miso[topo.index(src)] = True
                    processed[topo.index(src)] = True
                    count = generate_max_miso(src, count + 1)
            return count
        size = generate_max_miso(node, 1)
        # size = generate_max_miso(node, G, max_miso, 1, invalid, fanout, processed)
        print("size", size)
        if size > 1:
            max_misos.append(max_miso)
    print("max_misos", max_misos, len(max_misos))
    from itertools import compress
    max_misos_ = [list(compress(topo, max_miso)) for max_miso in max_misos]
    print("max_misos_", max_misos_)
    # max_misos__ = [nx.subgraph_view(G, filter_node=lambda node: max_miso[topo.index(node)]) for max_miso in max_misos]
    # max_misos__ = [nx.subgraph_view(G, filter_node=lambda node: node in max_miso) for max_miso in max_misos_]
    # max_misos__ = [nx.subgraph_view(G, filter_node=lambda node: node in max_miso) for max_miso in max_misos_]
    max_misos__ = [G.subgraph(max_miso) for max_miso in max_misos_]
    print("max_misos__", max_misos__)
    for i, mig in enumerate(max_misos__):
        print("i,mig", i, mig)
        write_dot(mig, f"maxmiso{i}.dot")
        labeldict = {node: mig.nodes[node]["label"] for node in mig.nodes}
        print("labeldict", labeldict)
        nx.draw(mig, labels=labeldict, with_labels=True)
        plt.savefig(f"maxmiso{i}.png")
        plt.close()
    labeldict = {node: G.nodes[node]["label"] for node in G.nodes}
    print("labeldict", labeldict)
    nx.draw(G, labels=labeldict, with_labels=True)
    plt.savefig(f"full.png")
    plt.close()

    print("invalid", invalid)
    print("fanout", fanout)
    print("processed", processed)
    print("fanout_org", fanout_org)

algo(G_)
