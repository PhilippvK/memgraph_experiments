from neo4j import GraphDatabase
import networkx as nx
from networkx.drawing.nx_agraph import write_dot
import matplotlib.pyplot as plt

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("", ""))

query_func = """
MATCH p0=(n00)-[r01:DFG]->(n01)
WHERE n00.func_name = 'tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_per_axis_add_clip_cast_subtract_1_compute_'
RETURN p0;
"""
query = """
MATCH (n00)
WHERE n00.func_name = 'tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_per_axis_add_clip_cast_subtract_1_compute_'
CALL path.subgraph_nodes(n00, {
      relationshipFilter: [],
      labelFilter: [],
      minLevel: 1,
      maxLevel: 3
})
YIELD nodes
RETURN n00, nodes;
"""

func_results = driver.session().run(query_func)
results = driver.session().run(query)
# print("results", results, dir(results))
# print("results.df", results.to_df())
# nodes = list(func_results.graph()._nodes.values())
for i, result in enumerate(results):
    print("result", result, i)
    # input(">")
input(">>>")

GF = nx.MultiDiGraph()
nodes = list(func_results.graph()._nodes.values())
# print("nodes", nodes)
for node in nodes:
    print("node", node)
    if len(node._labels) > 0:
        label = list(node._labels)[0]
    else:
        label = "?!"
    name = node._properties.get("name", "?")
    GF.add_node(node.id, xlabel=label, label=name, properties=node._properties)

rels = list(func_results.graph()._relationships.values())
for rel in rels:
    label = rel.type
    # GF.add_edge(rel.start_node.element_id, rel.end_node.element_id, key=rel.element_id, label=label, type=rel.type, properties=rel._properties)
    GF.add_edge(rel.start_node.id, rel.end_node.id, key=rel.id, label=label, type=rel.type, properties=rel._properties)
print("GF", GF)
write_dot(GF, f"func.dot")
input("?")

G = nx.MultiDiGraph()
nodes = list(results.graph()._nodes.values())
# print("nodes", nodes)
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
print("G", G)
write_dot(G, f"results.dot")

subs = []
for i, result in enumerate(results):
    # print("result", result, dir(result))
    # print("result.data", result.data())
    # print("result.value", result.value(), dir())
    nodes_ = set()
    # path = result.value()
    for path in result:
        # print("path", path)
        # input("p")
        # print("path", path, dir(path))
        nodes__ = path.nodes
        # print("nodes__", nodes__[0].element_id)
        # 'count', 'data', 'get', 'index', 'items', 'keys', 'value', 'values'
        nodes_ |= {n.id for n in nodes__}
    # print("nodes_", nodes_)
    G_ = G.subgraph(nodes_)
    # G_ = nx.subgraph_view(G, filter_node=lambda x: x in nodes_)
    print("G_", G_)
    write_dot(G_, f"result{i}.dot")
    count = subs.count(G_)
    if count > 0:
        input("2")
    subs.append(G_)

print("GF", GF)
print("GF.nodes", GF.nodes)
mapping = dict(zip(GF.nodes.keys(), range(len(GF.nodes))))
GF = nx.relabel_nodes(GF, mapping)
G = nx.relabel_nodes(G, mapping)
for i in range(len(subs)):
    subs[i] = nx.relabel_nodes(subs[i], mapping)
print("GF", GF)
print("G", G)
print("GF.nodes", GF.nodes)
print("G.nodes", G.nodes)
# mapping = dict(zip(G.nodes.keys(), range(len(G.nodes))))
# G = nx.relabel_nodes(G, mapping)
# for i in range(len(subs)):
#     subs[i] = nx.relabel_nodes(subs[i], mapping)
# print("G", G)
# print("G.nodes", G.nodes)
# topo = list(reversed(list(nx.topological_sort(G))))
topo = list(reversed(list(nx.topological_sort(GF))))
print("topo", topo)
# mapping = dict(zip(G.nodes.keys(), topo))
mapping = dict(zip(GF.nodes.keys(), topo))
GF = nx.relabel_nodes(GF, mapping)
G = nx.relabel_nodes(G, mapping)
for i in range(len(subs)):
    subs[i] = nx.relabel_nodes(subs[i], mapping)
print("GF", GF)
print("G", G)
print("GF.nodes", GF.nodes)
print("G.nodes", G.nodes)

print("subs[0]", subs[0])
print("subs[0].nodes", subs[0].nodes)


def calc_inputs(G, sub):
    print("calc_inputs", sub)
    inputs = []
    ret = 0
    sub_nodes = sub.nodes
    print("sub_nodes", sub_nodes)
    for node in sub_nodes:
        print("node", node, G.nodes[node].get("label"))
        ins = G.in_edges(node)
        print("ins", ins)
        for in_ in ins:
            # print("in_", in_, G.nodes[in_[0]].get("label"))
            src = in_[0]
            print("src", src, G.nodes[src].get("label"))
            # print("src in sub_nodes", src in sub_nodes)
            # print("src not in inputs", src not in inputs)
            if not (src in sub_nodes) and (src not in inputs):
                print("IN")
                ret += 1
                inputs.append(src)
    print("ret", ret)
    return ret


def calc_outputs(G, sub):
    print("calc_outputs", sub)
    ret = 0
    sub_nodes = sub.nodes
    print("sub_nodes", sub_nodes)
    for node in sub_nodes:
        print("node", node, G.nodes[node].get("label"))
        if G.nodes[node]["properties"]["op_type"] == "output":
            # print("A")
            print("OUT2")
            ret += 1
        else:
            # print("B")
            outs = G.out_edges(node)
            print("outs", outs)
            for out_ in outs:
                # print("out_", out_, G.nodes[out_[0]].get("label"))
                dst = out_[1]
                print("dst", dst, G.nodes[dst].get("label"))
                if dst not in sub_nodes:
                    print("OUT")
                    ret += 1
    print("ret", ret)
    # input("1")
    return ret


# if True:
for i, sub in enumerate(subs):
    # i = 3
    sub = subs[i]
    print("i, sub", i, sub)
    num_inputs = calc_inputs(GF, sub)
    num_outputs = calc_outputs(GF, sub)
    print("num_inputs", num_inputs)
    print("num_outputs", num_outputs)
    if num_outputs == 0 or num_inputs == 0:
        input(">?")
    elif num_outputs in [1] and num_inputs in [1, 2, 3]:
        input(">")
