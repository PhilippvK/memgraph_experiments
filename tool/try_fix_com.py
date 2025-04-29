import pickle

from .hash import add_hash_attr
import networkx as nx
import networkx.algorithms.isomorphism as iso

IO_SUB_A_PKL = "out/embench/tarfind/20250410T011056/work/benchmark_body_%bb.5_0/io_sub213.pkl"
IO_SUB_B_PKL = "out/embench/tarfind/20250410T011056/work/benchmark_body_%bb.5_0/io_sub219.pkl"

with open(IO_SUB_A_PKL, "rb") as f:
    io_sub = pickle.load(f)

with open(IO_SUB_B_PKL, "rb") as f:
    io_sub_ = pickle.load(f)

print("io_sub", io_sub)
print("io_sub_", io_sub_)

# ignore_const = False
ignore_const = True

# handle_commutable = False
handle_commutable = True

add_hash_attr(io_sub, attr_name="hash_attr2", ignore_const=ignore_const, ignore_names=False, handle_commutable=handle_commutable)
add_hash_attr(io_sub_, attr_name="hash_attr2", ignore_const=ignore_const, ignore_names=False, handle_commutable=handle_commutable)

print("A", [io_sub.nodes[node]["hash_attr2"] for node in io_sub.nodes])
print("B", [io_sub_.nodes[node]["hash_attr2"] for node in io_sub_.nodes])

print("AA", [io_sub.edges[edge]["hash_attr2"] for edge in io_sub.edges])
print("BB", [io_sub_.edges[edge]["hash_attr2"] for edge in io_sub_.edges])

nm = iso.categorical_node_match("hash_attr2", None)
em = iso.categorical_edge_match("hash_attr2", None)

def categorical_edge_match_multidigraph(attr, default):
    if isinstance(attr, str):

        def match(data1, data2):
            assert len(data1) == 1
            data1 = list(data1.values())[0]
            assert len(data2) == 1
            data2 = list(data2.values())[0]
            return data1.get(attr, default) == data2.get(attr, default)

    else:
        attrs = list(zip(attr, default))  # Python 3

        def match(data1, data2):
            return all(data1.get(attr, d) == data2.get(attr, d) for attr, d in attrs)

    return match

def nm2(*args):
    # print("nm2", args)
    ret = iso.categorical_node_match("hash_attr2", None)(*args)
    # print("ret", ret)
    return ret

def em2(*args):
    # print("em2", args)
    # ret = iso.categorical_edge_match("hash_attr2", None)(*args)
    ret = categorical_edge_match_multidigraph("hash_attr2", None)(*args)
    # print("ret", ret)
    return ret

# matcher = nx.algorithms.isomorphism.DiGraphMatcher(io_sub, io_sub_, node_match=nm, edge_match=em)
matcher = nx.algorithms.isomorphism.DiGraphMatcher(io_sub, io_sub_, node_match=nm2, edge_match=em2)
check = matcher.is_isomorphic()
print("check", check)
