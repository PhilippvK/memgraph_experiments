import pickle

from .hash import add_hash_attr
from .iso import categorical_edge_match_multidigraph
import networkx as nx
import networkx.algorithms.isomorphism as iso

IO_SUB_A_PKL = "/work/git/isaac-demo/out/embench/md5sum/20250408T211018/work/md5_%bb.10_0/io_sub587.pkl"
IO_SUB_B_PKL = "/work/git/isaac-demo/out/embench/picojpeg/20250409T085121/work/pjpeg_decode_mcu_%bb.151_0/io_sub324.pkl"

with open(IO_SUB_A_PKL, "rb") as f:
    io_sub = pickle.load(f)

with open(IO_SUB_B_PKL, "rb") as f:
    io_sub_ = pickle.load(f)

print("io_sub", io_sub)
print("io_sub_", io_sub_)

ignore_const = False
# ignore_const = True

ignore_names = False
# ignore_names = True

# handle_commutable = False
handle_commutable = True

add_hash_attr(io_sub, attr_name="hash_attr2", ignore_const=ignore_const, ignore_names=ignore_names, handle_commutable=handle_commutable)
add_hash_attr(io_sub_, attr_name="hash_attr2", ignore_const=ignore_const, ignore_names=ignore_names, handle_commutable=handle_commutable)

print("A", [io_sub.nodes[node]["hash_attr2"] for node in io_sub.nodes])
print("B", [io_sub_.nodes[node]["hash_attr2"] for node in io_sub_.nodes])

print("AA", [io_sub.edges[edge]["hash_attr2"] for edge in io_sub.edges])
print("BB", [io_sub_.edges[edge]["hash_attr2"] for edge in io_sub_.edges])

nm = iso.categorical_node_match("hash_attr2", None)
em = iso.categorical_edge_match("hash_attr2", None)

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
