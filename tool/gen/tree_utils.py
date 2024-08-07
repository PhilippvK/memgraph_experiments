import logging
import pickle
from pathlib import Path
from typing import Union
from anytree import AnyNode


logger = logging.getLogger("tree_utils")


def tree_from_pkl(path: Union[str, Path]):
    with open(path, "rb") as f:
        tree = pickle.load(f)
    assert isinstance(tree, AnyNode)
    return tree


class TreeGenContext:

    def __init__(self, graph, sub, inputs=None, outputs=None, explicit_types: bool = True) -> None:
        self.graph = graph
        self.sub = sub
        self.inputs = inputs if inputs is not None else []
        self.outputs = outputs if outputs is not None else []
        self.explicit_types = explicit_types
        self.node_map = {}
        self.defs = {}

    @property
    def visited(self):
        return set(self.node_map.keys())

    def visit(self, node):
        print("visit", node)
        if node in self.visited:
            print("not visited")
            op_type = self.graph.nodes[node]["properties"]["op_type"]
            print("op_type", op_type)
            if op_type == "constant":
                val = self.graph.nodes[node]["properties"]["inst"]
                val = int(val[:-1])
                ret = AnyNode(id=-1, value=val, op_type=op_type, children=[])
            else:
                assert node in self.defs, f"node {node} not in defs {self.defs}"
                ref = self.defs[node]
                ret = AnyNode(id=-1, name=ref, op_type="ref")
            return ret
            # return self.node_map[node]
        # if node in inputs:
        #     children = []
        # else:
        # print("node", node)
        srcs = [
            (src, edge_data["properties"].get("op_idx", None))
            for src, _, edge_data in self.graph.in_edges(node, data=True)
        ]
        # print("srcs1", srcs)
        srcs = sorted(srcs, key=lambda x: x[1])
        # print("srcs2", srcs)
        srcs = [src for src, _ in srcs if src in self.inputs or src in self.sub.nodes]
        # print("srcs3", srcs)
        # for src, _, edge_data in self.graph.in_edges(node, data=True):
        #     print("src", src)
        #     print("edge_data", edge_data)
        # input("%%%")
        # print("?", [self.graph[src, node]["properties"]["op_idx"] for src in srcs])
        # srcs = sorted(srcs, key=lambda src: self.graph[src, node]["properties"]["op_idx"])
        # print("srcs2", srcs)
        children = [self.visit(src) for src in srcs]
        # print("children", children)
        op_type = self.graph.nodes[node]["properties"]["op_type"]
        print("op_type", op_type)
        name = self.graph.nodes[node]["properties"]["name"]
        # print("name", name)
        # out_reg_class = self.graph.nodes[node]["properties"].get("out_reg_class", None)
        # print("out_reg_class", out_reg_class)
        out_reg_type = self.graph.nodes[node]["properties"].get("out_reg_type", None)
        # print("out_reg_type", out_reg_type)
        out_reg_size = self.graph.nodes[node]["properties"].get("out_reg_size", None)
        # print("out_reg_size", out_reg_size)
        out_reg_name = self.graph.nodes[node]["properties"].get("out_reg_name", None)
        # print("out_reg_name", out_reg_name)
        # print("!1", [self.graph[src, node]["properties"].get("op_reg_class", None) for src in srcs])
        # print("!2", [self.graph[src, node]["properties"].get("op_reg_type", None) for src in srcs])
        # print("!3", [self.graph[src, node]["properties"].get("op_reg_size", None) for src in srcs])
        # print("!4", [self.graph[src, node]["properties"].get("op_reg_name", None) for src in srcs])
        # input("!!")
        if op_type == "constant":
            val = self.graph.nodes[node]["properties"]["inst"]
            val = int(val[:-1])
            ret = AnyNode(id=-1, value=val, op_type=op_type, children=children)
        else:
            if node in self.inputs and node not in self.outputs:
                op_type = "input"
            ret = AnyNode(id=node, name=name, op_type=op_type, children=children)
        if self.explicit_types:
            type_str = f"unsigned<{out_reg_size}>"
            ret = AnyNode(id=-1, name="?", op_type="cast", children=[ret], to=type_str)
        self.node_map[node] = ret
        return ret
