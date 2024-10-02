import logging
import pickle
from pathlib import Path
from typing import Union
from anytree import AnyNode

from ..llvm_utils import parse_llvm_const_str, llvm_type_to_cdsl_type


logger = logging.getLogger("tree_utils")


class BaseNode(AnyNode):

    id_ = 0

    def __init__(self, children=None, node_id: int = -1, **kwargs):
        super().__init__(parent=None, children=children, id=BaseNode.id_, **kwargs)
        self.node_id = node_id
        BaseNode.id_ += 1

    @property
    def summary(self):
        return str(self)


class Operation(BaseNode):

    def __init__(self, name: str, children=None, node_id: int = -1, **kwargs):
        super().__init__(children=children, node_id=node_id, **kwargs)
        self.name = name


class Assignment(BaseNode):

    def __init__(self, children=None, node_id: int = -1, **kwargs):
        super().__init__(children=children, node_id=node_id, **kwargs)


class Declaration(BaseNode):

    def __init__(self, decl_type: str, children=None, node_id: int = -1, **kwargs):
        super().__init__(children=children, node_id=node_id, **kwargs)
        self.decl_type = decl_type


class Cast(BaseNode):

    def __init__(self, to: str, children=None, node_id: int = -1, **kwargs):
        super().__init__(children=children, node_id=node_id, **kwargs)
        self.to = to


class Ref(BaseNode):

    def __init__(self, name: str, node_id: int = -1, **kwargs):
        super().__init__(children=None, node_id=node_id, **kwargs)
        self.name = name


class Constant(BaseNode):

    def __init__(self, value: Union[int, float], node_id: int = -1, **kwargs):
        super().__init__(children=None, node_id=node_id, **kwargs)
        self.value = value


class Register(BaseNode):

    def __init__(self, name: str, reg_class: str, children=None, node_id: int = -1, **kwargs):
        super().__init__(children=children, node_id=node_id, **kwargs)
        self.name = name
        self.reg_class = reg_class


class Statements(BaseNode):

    def __init__(self, children=None, node_id: int = -1, **kwargs):
        super().__init__(children=children, node_id=node_id, **kwargs)


def tree_from_pkl(path: Union[str, Path]):
    with open(path, "rb") as f:
        tree = pickle.load(f)
    assert isinstance(tree, AnyNode)
    return tree


class TreeGenContext:

    def __init__(self, graph, sub, inputs=None, outputs=None, constants=None, explicit_types: bool = True) -> None:
        self.graph = graph
        self.sub = sub
        self.inputs = inputs if inputs is not None else []
        self.outputs = outputs if outputs is not None else []
        self.constants = constants if constants is not None else []
        self.explicit_types = explicit_types
        self.node_map = {}
        self.defs = {}
        self.temps = {}
        self.temp_idx = 0

    def get_temp(self):
        ret = f"temp{self.temp_idx}"
        self.temp_idx += 1
        return ret

    @property
    def visited(self):
        return set(self.node_map.keys())

    def visit(self, node):
        # print("visit", node)
        if node in self.visited:
            # print("not visited")
            op_type = self.graph.nodes[node]["properties"]["op_type"]
            # print("op_type", op_type)
            if op_type == "constant":
                val_str = self.graph.nodes[node]["properties"]["inst"]
                val, llvm_type, signed = parse_llvm_const_str(val_str)
                out_reg_size = self.graph.nodes[node]["properties"].get("out_reg_size", None)
                cdsl_type = llvm_type_to_cdsl_type(llvm_type, signed, reg_size=out_reg_size)
                ret = Constant(value=val, in_types=[], out_types=cdsl_type)
            else:
                # assert node in self.defs, f"node {node} not in defs {self.defs}"
                if node in self.defs:
                    ref = self.defs[node]
                    tree_node = self.node_map[node]
                    ret = Ref(name=ref, in_types=[], out_types=tree_node.out_types)
                    self.node_map[node] = ret
                else:
                    tree_node = self.node_map[node]
                    ref = self.get_temp()
                    self.defs[node] = ref
                    self.temps[ref] = tree_node
                    ret = Ref(name=ref, in_types=[], out_types=tree_node.out_types)
                    ret2 = Ref(name=ref, in_types=[], out_types=tree_node.out_types)
                    parent = self.node_map[node].parent
                    print("node", node)
                    print("tree_node", tree_node)
                    print("parent", parent)
                    print("parent.children", parent.children)
                    parent.children = [child if child.id != tree_node.id else ret2 for child in parent.children]
                    print("parent.children1", parent.children)

                    # input("444")
                    # = ret2
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
        srcs = [src for src, _ in srcs if src in self.inputs or src in self.constants or src in self.sub.nodes]
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
        # print("op_type", op_type)
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
            val_str = self.graph.nodes[node]["properties"]["inst"]
            val, llvm_type, signed = parse_llvm_const_str(val_str)
            cdsl_type = llvm_type_to_cdsl_type(llvm_type, signed, reg_size=out_reg_size)
            assert len(children) == 0
            ret = Constant(value=val, in_types=[], out_types=[cdsl_type])
        else:
            if node in self.inputs and node not in self.outputs:
                op_type = "input"
            signed = False  # ?
            cdsl_type = llvm_type_to_cdsl_type(out_reg_type, signed, reg_size=out_reg_size)
            # print("cdsl_type", cdsl_type)
            # print("children", children)
            # TODO: fix this for output lists
            in_types = sum([[x.out_types] if not isinstance(x.out_types, list) else x.out_types for x in children], [])
            ret = Operation(node_id=node, name=name, children=children, in_types=in_types, out_types=[cdsl_type])
        if self.explicit_types:
            type_str = f"unsigned<{out_reg_size}>"  # TODO
            ret = Cast(
                # name="?",
                children=[ret],
                to=type_str,
                in_types=ret.out_types,
                out_types=[type_str],
            )
        self.node_map[node] = ret
        return ret

    def visit_output(self, node):
        rets = []
        temp_idx_before = self.temp_idx
        ret = self.visit(node)
        temp_idx_after = self.temp_idx
        print("temp_idx_before", temp_idx_before)
        print("temp_idx_after", temp_idx_after)
        if temp_idx_after > temp_idx_before:
            refs = [f"temp{idx}" for idx in range(temp_idx_before, temp_idx_after)]
            print("temps")
            for ref in refs:
                print("ref", ref)
                print("self.temps[ref]", self.temps[ref])
                temp = self.temps[ref]
                ref_node = Ref(name=ref, in_types=[], out_types=[None])
                assert len(temp.out_types) == 1
                decl_type = temp.out_types[0]
                decl = Declaration(
                    children=[ref_node, temp],
                    decl_type=decl_type,
                    in_types=[ref_node.out_types[0], temp.out_types[0]],
                    out_types=[decl_type],
                )
                rets.append(decl)
        # # input("123")
        rets.append(ret)
        return rets
