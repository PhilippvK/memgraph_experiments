import logging
import pickle
from pathlib import Path
from typing import Union, Optional
from anytree import AnyNode, RenderTree

from ..llvm_utils import parse_llvm_const_str, llvm_type_to_cdsl_type
from .utils import mem_lookup


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


class Statement(BaseNode):

    def __init__(self, children=None, node_id: int = -1, **kwargs):
        super().__init__(children=children, node_id=node_id, **kwargs)


class Assignment(Statement):

    def __init__(self, children=None, node_id: int = -1, **kwargs):
        super().__init__(children=children, node_id=node_id, **kwargs)


class Declaration(Statement):

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

    def __init__(
        self,
        graph,
        sub,
        sub_data,
        node_aliases=None,
        inputs=None,
        outputs=None,
        constants=None,
        explicit_types: bool = True,
        xlen: Optional[int] = None,
    ) -> None:
        self.graph = graph
        self.sub = sub
        self.sub_data = sub_data
        self.node_aliases = node_aliases
        self.inputs = inputs if inputs is not None else []
        self.outputs = outputs if outputs is not None else []
        self.constants = constants if constants is not None else []
        self.explicit_types = explicit_types
        self.node_map = {}
        self.defs = {}
        self.temps = {}
        self.temp_idx = 0
        self.xlen = xlen

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
            # print("visited")
            op_type = self.graph.nodes[node]["properties"]["op_type"]
            # print("op_type", op_type)
            if op_type == "constant":
                val_str = self.graph.nodes[node]["properties"]["inst"]
                val, llvm_type, signed = parse_llvm_const_str(val_str)
                out_reg_size = self.graph.nodes[node]["properties"].get("out_reg_size", None)
                cdsl_type = llvm_type_to_cdsl_type(llvm_type, signed, reg_size=out_reg_size, allow_unknown=True)
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
                    parent.children = [child if child.id != tree_node.id else ret2 for child in parent.children]
            # print("ret", ret)
            return ret
        # print("not visited")
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
        print("children", children)
        op_type = self.graph.nodes[node]["properties"]["op_type"]
        print("op_type", op_type)
        name = self.graph.nodes[node]["properties"]["name"]
        print("name", name)
        out_reg_class = self.graph.nodes[node]["properties"].get("out_reg_class", None)
        print("out_reg_class", out_reg_class)
        out_reg_type = self.graph.nodes[node]["properties"].get("out_reg_type", None)
        print("out_reg_type", out_reg_type)
        out_reg_size = self.graph.nodes[node]["properties"].get("out_reg_size", None)
        print("out_reg_size", out_reg_size)
        out_reg_name = self.graph.nodes[node]["properties"].get("out_reg_name", None)
        print("out_reg_name", out_reg_name)
        # print("!1", [self.graph[src, node]["properties"].get("op_reg_class", None) for src in srcs])
        # print("!2", [self.graph[src, node]["properties"].get("op_reg_type", None) for src in srcs])
        # print("!3", [self.graph[src, node]["properties"].get("op_reg_size", None) for src in srcs])
        # print("!4", [self.graph[src, node]["properties"].get("op_reg_name", None) for src in srcs])
        # input("!!")
        is_phys_reg = out_reg_name.startswith("$x") or out_reg_name.startswith("$f")
        if is_phys_reg and (out_reg_size is None or out_reg_size == "unknown"):
            if self.xlen is not None:
                out_reg_size = self.xlen
            else:
                out_reg_size = "XLEN"
        if op_type == "constant":
            val_str = self.graph.nodes[node]["properties"]["inst"]
            val, llvm_type, signed = parse_llvm_const_str(val_str)
            cdsl_type = llvm_type_to_cdsl_type(llvm_type, signed, reg_size=out_reg_size, allow_unknown=True)
            assert len(children) == 0
            ret = Constant(value=val, in_types=[], out_types=[cdsl_type])
        else:
            print("op_type", op_type)
            if op_type == "label" and out_reg_size == "unknown":
                # TODO: handle labels properly!
                print("if")
                print("self.xlen", self.xlen)
                # if self.xlen is not None:
                #     out_reg_size = self.xlen
                if self.xlen is not None:
                    out_reg_size = self.xlen
                else:
                    out_reg_size = "XLEN"
            if node in self.inputs and node not in self.outputs:
                op_type = "input"
            print("op_type", op_type)
            signed = False  # ?
            print("node", node)
            # TODO: flag $x0 as register, not input!
            cdsl_type = llvm_type_to_cdsl_type(out_reg_type, signed, reg_size=out_reg_size)
            # print("cdsl_type", cdsl_type)
            # print("children", children)
            # TODO: fix this for output lists
            in_types = sum([[x.out_types] if not isinstance(x.out_types, list) else x.out_types for x in children], [])
            ret = Operation(node_id=node, name=name, children=children, in_types=in_types, out_types=[cdsl_type])
        if self.explicit_types and op_type != "constant":
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

    def resolve_op_idx(self, node):
        operand_nodes = self.sub_data["OperandNodes"]
        if node in operand_nodes:
            # print("Found!")
            op_idx = operand_nodes.index(node)
        else:
            # print("Not Found!")
            assert node in self.node_aliases
            node = self.node_aliases[node]
            op_idx = operand_nodes.index(node)
        return op_idx

    def visit_input(self, inp):
        # print("visit_input", inp)
        operand_names = self.sub_data["OperandNames"]
        # print("operand_names", operand_names)
        operand_nodes = self.sub_data["OperandNodes"]
        # print("operand_nodes", operand_nodes)
        operand_dirs = self.sub_data["OperandDirs"]
        # print("operand_dirs", operand_dirs)
        operand_types = self.sub_data["OperandTypes"]  # TODO: rename to Classes?
        # operand_enc_bits = self.sub_data["OperandEncBits"]
        operand_reg_classes = self.sub_data["OperandRegClasses"]
        if inp in self.node_aliases:
            inp = self.node_aliases[inp]
            assert inp in operand_nodes
            # print("return None")
            # return None
        # if inp in self.visited:
        #     return None
        op_idx = operand_nodes.index(inp)
        # print("op_idx", op_idx)
        # print("i", i)
        # print("inp", inp)
        operand_name = operand_names[op_idx]
        operand_dir = operand_dirs[op_idx]
        operand_type = operand_types[op_idx]
        if operand_dir == "INOUT":
            # print("INOUT")

            assert len(self.node_aliases) > 0
            assert inp in self.node_aliases.values()
            alias = list(self.node_aliases.keys())[list(self.node_aliases.values()).index(inp)]
            inp = alias

        # print("inp", inp)
        node_data = self.graph.nodes[inp]
        # print("node_data", node_data)
        # print("node_data", node_data)
        node_properties = node_data["properties"]
        # print("node_properties", node_properties)
        op_type = node_properties["op_type"]
        print("op_type", op_type)

        if op_type == "constant":
            return None

        assert operand_type == "REG"
        # reg_class = node_properties.get("out_reg_class", None)
        reg_class = operand_reg_classes[op_idx].lower()
        reg_size = node_properties.get("out_reg_size", None)
        reg_name = node_properties.get("out_reg_name", None)
        # enc_bits = int(operand_enc_bits[op_idx])
        is_phys_reg = reg_name.startswith("$x")

        if op_type != "input":
            assert reg_class in ["gpr"], f"Unexpected reg_class: {reg_class}"
            if isinstance(reg_size, str):
                if reg_size == "unknown":
                    reg_size = None
                else:
                    reg_size = int(reg_size)
            if reg_size is not None:
                assert reg_size in [32, 64], f"Unexpected reg_size: {reg_size}"
                if self.xlen is not None:
                    pass  # Ignore XLEN for now: check if casts are added automatically
                    # assert reg_size == xlen, f"reg_size ({reg_size}) does not match xlen ({xlen})"

        if is_phys_reg and (reg_size is None or reg_size == "unknown"):
            if self.xlen is not None:
                reg_size = self.xlen
            else:
                reg_size = "XLEN"
        if op_type == "label" and (reg_size == "unknown" or reg_size is None):
            if self.xlen is not None:
                reg_size = self.xlen
            else:
                reg_size = "XLEN"

        # print("visit", inp)
        res = self.visit(inp)
        # print("res", res)
        # input(">>>")
        # name = f"inp{j}"
        name = f"{operand_name}_val"
        # print("name", name)
        self.defs[inp] = name
        # print("res.name", dir(res), res.name if "name" in dir(res) else None)
        print("reg_class", reg_class)
        # ret[name] = res
        if hasattr(res, "name") and (res.name[:2] == "$x" or res.name[:2] == "$f"):
            print("if")
            idx = int(res.name[2:])
            # print("idx", idx)
            # TODO: make more generic to also work for assignments
            ref_ = Constant(value=idx, in_types=[], out_types=[None])
            # print("ref_", ref_)
            signed = False
            cdsl_type = llvm_type_to_cdsl_type(None, signed, reg_size=reg_size)
            # print("cdsl_type", cdsl_type)
            res = Register(
                name="X[?]",
                children=[ref_],
                reg_class=reg_class,
                in_types=ref_.out_types,
                out_types=[cdsl_type],
            )
            # print("res", res)
        else:
            # name_ = f"rs{j+1}"
            name_ = operand_name
            ref_ = Ref(name=name_, in_types=[], out_types=[None])
            mem_name = mem_lookup.get(reg_class, None)
            assert mem_name is not None, f"Unable to find mem_name for reg_class: {reg_class}"
            signed = False
            cdsl_type = llvm_type_to_cdsl_type(None, signed, reg_size=reg_size)
            res = Register(
                name=f"{mem_name}[?]",
                children=[ref_],
                reg_class=reg_class,
                in_types=ref_.out_types,
                out_types=[cdsl_type],
            )
        ref = Ref(name=name, in_types=[], out_types=[None])
        decl_type = f"unsigned<{reg_size}>"
        root = Declaration(
            children=[ref, res],
            decl_type=decl_type,
            in_types=[ref.out_types[0], res.out_types[0]],
            out_types=[decl_type],
        )
        # print(f"{name}:")
        # print(RenderTree(res))
        return root

    def visit_inputs(self):
        # print("visit_inputs")
        stmts = []
        for i, inp in enumerate(self.inputs):
            input_stmt = self.visit_input(inp)
            if input_stmt is None:
                continue
            # print("input_stmt", input_stmt, type(input_stmt))
            assert isinstance(input_stmt, Statement)
            stmts.append(input_stmt)
        return Statements(
            name="inputs_statements",
            children=stmts,
            op_type="statements",
            in_types=[None] * len(stmts),
            out_types=[None],
        )

    def visit_output(self, outp):
        # print("visit_output", outp)
        operand_types = self.sub_data["OperandTypes"]  # TODO: rename to Classes?
        operand_names = self.sub_data["OperandNames"]
        operand_nodes = self.sub_data["OperandNodes"]
        output_nodes = self.sub_data["OutputNodes"]
        output_names = self.sub_data["OutputNames"]
        op_idx = operand_nodes.index(outp)
        operand_name = operand_names[op_idx]
        out_idx = output_nodes.index(outp)
        output_name = output_names[out_idx]
        # operand_enc_bits = self.sub_data["OperandEncBits"]
        operand_reg_classes = self.sub_data["OperandRegClasses"]
        op_idx = self.resolve_op_idx(outp)
        # print("op_idx", op_idx)
        operand_type = operand_types[op_idx]
        node_data = self.graph.nodes[outp]
        # print("node_data", node_data)
        node_properties = node_data["properties"]
        # print("node_properties", node_properties)
        # print("op_type", op_type)
        operand_type = operand_types[op_idx]
        assert operand_type == "REG"
        # reg_class = node_properties.get("out_reg_class", None)
        reg_class = operand_reg_classes[op_idx].lower()
        assert reg_class in ["gpr"], f"Unexpected reg_class: {reg_class}"
        reg_size = node_properties.get("out_reg_size", None)
        # enc_bits = int(operand_enc_bits[op_idx])
        # reg_size = int(2**enc_bits)
        if isinstance(reg_size, str):
            if reg_size == "unknown":
                reg_size = None
            else:
                reg_size = int(reg_size)
        if reg_size is not None:
            assert reg_size in [32, 64], f"Unexpected reg_size: {reg_size}"
            if self.xlen is not None:
                pass  # Ignore XLEN for now: check if casts are added automatically
                # assert reg_size == self.xlen, f"reg_size ({reg_size}) does not match xlen ({self.xlen})"
        # res = self.visit(outp)
        stmts = []
        temp_idx_before = self.temp_idx
        in_edges = list(self.graph.in_edges(outp))
        assert len(in_edges) == 1
        outp_ = in_edges[0][0]
        ret = self.visit(outp_)
        # print("ret", ret)
        temp_idx_after = self.temp_idx
        # print("temp_idx_before", temp_idx_before)
        # print("temp_idx_after", temp_idx_after)
        if temp_idx_after > temp_idx_before:
            # print("TEMP")
            # input("!")
            refs = [f"temp{idx}" for idx in range(temp_idx_before, temp_idx_after)]
            # print("temps")
            for ref in refs:
                # print("ref", ref)
                # print("self.temps[ref]", self.temps[ref])
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
                stmts.append(decl)
        # # input("123")
        # stmts.append(ret)
        # print("res", res)
        # assert isinstance(res, list)  # TODO: move input/output assignments to tree_utils
        # print(RenderTree(res))
        # input("!!!")
        # TODO: check for multiple statements
        # print("res", res)
        # TODO: check for may_store, may_branch
        # name = f"outp{j}"
        name = output_name
        # print("name", name)
        self.defs[outp] = name
        # print("self.defs", self.defs)
        # ret[name] = root
        # ret[name] = res
        # print("res", res)
        # print(RenderTree(res))
        if hasattr(ret, "name") and ret.name in [
            "SD",
            "SW",
            "SH",
            "SB",
            "BEQ",
            "BNE",
        ]:  # TODO: DETECT via predicates or zero outputs!
            # print("A if")
            root = ret
            stmts.append(root)
        else:
            # print("A else")
            ref = Ref(name=name, in_types=[], out_types=[None])
            ref_ = Ref(name=name, in_types=[], out_types=[None])
            # import pdb; pdb.set_trace()
            # print("ref", ref, ref.children)
            # print("ret", ret, ret.children)
            decl_type = f"unsigned<{reg_size}>"
            root = Declaration(
                children=[ref, ret],
                decl_type=decl_type,
                in_types=[ref.out_types[0], ret.out_types[0]],
                out_types=[decl_type],
            )
            # print("root", root, root.children)
            # print(RenderTree(root))
            stmts.append(root)
            name_ = operand_name
            ref2 = Ref(name=name_, in_types=[], out_types=[None])
            mem_name = mem_lookup.get(reg_class, None)
            assert mem_name is not None, f"Unable to find mem_name for reg_class: {reg_class}"
            reg = Register(
                name=f"{mem_name}[?]",
                children=[ref2],
                reg_class=reg_class,
                in_types=[ref2.out_types[0]],
                out_types=[None],
            )
            # cast_ = AnyNode(children=[ref_], op_type="cast", to=f"unsigned<{reg_size}>")
            # root2 = AnyNode(children=[reg, cast_], op_type="assignment")
            root2 = Assignment(
                children=[reg, ref_],
                in_types=[reg.out_types[0], ref_.out_types[0]],
                out_types=[None],
            )
            stmts.append(root2)
        # print(f"{name}:")
        # print(RenderTree(res))
        return Statements(
            name="output_statements",
            children=stmts,
            op_type="statements",
            in_types=[None] * len(stmts),
            out_types=[None],
        )

    def visit_outputs(self):
        # print("visit_outputs")
        stmts = []
        for i, outp in enumerate(self.outputs):
            # print("i", i)
            # print("outp", outp)
            # output_name = ?
            output_stmts = self.visit_output(outp)
            assert isinstance(output_stmts, Statements)
            stmts.extend(output_stmts.children)
        return Statements(
            name="outputs_statements",
            children=stmts,
            op_type="statements",
            in_types=[None] * len(stmts),
            out_types=[None],
        )

    def generate_tree(self):
        # print("generate_tree")
        # i = 0  # reg
        input_stmts = self.visit_inputs()
        output_stmts = self.visit_outputs()
        all_stmts = input_stmts.children + output_stmts.children

        root = Statements(
            name="statements",
            children=all_stmts,
            op_type="statements",
            in_types=[None] * len(all_stmts),
            out_types=[None],
        )
        # print("root", root)
        # print(RenderTree(root))
        # input(">>>")
        return root
