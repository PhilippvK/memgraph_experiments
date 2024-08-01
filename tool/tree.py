import logging
import networkx as nx
from anytree import AnyNode

from .cdsl_utils import wrap_cdsl
from .cdsl_utils import mem_lookup
from .cdsl_utils import CDSLEmitter
from .cdsl_utils import FlatCodeEmitter  # TODO: move


logger = logging.getLogger("tree")


class TreeGenContext:

    def __init__(self, graph, sub, inputs=None, explicit_types: bool = True) -> None:
        self.graph = graph
        self.sub = sub
        self.inputs = inputs if inputs is not None else []
        self.explicit_types = explicit_types
        self.node_map = {}
        self.defs = {}

    @property
    def visited(self):
        return set(self.node_map.keys())

    def visit(self, node):
        # print("visit", node)
        if node in self.visited:
            op_type = self.graph.nodes[node]["properties"]["op_type"]
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
        print("node", node)
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
        if op_type == "constant":
            val = self.graph.nodes[node]["properties"]["inst"]
            val = int(val[:-1])
            ret = AnyNode(id=-1, value=val, op_type=op_type, children=children)
        else:
            if node in self.inputs:
                op_type = "input"
            ret = AnyNode(id=node, name=name, op_type=op_type, children=children)
        if self.explicit_types:
            type_str = f"unsigned<{out_reg_size}>"
            ret = AnyNode(id=-1, name="?", op_type="cast", children=[ret], to=type_str)
        self.node_map[node] = ret
        return ret


def gen_tree(GF, sub, inputs, outputs, xlen=None):
    ret = {}
    ret_ = []
    operands = {}
    # print("gen_tree", GF, sub, inputs, outputs)
    topo = list(nx.topological_sort(GF))
    inputs = sorted(inputs, key=lambda x: topo.index(x))
    outputs = sorted(outputs, key=lambda x: topo.index(x))
    # treegen = TreeGenContext(sub)
    treegen = TreeGenContext(GF, sub, inputs=inputs)
    # i = 0  # reg
    j = 0  # imm
    for i, inp in enumerate(inputs):
        print("i", i)
        print("inp", inp)
        node_data = GF.nodes[inp]
        print("node_data", node_data)
        node_properties = node_data["properties"]
        print("node_properties", node_properties)
        op_type = node_properties["op_type"]
        print("op_type", op_type)
        reg_class = node_properties.get("out_reg_class", None)
        reg_size = node_properties.get("out_reg_size", None)
        if op_type == "constant":
            continue
        if op_type != "input":
            assert reg_class in ["gpr"], f"Unexpected reg_class: {reg_class}"
            if isinstance(reg_size, str):
                reg_size = int(reg_size)
            assert reg_size in [32, 64], f"Unexpected reg_size: {reg_size}"
            if xlen is not None:
                assert reg_size == xlen, f"reg_size ({reg_size}) does not match xlen ({xlen})"
        res = treegen.visit(inp)
        print("res", res)
        name = f"inp{j}"
        print("name", name)
        # input("<>")
        treegen.defs[inp] = name
        ret[name] = res
        if res.name[:2] == "$x":
            idx = int(res.name[2:])
            # TODO: make more generic to also work for assignments
            ref_ = AnyNode(id=-1, name=res.name, op_type="constant", value=idx)
            res = AnyNode(id=-1, name="X[?]", children=[ref_], op_type="register", reg_class=reg_class)
        else:
            name_ = f"rs{j+1}"
            ref_ = AnyNode(id=-1, name=name_, op_type="ref")
            mem_name = mem_lookup.get(reg_class, None)
            assert mem_name is not None, f"Unable to find mem_name for reg_class: {reg_class}"
            res = AnyNode(id=-1, name=f"{mem_name}[?]", children=[ref_], op_type="register", reg_class=reg_class)
            operand_bits = 5  # log2(32)
            WITH_SEAL5_ATTRS = True
            operand_str = f"unsigned<{operand_bits}> {name_}"
            if WITH_SEAL5_ATTRS:
                attrs = {
                    "is_reg": None,
                    "reg_class": reg_class.upper(),
                    "reg_type": str(xlen),
                    "in": None,
                }
                attr_strs = [
                    f"[[{key}={value}]]" if value is not None else f"[[{key}]]" for key, value in attrs.items()
                ]
                attrs_str = " ".join(attr_strs)
                operand_str += " " + attrs_str
            operands[name_] = (operand_str, reg_class, False)
        ref = AnyNode(id=-1, name=name, op_type="ref")
        decl_type = f"unsigned<{reg_size}>"
        root = AnyNode(id=-1, name="ASSIGN1", children=[ref, res], op_type="declaration", decl_type=decl_type)
        ret_.append(root)
        j += 1
        # print(f"{name}:")
        # print(RenderTree(res))
    j = 0
    for i, outp in enumerate(outputs):
        print("i", i)
        print("outp", outp)
        node_data = GF.nodes[outp]
        print("node_data", node_data)
        node_properties = node_data["properties"]
        print("node_properties", node_properties)
        op_type = node_properties["op_type"]
        print("op_type", op_type)
        reg_class = node_properties.get("out_reg_class", None)
        assert reg_class in ["gpr"], f"Unexpected reg_class: {reg_class}"
        reg_size = node_properties.get("out_reg_size", None)
        if isinstance(reg_size, str):
            reg_size = int(reg_size)
        assert reg_size in [32, 64], f"Unexpected reg_size: {reg_size}"
        if xlen is not None:
            assert reg_size == xlen, f"reg_size ({reg_size}) does not match xlen ({xlen})"
        res = treegen.visit(outp)
        print("res", res)
        # TODO: check for may_store, may_branch
        name = f"outp{j}"
        print("name", name)
        # input("<>")
        treegen.defs[outp] = name
        # ret[name] = root
        ret[name] = res
        # print("res", res)
        # print(RenderTree(res))
        if res.name in ["SD", "SW", "SH", "SB", "BEQ", "BNE"]:
            root = res
            ret_.append(root)
        else:
            ref = AnyNode(id=-1, name=name, op_type="ref")
            ref_ = AnyNode(id=-1, name=name, op_type="ref")
            # import pdb; pdb.set_trace()
            # print("ref", ref, ref.children)
            # print("res", res, res.children)
            decl_type = f"unsigned<{reg_size}>"
            root = AnyNode(id=-1, name="ASSIGN2", children=[ref, res], op_type="declaration", decl_type=decl_type)
            # print("root", root, root.children)
            ret_.append(root)
            idx = j + 1
            name_ = "rd" if idx == 1 else f"rd{idx}"
            ref2 = AnyNode(id=-1, name=name_, op_type="ref")
            reg = AnyNode(id=-1, name=f"{mem_name}[?]", children=[ref2], op_type="register", reg_class=reg_class)
            # cast_ = AnyNode(id=-1, name="CAST3", children=[ref_], op_type="cast", to=f"unsigned<{reg_size}>")
            # root2 = AnyNode(id=-1, name="ASSIGN3", children=[reg, cast_], op_type="assignment")
            root2 = AnyNode(id=-1, name="ASSIGN3", children=[reg, ref_], op_type="assignment")
            ret_.append(root2)
            operand_bits = 5  # log2(32)
            WITH_SEAL5_ATTRS = True
            operand_str = f"unsigned<{operand_bits}> {name_}"
            if WITH_SEAL5_ATTRS:
                attrs = {
                    "is_reg": None,
                    "reg_class": reg_class.upper(),
                    "reg_type": str(xlen),
                    "out": None,
                }
                attr_strs = [
                    f"[[{key}={value}]]" if value is not None else f"[[{key}]]" for key, value in attrs.items()
                ]
                attrs_str = " ".join(attr_strs)
                operand_str += " " + attrs_str
            operands[name_] = (operand_str, reg_class, True)
        j += 1

        # print(f"{name}:")
        # print(RenderTree(res))
    # print("Generating CDSL...")
    codes = []
    header = "// TODO"
    codes.append(header)
    for item in ret_:
        # print("item", item)
        emitter = CDSLEmitter(xlen)
        try:
            emitter.visit(item)
            output = emitter.output
        except Exception as e:
            logger.exception(e)
            codes = None
            input("!!!")
            break
        # print("output", output)
        codes.append(output)
    # print("CDSL Code:")
    if codes is not None:
        codes = ["    " + code for code in codes]
        asm_ins = []
        asm_outs = []
        mnemonic = "myinst"
        operands_code = "operands: {\n"
        for operand_name, operand in operands.items():
            operand_code, reg_class, is_output = operand
            asm_str = operand_name
            if reg_class == "gpr":
                asm_str = f"name({operand_name})"
            elif reg_class == "gpr":
                asm_str = f"fname({operand_name})"
            asm_str = f"{{{asm_str}}}"
            if is_output:
                asm_outs.append(asm_str)
            else:
                asm_ins.append(asm_str)
            operands_code += "    " + operand_code + ";\n"
        asm_all = asm_outs + asm_ins
        asm_syntax = ", ".join(asm_all)
        operands_code += "}"
        codes = (
            [operands_code, "encoding: auto;", f'assembly: {{"{mnemonic}", {asm_syntax}}};', "behavior: {"]
            + codes
            + ["}"]
        )
        codes = wrap_cdsl("MyInst", "\n".join(codes)).split("\n")
        codes = ["    " + code for code in codes]
        code = "\n".join(codes) + "\n"
        code = f"""InstructionSet MySet extends RV{xlen}I {{
    instructions {{
{code}
    }}
}}
"""
        # TODO: add encoding etc.!
    else:
        code = None
    # print(code)
    # print("Done!")
    return ret, ret_, code


def gen_flat_code(xtrees, desc=None):
    # print("gen_flat_code")
    # print("xtrees", xtrees)
    # input(">")
    codes = []
    if desc:
        header = f"// {desc}"
        codes.append(header)
    for item in xtrees:
        # print("item", item)
        emitter = FlatCodeEmitter()
        emitter.visit(item)
        output = emitter.output
        # print("output", output)
        codes.append(output)
    code = "\n".join(codes) + "\n"
    # print("code", code)
    # input(">>")
    return code
