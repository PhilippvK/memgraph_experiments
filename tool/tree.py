import logging
import networkx as nx
from anytree import AnyNode

from .cdsl_utils import CDSLEmitter
from .cdsl_utils import FlatCodeEmitter  # TODO: move


logger = logging.getLogger("tree")


class TreeGenContext:

    def __init__(self, graph, sub, inputs=None) -> None:
        self.graph = graph
        self.sub = sub
        self.inputs = inputs if inputs is not None else []
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
        srcs = [src for src, _ in self.graph.in_edges(node)]
        srcs = [src for src in srcs if src in self.inputs or src in self.sub.nodes]
        children = [self.visit(src) for src in srcs]
        # print("children", children)
        op_type = self.graph.nodes[node]["properties"]["op_type"]
        name = self.graph.nodes[node]["properties"]["name"]
        if op_type == "constant":
            val = self.graph.nodes[node]["properties"]["inst"]
            val = int(val[:-1])
            ret = AnyNode(id=-1, value=val, op_type=op_type, children=children)
        else:
            if node in self.inputs:
                op_type = "input"
            ret = AnyNode(id=node, name=name, op_type=op_type, children=children)
        self.node_map[node] = ret
        return ret


def gen_tree(GF, sub, inputs, outputs, xlen=None):
    ret = {}
    ret_ = []
    # print("gen_tree", GF, sub, inputs, outputs)
    topo = list(nx.topological_sort(GF))
    inputs = sorted(inputs, key=lambda x: topo.index(x))
    outputs = sorted(outputs, key=lambda x: topo.index(x))
    # treegen = TreeGenContext(sub)
    treegen = TreeGenContext(GF, sub, inputs=inputs)
    # i = 0  # reg
    j = 0  # imm
    for i, inp in enumerate(inputs):
        op_type = GF.nodes[inp]["properties"]["op_type"]
        if op_type == "constant":
            continue
        res = treegen.visit(inp)
        name = f"inp{j}"
        treegen.defs[inp] = name
        ret[name] = res
        if res.name[:2] == "$x":
            idx = int(res.name[2:])
            # TODO: make more generic to also work for assignments
            ref_ = AnyNode(id=-1, name=res.name, op_type="constant", value=idx)
            res = AnyNode(id=-1, name="X[?]", children=[ref_], op_type="register")
        else:
            name_ = f"rs{j+1}"
            ref_ = AnyNode(id=-1, name=name_, op_type="ref")
            res = AnyNode(id=-1, name="X[?]", children=[ref_], op_type="register")
        ref = AnyNode(id=-1, name=name, op_type="ref")
        root = AnyNode(id=-1, name="ASSIGN1", children=[ref, res], op_type="assignment")
        ret_.append(root)
        j += 1
        # print(f"{name}:")
        # print(RenderTree(res))
    j = 0
    for i, outp in enumerate(outputs):
        res = treegen.visit(outp)
        # TODO: check for may_store, may_branch
        name = f"outp{j}"
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
            root = AnyNode(id=-1, name="ASSIGN2", children=[ref, res], op_type="assignment")
            # print("root", root, root.children)
            ret_.append(root)
            idx = j + 1
            name_ = "rd" if idx == 1 else f"rd{idx}"
            ref2 = AnyNode(id=-1, name=name_, op_type="ref")
            reg = AnyNode(id=-1, name="X[?]", children=[ref2], op_type="register")
            root2 = AnyNode(id=-1, name="ASSIGN3", children=[reg, ref_], op_type="assignment")
            ret_.append(root2)
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
            break
        # print("output", output)
        codes.append(output)
    # print("CDSL Code:")
    if codes is not None:
        codes = ["    " + code for code in codes]
        codes = ["operands: TODO;", "encoding: auto;", 'assembly: {TODO, "TODO"};', "behavior: {"] + codes + ["}"]
        code = "\n".join(codes) + "\n"
    else:
        code = None
    # print(code)
    # print("Done!")
    return ret, ret_, code


def gen_flat_code(xtrees, desc=None):
    print("gen_flat_code")
    print("xtrees", xtrees)
    input(">")
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
    print("code", code)
    input(">>")
    return code
