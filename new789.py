from collections import defaultdict
from neo4j import GraphDatabase
import networkx as nx
from networkx.drawing.nx_agraph import write_dot
import matplotlib.pyplot as plt

driver = GraphDatabase.driver('bolt://localhost:7687', auth=("", ""))

SESSION = "default"

from anytree import AnyNode, RenderTree

class TreeGenContext:

    def __init__(self, graph, sub, inputs=None) -> None:
       self.graph = graph
       self.sub = sub
       self.inputs = inputs if inputs is not None else []
       # self.tree = tree
       # self.parent_stack = [root]
       # self.visited = []  # self.node_map.keys()?
       self.node_map = {}
       self.defs = {}

    @property
    def visited(self):
        return set(self.node_map.keys())

    # @property
    # def parent(self):
    #     return self.parent_stack[-1]

    # def push(self, new_id):
    #     print("push", new_id)
    #     self.parent_stack.append(new_id)

    # def pop(self):
    #     print("pop")
    #     return self.parent_stack.pop()

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


def gen_tree(GF, sub, inputs, outputs):
    ret = {}
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
        print(f"{name}:")
        print(RenderTree(res))
    for i, outp in enumerate(outputs):
        res = treegen.visit(outp)
        name = f"outp{i}"
        treegen.defs[outp] = name
        # root = AnyNode(id=-1, name=name, children=[res])
        # ret[name] = root
        ret[name] = res
        print(f"{name}:")
        # print(RenderTree(root))
        print(RenderTree(res))
    return ret


def gen_mir_func(func_name, code, desc=None):
    ret = "# MIR"
    if desc:
        ret += f" [{desc}]"
    ret += f"""
---
name: {func_name}
body: |
  bb.0:
"""
    ret += "\n".join(["    " + line for line in code.splitlines()])
    return ret


# PHI nodes sometimes create cycles which are not allowed,
# hence we drop all ingoing edges to PHIs as their src MI
# is automatically marked as OUTPUT anyways.
query_func = f"""
MATCH p0=(n00)-[r01:DFG]->(n01)
WHERE n00.func_name = 'tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_per_axis_add_clip_cast_subtract_1_compute_'
AND n01.name != "PHI"
AND n00.session = "{SESSION}"
RETURN p0;
"""
query = f"""
MATCH p0=(n00)-[r01:DFG]->(n01)
WHERE n00.func_name = 'tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_per_axis_add_clip_cast_subtract_1_compute_'
AND n00.session = "{SESSION}"
AND n00.op_type != "input" AND n01.op_type != "input" AND n02.op_type != "input"
AND n00.name != "COPY" AND n01.name != "COPY" AND n02.name != "COPY"
AND n00.name != "PHI" AND n01.name != "PHI" AND n02.name != "PHI"
AND n00.name != "Const" AND n01.name != "Const" AND n02.name != "Const"
// AND n00.name != "Reg" AND n01.name != "Reg"
// AND n00.name != "$x0" AND n01.name != "$x0"
// AND n00.name != "$x1" AND n01.name != "$x1"
// AND n00.name != "$x2" AND n01.name != "$x2"
AND n00.name != "PseudoCALLIndirect" AND n01.name != "PseudoCALLIndirect" AND n02.name != "PseudoCALLIndirect"
AND n00.name != "PseudoLGA" AND n01.name != "PseudoLGA" AND n02.name != "PseudoLGA"
AND n00.name != "Select_GPR_Using_CC_GPR" AND n01.name != "Select_GPR_Using_CC_GPR"

RETURN p0
// , count(*) as count
// ORDER BY count DESC
"""
query2 = f"""
MATCH p0=(n00)-[r01:DFG]->(n01)-[r02:DFG]->(n02)
WHERE n00.func_name = 'tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_per_axis_add_clip_cast_subtract_1_compute_'
AND n00.session = "{SESSION}"
AND n00.op_type != "input" AND n01.op_type != "input"
AND n00.name != "COPY" AND n01.name != "COPY"
AND n00.name != "PHI" AND n01.name != "PHI"
AND n00.name != "Const" AND n01.name != "Const"
// AND n00.name != "Reg" AND n01.name != "Reg"
// AND n00.name != "$x0" AND n01.name != "$x0"
// AND n00.name != "$x1" AND n01.name != "$x1"
// AND n00.name != "$x2" AND n01.name != "$x2"
AND n00.name != "PseudoCALLIndirect" AND n01.name != "PseudoCALLIndirect"
AND n00.name != "PseudoLGA" AND n01.name != "PseudoLGA"
AND n00.name != "Select_GPR_Using_CC_GPR" AND n01.name != "Select_GPR_Using_CC_GPR"

RETURN p0
// , count(*) as count
// ORDER BY count DESC
"""

func_results = driver.session().run(query_func)
results = driver.session().run(query2)
# print("results", results, dir(results))
# print("results.df", results.to_df())
# nodes = list(func_results.graph()._nodes.values())

GF = nx.MultiDiGraph()
nodes = list(func_results.graph()._nodes.values())
# print("nodes", nodes)
for node in nodes:
    # print("node", node)
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
# print("GF", GF)
write_dot(GF, f"func.dot")
# input("?")

G = nx.MultiDiGraph()
nodes = list(results.graph()._nodes.values())
# print("nodes", nodes)
# input("z")
for node in nodes:
    # print("node", node)
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
# print("G", G)
write_dot(G, f"results.dot")

subs = []
for i, result in enumerate(results):
    # print("result", result, dir(result))
    # print("result.data", result.data())
    # print("result.value", result.value())
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
    # print("G_", G_)
    write_dot(G_, f"result{i}.dot")
    count = subs.count(G_)
    if count > 0:
        pass
        # input("2")
    subs.append(G_)

# print("subs", subs)

# for i, result in enumerate(results):
#     print("result", result, i, dir(result), result.data())
#     input(">")
# input(">>>")

# print("GF", GF)
# print("GF.nodes", GF.nodes)
mapping = dict(zip(GF.nodes.keys(), range(len(GF.nodes))))
mapping1 = mapping
GF = nx.relabel_nodes(GF, mapping)
G = nx.relabel_nodes(G, mapping)
for i in range(len(subs)):
    subs[i] = nx.relabel_nodes(subs[i], mapping)
# print("GF", GF)
# print("G", G)
# print("GF.nodes", GF.nodes)
# print("G.nodes", G.nodes)
# mapping = dict(zip(G.nodes.keys(), range(len(G.nodes))))
# G = nx.relabel_nodes(G, mapping)
# for i in range(len(subs)):
#     subs[i] = nx.relabel_nodes(subs[i], mapping)
# print("G", G)
# print("G.nodes", G.nodes)
# topo = list(reversed(list(nx.topological_sort(G))))
topo = list(reversed(list(nx.topological_sort(GF))))
# print("topo", topo)
# mapping = dict(zip(G.nodes.keys(), topo))
mapping = dict(zip(GF.nodes.keys(), topo))
GF = nx.relabel_nodes(GF, mapping)
G = nx.relabel_nodes(G, mapping)
for i in range(len(subs)):
    subs[i] = nx.relabel_nodes(subs[i], mapping)
# print("GF", GF)
# print("G", G)
# print("GF.nodes", GF.nodes)
# print("G.nodes", G.nodes)
# topo2 = list(reversed(list(nx.topological_sort(GF))))
# print("topo2", topo2)

# # @A
# topo2 = list(nx.topological_sort(subs[380]))
# print("topo2", topo2)
# # ADDI
# print("mapping1[260397]", mapping1[260397])
# print("mapping[1094]", mapping[1094])
# print("55", subs[380].nodes[55])
# # BEQ
# print("mapping1[260396]", mapping1[260396])
# print("mapping[1093]", mapping[1093])
# print("57", subs[380].nodes[57])
#
# # @B
# topo2 = list(nx.topological_sort(subs[383]))
# print("topo2", topo2)
# # SLTU
# print("mapping1[260420]", mapping1[260420])
# print("mapping[1116]", mapping[1116])
# print("6", subs[383].nodes[6])
# # SLLI
# print("mapping1[260421]", mapping1[260421])
# print("mapping[1117]", mapping[1117])
# print("4", subs[383].nodes[4])
# input("<>")
topo = list(nx.topological_sort(GF))

# print("subs[0]", subs[0])
# print("subs[0].nodes", subs[0].nodes)

def calc_inputs(G, sub, ignore_const: bool = False):
   # print("calc_inputs", sub)
   inputs = []
   ret = 0
   sub_nodes = sub.nodes
   # print("sub_nodes", sub_nodes)
   for node in sub_nodes:
     # print("node", node, G.nodes[node].get("label"))
     ins = G.in_edges(node)
     # print("ins", ins)
     for in_ in ins:
       # print("in_", in_, G.nodes[in_[0]].get("label"))
       src = in_[0]
       if G.nodes[src]["properties"]["op_type"] == "constant" and ignore_const:
          continue
       # print("src", src, G.nodes[src].get("label"))
       # print("src in sub_nodes", src in sub_nodes)
       # print("src not in inputs", src not in inputs)
       if not (src in sub_nodes) and (src not in inputs):
         # print("IN")
         ret += 1
         inputs.append(src)
   # print("ret", ret)
   return ret, inputs

def calc_outputs(G, sub):
   # print("calc_outputs", sub)
   ret = 0
   sub_nodes = sub.nodes
   # print("sub_nodes", sub_nodes)
   outputs = []
   for node in sub_nodes:
     # print("node", node, G.nodes[node].get("label"))
     if G.nodes[node]["properties"]["op_type"] == "output":
       # print("A")
       # print("OUT2")
       ret += 1
       if node not in outputs:
         outputs.append(node)
     else:
       # print("B")
       outs = G.out_edges(node)
       # print("outs", outs)
       for out_ in outs:
         # print("out_", out_, G.nodes[out_[0]].get("label"))
         dst = out_[1]
         # print("dst", dst, G.nodes[dst].get("label"))
         if dst not in sub_nodes:
           # print("OUT")
           ret += 1
           if node not in outputs:
             outputs.append(node)
   # print("ret", ret)
   # input("1")
   return ret, outputs


all_codes = {}
duplicate_counts = defaultdict(int)

# if True:
for i, sub in enumerate(subs):
    print("===========================")
    # i = 3
    sub = subs[i]
    # print("topo", topo)
    # print("i, sub", i, sub)
    codes = []
    # for node in sorted(sub.nodes):
    # print("sub.nodes", sub.nodes)
    for node in sorted(sub.nodes, key=lambda x: topo.index(x)):
        node_ = G.nodes[node]
        code_ = node_["properties"]["inst"]
        # assert code_[-1] == "_"
        # code_ = code_[:-1]
        code_ = code_.split(", debug-location", 1)[0]
        if code_[-1] != "_":
            code_ += "_"
        codes.append(code_)
        # print("CODE", node_["properties"]["inst"])
    code = "\n".join(codes)
    # print(f"Code:\n{code}")
    num_inputs, inputs = calc_inputs(GF, sub)
    num_inputs_noconst, inputs_noconst = calc_inputs(GF, sub, ignore_const=True)
    num_outputs, outputs = calc_outputs(GF, sub)
    print("num_inputs", num_inputs)
    print("num_inputs_noconst", num_inputs_noconst)
    print("num_outputs", num_outputs)
    # print("inputs", [GF.nodes[inp] for inp in inputs])
    # print("outputs", [GF.nodes[outp] for outp in outputs])
    j = 0  # reg's
    # j_ = 0  # imm's
    for inp in inputs:
        node = GF.nodes[inp]
        inst = node["properties"]["inst"]
        op_type = node["properties"]["op_type"]
        # print("inst", inst)
        if "=" in inst:
            name = f"%inp{j}:gpr"
            j += 1
            # print("if")
            lhs, _ = inst.split("=", 1)
            lhs = lhs.strip()
            assert "gpr" in lhs
            code = code.replace(lhs, name)
        else:
            # print("else")
            if inst.startswith("$x"):  # phys reg
                pass
                # physreg = inst[:-1]
                # tmp = physreg[2:]
                # new = f"X[{tmp}]"
                # code = code.replace(physreg, new)
            else:
                assert op_type == "constant"
                assert inst[-1] == "_"
                const = inst[:-1]
                val = int(const)
                def get_ty_for_val(val):
                    def get_min_pow(x):
                        # print("x", x)
                        assert x >= 0
                        max_pow = 6
                        for i in range(max_pow + 1):
                            # print("i", i)
                            pow_val = 2**i
                            # print("pow_val", pow_val)
                            if x < 2**pow_val:
                                return pow_val
                        assert False
                    if val < 0:
                        val *= -1
                    min_pow = get_min_pow(val)
                    return f"i{min_pow}"


                ty = get_ty_for_val(val)
                # print("code", code)
                # print("ty", ty)
                # print("const", const)
                # print("inst", inst)
                # print("const", const)
                # if inst[-1] == "_":
                #     inst = inst[:-1]
                code = code.replace(" " + inst, f" {ty} " + const)  # TODO: buggy?
                # print("code")
    for j, outp in enumerate(outputs):
        # print("name", name)
        node = GF.nodes[outp]
        inst = node["properties"]["inst"]
        if "=" in inst:
            name = f"%outp{j}:gpr"
            # print("if")
            lhs, _ = inst.split("=", 1)
            lhs = lhs.strip()
            assert "gpr" in lhs
            # print("lhs", lhs)
            code = code.replace(lhs, name)
        else:
            # print("else")
            pass  # TODO: assert?
    # TODO: handle bbs:
    is_branch = False
    if "bb." in code:
        is_branch = True
    """
---
name: result273
body: |
  bb.0:
    successors: %bb.77
    %outp0:gpr = nuw nsw ADDI %inp0:gpr, i1 1
    BEQ %outp0:gpr, %inp1:gpr, %bb.77

  bb.77:
    PseudoRET
    """
    # TODO: may_load, may_store,...
    # print(f"Code2:\n{code}")
    code = "\n".join([line[:-1] if line.endswith("_") else line for line in code.splitlines()])
    if code in all_codes.values():
        print("Duplicate!")
        orig = list(all_codes.keys())[list(all_codes.values()).index(code)]
        duplicate_counts[orig] += 1
        continue
    all_codes[i] = code
    desc = f"Inputs (with imm): {num_inputs}, Inputs (without imm): {num_inputs_noconst}, Outputs: {num_outputs}"
    if is_branch:
        desc += ", IsBranch"
    mir_code = gen_mir_func(f"result{i}", code, desc=desc)
    # print(f"Code3:\n{mir_code}")
    print(mir_code)
    with open(f"result{i}.mir", "w") as f:
        f.write(mir_code)
    if num_outputs == 0 or num_inputs == 0:
        pass
        # input(">?")
    elif num_outputs in [1] and num_inputs in [1, 2, 3]:
        pass
        # input(">")
    # print("---------------------------")
    tree = gen_tree(GF, sub, inputs, outputs)
    print("tree", tree)
    input("123")

if len(duplicate_counts) > 0:
    print()
    print("Duplicates:")
    for orig, dups in duplicate_counts.items():
        print(f"result{orig}:\t", dups)
