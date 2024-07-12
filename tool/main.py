import logging
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
import plotly.express as px

# import matplotlib.pyplot as plt

# from anytree import RenderTree
# from anytree.iterators import AbstractIter

from .enums import ExportFormat, ExportFilter
from .memgraph import connect_memgraph, run_query
from .cdsl_utils import wrap_cdsl
from .mir_utils import gen_mir_func
from .graph_utils import graph_to_file
from .tree import gen_tree
from .queries import generate_func_query, generate_candidates_query

logger = logging.getLogger("main")


# TODO: rename result to sub & gen
# TODO: actually implement filters
FUNC_FMT_DEFAULT = ExportFormat.DOT
FUNC_FLT_DEFAULT = ExportFilter.SELECTED
SUB_FMT_DEFAULT = ExportFormat.DOT | ExportFormat.PDF | ExportFormat.PNG
SUB_FLT_DEFAULT = ExportFilter.SELECTED
IO_SUB_FMT_DEFAULT = ExportFormat.DOT | ExportFormat.PDF | ExportFormat.PNG
IO_SUB_FLT_DEFAULT = ExportFilter.SELECTED
GEN_FMT_DEFAULT = ExportFormat.CDSL | ExportFormat.MIR
GEN_FLT_DEFAULT = ExportFilter.SELECTED
PIE_FMT_DEFAULT = ExportFormat.PDF | ExportFormat.CSV
DF_FMT_DEFAULT = ExportFormat.CSV
IGNORE_NAMES_DEFAULT = ["PHI", "COPY", "PseudoCALLIndirect", "PseudoLGA", "Select_GPR_Using_CC_GPR"]
IGNORE_OP_TYPES_DEFAULT = ["input", "constant"]


def handle_cmdline():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--log", default="info", choices=["critical", "error", "warning", "info", "debug"])
    parser.add_argument("--host", default="localhost", help="TODO")
    parser.add_argument("--port", type=int, default=7687, help="TODO")
    parser.add_argument("--session", default="default", help="TODO")
    parser.add_argument("--max-inputs", type=int, default=3, help="TODO")
    parser.add_argument("--max-outputs", type=int, default=2, help="TODO")
    parser.add_argument("--max-nodes", type=int, default=5, help="TODO")
    parser.add_argument("--min-path-length", type=int, default=1, help="TODO")
    parser.add_argument("--max-path-length", type=int, default=3, help="TODO")
    parser.add_argument("--max-path-width", type=int, default=2, help="TODO")
    parser.add_argument("--function", "--func", default=None, help="TODO")
    parser.add_argument("--basic-block", "--bb", default=None, help="TODO")
    parser.add_argument("--ignore-names", default=",".join(IGNORE_NAMES_DEFAULT), help="TODO")
    parser.add_argument("--ignore-op-types", default=",".join(IGNORE_OP_TYPES_DEFAULT), help="TODO")
    parser.add_argument("--ignore-const-inputs", action="store_true", help="TODO")
    parser.add_argument("--xlen", type=int, default=64, help="TODO")
    parser.add_argument("--output-dir", "-o", default="./out", help="TODO")
    parser.add_argument("--write-func", action="store_true", help="TODO")
    parser.add_argument("--write-func-fmt", type=int, default=FUNC_FMT_DEFAULT, help="TODO")
    parser.add_argument("--write-func-flt", type=int, default=FUNC_FLT_DEFAULT, help="TODO")
    parser.add_argument("--write-sub", action="store_true", help="TODO")
    parser.add_argument("--write-sub-fmt", type=int, default=SUB_FMT_DEFAULT, help="TODO")
    parser.add_argument("--write-sub-flt", type=int, default=SUB_FLT_DEFAULT, help="TODO")
    parser.add_argument("--write-io-sub", action="store_true", help="TODO")
    parser.add_argument("--write-io-sub-fmt", type=int, default=IO_SUB_FMT_DEFAULT, help="TODO")
    parser.add_argument("--write-io-sub-flt", type=int, default=IO_SUB_FLT_DEFAULT, help="TODO")
    parser.add_argument("--write-gen", action="store_true", help="TODO")
    parser.add_argument("--write-gen-fmt", type=int, default=GEN_FMT_DEFAULT, help="TODO")
    parser.add_argument("--write-gen-flt", type=int, default=GEN_FLT_DEFAULT, help="TODO")
    parser.add_argument("--write-pie", action="store_true", help="TODO")
    parser.add_argument("--write-pie-fmt", type=int, default=PIE_FMT_DEFAULT, help="TODO")
    # TODO: pie filters?
    parser.add_argument("--write-df", action="store_true", help="TODO")
    parser.add_argument("--write-df-fmt", type=int, default=DF_FMT_DEFAULT, help="TODO")
    # TODO: df filters?
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()))
    logging.getLogger("neo4j.io").setLevel(logging.INFO)
    logging.getLogger("neo4j.pool").setLevel(logging.INFO)
    return args


args = handle_cmdline()
MAX_INPUTS = args.max_inputs
MAX_OUTPUTS = args.max_outputs
MAX_NODES = args.max_nodes
XLEN = args.xlen
OUT = Path(args.output_dir)
SESSION = args.session
HOST = args.host
PORT = args.port
FUNC = args.function
BB = args.basic_block
MIN_PATH_LEN = args.min_path_length
MAX_PATH_LEN = args.max_path_length
MAX_PATH_WIDTH = args.max_path_width
IGNORE_NAMES = args.ignore_names.split(",")
IGNORE_OP_TYPES = args.ignore_op_types.split(",")
IGNORE_CONST_INPUTS = args.ignore_const_inputs
WRITE_FUNC = args.write_func
WRITE_FUNC_FMT = args.write_func_fmt
WRITE_FUNC_FLT = args.write_func_flt
WRITE_SUB = args.write_sub
WRITE_SUB_FMT = args.write_sub_fmt
WRITE_SUB_FLT = args.write_sub_flt
WRITE_IO_SUB = args.write_io_sub
WRITE_IO_SUB_FMT = args.write_io_sub_fmt
WRITE_IO_SUB_FLT = args.write_io_sub_flt
WRITE_GEN = args.write_gen
WRITE_GEN_FMT = args.write_gen_fmt
WRITE_GEN_FLT = args.write_gen_flt
WRITE_PIE = args.write_pie
WRITE_PIE_FMT = args.write_pie_fmt
WRITE_DF = args.write_df
WRITE_DF_FMT = args.write_df_fmt

assert OUT.is_dir(), f"OUT ({OUT}) is not a directory"
assert FUNC is not None
if not IGNORE_CONST_INPUTS:
    raise NotImplementedError("!IGNORE_CONST_INPUTS")


driver = connect_memgraph(HOST, PORT, user="", password="")

query_func = generate_func_query(SESSION, FUNC)
query = generate_candidates_query(
    SESSION, FUNC, BB, MIN_PATH_LEN, MAX_PATH_LEN, MAX_PATH_WIDTH, IGNORE_NAMES, IGNORE_OP_TYPES
)

func_results = run_query(driver, query_func)
results = run_query(driver, query)

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
    # GF.add_edge(
    #     rel.start_node.element_id, rel.end_node.element_id, key=rel.element_id,
    #     label=label, type=rel.type, properties=rel._properties,
    # )
    GF.add_edge(rel.start_node.id, rel.end_node.id, key=rel.id, label=label, type=rel.type, properties=rel._properties)
# print("GF", GF)


if WRITE_FUNC:
    if WRITE_FUNC_FMT & ExportFormat.DOT:
        graph_to_file(GF, OUT / "func.dot")
    if WRITE_FUNC_FMT & ExportFormat.PDF:
        graph_to_file(GF, OUT / "func.pdf")
    if WRITE_FUNC_FMT & ExportFormat.PNG:
        graph_to_file(GF, OUT / "func.png")


G = nx.MultiDiGraph()
nodes = list(results.graph()._nodes.values())
# print("nodes", nodes)
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
    # G.add_edge(
    #     rel.start_node.element_id, rel.end_node.element_id, key=rel.element_id,
    #     label=label, type=rel.type, properties=rel._properties
    # )
    G.add_edge(rel.start_node.id, rel.end_node.id, key=rel.id, label=label, type=rel.type, properties=rel._properties)

subs = []
for i, result in enumerate(results):
    # print("result", result, dir(result))
    # print("result.data", result.data())
    # print("result.value", result.value())
    nodes_ = set()
    # path = result.value()
    for path in result:
        # print("path", path)
        # print("path", path, dir(path))
        nodes__ = path.nodes
        # print("nodes__", nodes__[0].element_id)
        # 'count', 'data', 'get', 'index', 'items', 'keys', 'value', 'values'
        nodes_ |= {n.id for n in nodes__}
    # print("nodes_", nodes_)
    G_ = G.subgraph(nodes_)
    # G_ = nx.subgraph_view(G, filter_node=lambda x: x in nodes_)
    # print("G_", G_)
    if WRITE_SUB:
        if WRITE_SUB_FMT & ExportFormat.DOT:
            graph_to_file(G_, OUT / f"sub{i}.dot")
        if WRITE_SUB_FMT & ExportFormat.PDF:
            graph_to_file(G_, OUT / f"sub{i}.pdf")
        if WRITE_SUB_FMT & ExportFormat.PNG:
            graph_to_file(G_, OUT / f"sub{i}.png")
    count = subs.count(G_)
    if count > 0:
        pass
    subs.append(G_)

# for i, result in enumerate(results):
#     print("result", result, i, dir(result), result.data())

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
    return ret, outputs


# print("subs", subs, len(subs))
# isos = set()
# for i, sub in enumerate(subs):
#     print("sub", sub, sub.nodes)
#     nm = lambda x, y: x["label"] == y["label"]
#     isos_ = set(j for j, sub_ in enumerate(subs) if j > i and nx.is_isomorphic(sub, sub_, node_match=nm))
#     iso_count = len(isos_)
#     isos |= isos_
#     print("iso_count", iso_count)
# print("isos", isos, len(isos))


io_subs = []
all_codes = {}
errs = set()
filtered_io = set()
filtered_complex = set()
invalid = set()
duplicate_counts = defaultdict(int)


subs_df = pd.DataFrame({"result": list(range(len(subs)))})
subs_df["Inputs"] = [np.array([])] * len(subs_df)
subs_df["#Inputs"] = np.nan
subs_df["InputsNC"] = [np.array([])] * len(subs_df)
subs_df["#InputsNC"] = np.nan
subs_df["Outputs"] = [np.array([])] * len(subs_df)
subs_df["#Outputs"] = np.nan
# print("subs_df")
# print(subs_df)

# if True:
for i, sub in enumerate(subs):
    # if i in isos:
    #     continue
    # i = 3
    # sub = subs[i]
    # print("topo", topo)
    # print("===========================")
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
    subs_df.at[i, "Inputs"] = set(inputs)
    subs_df.loc[i, "#Inputs"] = num_inputs
    subs_df.at[i, "InputsNC"] = set(inputs_noconst)
    subs_df.loc[i, "#InputsNC"] = num_inputs_noconst
    subs_df.at[i, "Outputs"] = set(outputs)
    subs_df.loc[i, "#Outputs"] = num_outputs
    # print("num_inputs", num_inputs)
    # print("num_inputs_noconst", num_inputs_noconst)
    # print("num_outputs", num_outputs)
    # print("inputs", [GF.nodes[inp] for inp in inputs])
    # print("outputs", [GF.nodes[outp] for outp in outputs])
    # TODO: copy fine?
    io_sub = GF.subgraph(list(sub.nodes) + inputs).copy()
    j = 0
    for inp in inputs:
        if io_sub.nodes[inp]["label"] == "Const":
            io_sub.nodes[inp]["xlabel"] = "CONST"
            io_sub.nodes[inp]["label"] = io_sub.nodes[inp]["properties"]["inst"][:-1]
        else:
            io_sub.nodes[inp]["xlabel"] = "IN"
            io_sub.nodes[inp]["label"] = f"src{j}"
            j += 1
    # print("io_sub", io_sub)
    io_subs.append(io_sub)

# print("io_subs", [str(x) for x in io_subs], len(io_subs))
io_isos = set()
for i, io_sub in enumerate(io_subs):
    # break  # TODO
    # print("io_sub", i, io_sub, io_sub.nodes)
    # print("io_sub nodes", [GF.nodes[n] for n in io_sub.nodes])
    def node_match(x, y):
        return x["label"] == y["label"] and (
            x["label"] != "Const" or x["properties"]["inst"] == y["properties"]["inst"]
        )

    io_isos_ = set(
        j for j, io_sub_ in enumerate(io_subs) if j > i and nx.is_isomorphic(io_sub, io_sub_, node_match=node_match)
    )
    # print("io_isos_", io_isos_)
    io_iso_count = len(io_isos_)
    # print("io_iso_count", io_iso_count)
    io_isos |= io_isos_
# print("subs_df")
# print(subs_df)
# print("io_isos", io_isos, len(io_isos))

if WRITE_IO_SUB:
    for i, io_sub in enumerate(io_subs):
        if i in io_isos:
            continue
        if WRITE_IO_SUB_FMT & ExportFormat.DOT:
            graph_to_file(io_sub, OUT / f"io_sub{i}.dot")
        if WRITE_IO_SUB_FMT & ExportFormat.PDF:
            graph_to_file(io_sub, OUT / f"io_sub{i}.pdf")
        if WRITE_IO_SUB_FMT & ExportFormat.PNG:
            graph_to_file(io_sub, OUT / f"io_sub{i}.png")

for i, sub in enumerate(subs):
    if i in io_isos:
        continue
    # print("===========================")
    # print("i, sub", i, sub)
    num_nodes = len(sub.nodes)
    sub_data = subs_df.iloc[i]
    inputs = sub_data["Inputs"]
    num_inputs = int(sub_data["#Inputs"])
    inputs_noconst = sub_data["InputsNC"]
    num_inputs_noconst = int(sub_data["#InputsNC"])
    outputs = sub_data["Outputs"]
    num_outputs = int(sub_data["#Outputs"])
    if num_inputs_noconst == 0 or num_outputs == 0:
        invalid.add(i)
    elif num_inputs_noconst <= MAX_INPUTS and num_outputs <= MAX_OUTPUTS:
        if num_nodes > MAX_NODES:
            filtered_complex.add(i)
    else:
        filtered_io.add(i)

for i, sub in enumerate(subs):
    if i in io_isos or i in filtered_io or i in filtered_complex:
        continue
    # print("===========================")
    # print("i, sub", i, sub)
    sub_data = subs_df.iloc[i]
    inputs = sub_data["Inputs"]
    num_inputs = int(sub_data["#Inputs"])
    inputs_noconst = sub_data["InputsNC"]
    num_inputs_noconst = int(sub_data["#InputsNC"])
    outputs = sub_data["Outputs"]
    num_outputs = int(sub_data["#Outputs"])
    # print("num_inputs", num_inputs)
    # print("num_inputs_noconst", num_inputs_noconst)
    # print("num_outputs", num_outputs)
    # print("inputs", [GF.nodes[inp] for inp in inputs])
    # print("inputs_noconst", [GF.nodes[inp] for inp in inputs_noconst])
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
        # print("Duplicate!")
        orig = list(all_codes.keys())[list(all_codes.values()).index(code)]
        duplicate_counts[orig] += 1
        # continue
    else:
        all_codes[i] = code
    desc = f"Inputs (with imm): {num_inputs}, Inputs (without imm): {num_inputs_noconst}, Outputs: {num_outputs}"
    if is_branch:
        desc += ", IsBranch"
    mir_code = gen_mir_func(f"result{i}", code, desc=desc)
    # print(f"Code3:\n{mir_code}")
    # print(mir_code)
    if WRITE_GEN:
        if WRITE_GEN_FMT & ExportFormat.MIR:
            with open(OUT / f"result{i}.mir", "w") as f:
                f.write(mir_code)
    if num_outputs == 0 or num_inputs == 0:
        pass
    elif num_outputs in [1] and num_inputs in [1, 2, 3]:
        pass
    # print("---------------------------")
    try:
        tree, cdsl_code = gen_tree(GF, sub, inputs, outputs, xlen=XLEN)
    except AssertionError as e:
        print(e)  # TODO: logger.exception
        errs.add(i)
        continue
    full_cdsl_code = wrap_cdsl(f"RESULT_{i}", cdsl_code)
    # print("tree", tree)
    # print("cdsl_code", cdsl_code)
    # TODO: add encoding etc.!
    if WRITE_GEN:
        if WRITE_GEN_FMT & ExportFormat.CDSL:
            with open(OUT / f"result{i}.core_desc", "w") as f:
                f.write(full_cdsl_code)

# if len(duplicate_counts) > 0:
#     print()
#     print("Duplicates:")
#     for orig, dups in duplicate_counts.items():
#         print(f"result{orig}:\t", dups)

# subs_df["Iso"] = subs_df["result"].apply(lambda x: x in isos)
subs_df["Label"] = "Selected"
subs_df.loc[list(io_isos), "Label"] = "Iso"
subs_df.loc[list(filtered_io), "Label"] = "Filtered (I/O)"
subs_df.loc[list(filtered_complex), "Label"] = "Filtered (Complex)"
subs_df.loc[list(invalid), "Label"] = "Invalid"
subs_df.loc[list(errs), "Label"] = "Error"
# print("subs_df")
# print(subs_df)
pie_df = subs_df.value_counts("Label").rename_axis("Label").reset_index(name="Count")
# print("pie_df")
# print(pie_df)
fig = px.pie(pie_df, values="Count", names="Label", title="Candidates")
fig.update_traces(hoverinfo="label+percent", textinfo="value")
# fig.show()
if WRITE_PIE:
    if WRITE_PIE_FMT & ExportFormat.PDF:
        fig.write_image(OUT / "pie.pdf")
    if WRITE_PIE_FMT & ExportFormat.PNG:
        fig.write_image(OUT / "pie.png")
    if WRITE_PIE_FMT & ExportFormat.CSV:
        pie_df.to_csv(OUT / "pie.csv")
if WRITE_DF:
    if WRITE_DF_FMT & ExportFormat.CSV:
        subs_df.to_csv(OUT / "subs.csv")
    elif WRITE_DF_FMT & ExportFormat.PKL:
        subs_df.to_pickle(OUT / "subs.pkl")
