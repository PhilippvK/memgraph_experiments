import pickle
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import plotly.express as px

# import matplotlib.pyplot as plt

# from anytree import RenderTree
# from anytree.iterators import AbstractIter

from .enums import ExportFormat, ExportFilter, InstrPredicate
from .memgraph import connect_memgraph, run_query
from .cdsl_utils import wrap_cdsl
from .mir_utils import gen_mir_func
from .graph_utils import graph_to_file, calc_inputs, calc_outputs, memgraph_to_nx
from .tree import gen_tree, gen_flat_code
from .queries import generate_func_query, generate_candidates_query
from .pred import check_predicates, detect_predicates
from .timing import MeasureTime

logger = logging.getLogger("main")


# TODO: rename result to sub & gen
# TODO: actually implement filters
FUNC_FMT_DEFAULT = ExportFormat.DOT
FUNC_FLT_DEFAULT = ExportFilter.SELECTED
# SUB_FMT_DEFAULT = ExportFormat.DOT  # | ExportFormat.PDF | ExportFormat.PNG
SUB_FMT_DEFAULT = ExportFormat.DOT | ExportFormat.PDF  # | ExportFormat.PNG | ExportFormat.PKL
SUB_FLT_DEFAULT = ExportFilter.SELECTED
IO_SUB_FMT_DEFAULT = ExportFormat.DOT | ExportFormat.PDF | ExportFormat.PNG
IO_SUB_FLT_DEFAULT = ExportFilter.SELECTED
GEN_FMT_DEFAULT = ExportFormat.CDSL | ExportFormat.MIR | ExportFormat.TXT
GEN_FLT_DEFAULT = ExportFilter.SELECTED
PIE_FMT_DEFAULT = ExportFormat.PDF | ExportFormat.CSV
PIE_FLT_DEFAULT = ExportFilter.ALL
DF_FMT_DEFAULT = ExportFormat.CSV
DF_FLT_DEFAULT = ExportFilter.SELECTED
INSTR_PREDICATES_DEFAULT = InstrPredicate.ALL
IGNORE_NAMES_DEFAULT = ["PHI", "COPY", "PseudoCALLIndirect", "PseudoLGA", "Select_GPR_Using_CC_GPR"]
IGNORE_OP_TYPES_DEFAULT = ["input", "constant"]


def handle_cmdline():
    # TODO: add help messages
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--log", default="info", choices=["critical", "error", "warning", "info", "debug"], help="TODO")
    parser.add_argument("--times", action="store_true", help="TODO")
    parser.add_argument("--progress", action="store_true", help="TODO")
    parser.add_argument("--host", default="localhost", help="TODO")
    parser.add_argument("--port", type=int, default=7687, help="TODO")
    parser.add_argument("--session", default="default", help="TODO")
    parser.add_argument("--limit-results", type=int, default=None, help="TODO")
    parser.add_argument("--min-inputs", type=int, default=0, help="TODO")
    parser.add_argument("--max-inputs", type=int, default=3, help="TODO")
    parser.add_argument("--min-outputs", type=int, default=0, help="TODO")
    parser.add_argument("--max-outputs", type=int, default=2, help="TODO")
    parser.add_argument("--max-nodes", type=int, default=5, help="TODO")
    parser.add_argument("--min-nodes", type=int, default=1, help="TODO")
    parser.add_argument("--min-path-length", type=int, default=1, help="TODO")
    parser.add_argument("--max-path-length", type=int, default=3, help="TODO")
    parser.add_argument("--max-path-width", type=int, default=2, help="TODO")
    parser.add_argument("--instr-predicates", type=int, default=INSTR_PREDICATES_DEFAULT, help="TODO")
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
    parser.add_argument("--write-pie-flt", type=int, default=PIE_FLT_DEFAULT, help="TODO")
    parser.add_argument("--write-df", action="store_true", help="TODO")
    parser.add_argument("--write-df-fmt", type=int, default=DF_FMT_DEFAULT, help="TODO")
    parser.add_argument("--write-df-flt", type=int, default=DF_FLT_DEFAULT, help="TODO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()))
    logging.getLogger("neo4j.io").setLevel(logging.INFO)
    logging.getLogger("neo4j.pool").setLevel(logging.INFO)
    return args


args = handle_cmdline()
PROGRESS = args.progress
TIMES = args.times
MIN_INPUTS = args.min_inputs
MAX_INPUTS = args.max_inputs
MIN_OUTPUTS = args.min_outputs
MAX_OUTPUTS = args.max_outputs
MAX_NODES = args.max_nodes
MIN_NODES = args.min_nodes
XLEN = args.xlen
OUT = Path(args.output_dir)
SESSION = args.session
HOST = args.host
PORT = args.port
FUNC = args.function
BB = args.basic_block
LIMIT_RESULTS = args.limit_results
MIN_PATH_LEN = args.min_path_length
MAX_PATH_LEN = args.max_path_length
MAX_PATH_WIDTH = args.max_path_width
INSTR_PREDICATES = args.instr_predicates
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
WRITE_PIE_FLT = args.write_pie_flt
WRITE_DF = args.write_df
WRITE_DF_FMT = args.write_df_fmt
WRITE_DF_FLT = args.write_df_flt

with MeasureTime("Settings Validation", verbose=TIMES):
    logger.info("Validating settings...")
    assert OUT.is_dir(), f"OUT ({OUT}) is not a directory"
    assert FUNC is not None
    if not IGNORE_CONST_INPUTS:
        raise NotImplementedError("!IGNORE_CONST_INPUTS")


logger.info("Running queries...")
with MeasureTime("Connect to DB", verbose=TIMES):
    driver = connect_memgraph(HOST, PORT, user="", password="")

with MeasureTime("Query func from DB", verbose=TIMES):
    query_func = generate_func_query(SESSION, FUNC)
    func_results = run_query(driver, query_func)

with MeasureTime("Query candidates from DB", verbose=TIMES):
    query = generate_candidates_query(
        SESSION,
        FUNC,
        BB,
        MIN_PATH_LEN,
        MAX_PATH_LEN,
        MAX_PATH_WIDTH,
        IGNORE_NAMES,
        IGNORE_OP_TYPES,
        limit=LIMIT_RESULTS,
    )
    results = run_query(driver, query)

# TODO: move to helper func
# TODO: print number of results
with MeasureTime("Conversion to NX (func)", verbose=TIMES):
    logger.info("Converting func graph to NX...")
    GF = memgraph_to_nx(func_results)
    # print("GF", GF)


if WRITE_FUNC:
    with MeasureTime("Dumping GF graph", verbose=TIMES):
        logger.info("Exporting GF graph...")
        if WRITE_FUNC_FMT & ExportFormat.DOT:
            graph_to_file(GF, OUT / "func.dot")
        if WRITE_FUNC_FMT & ExportFormat.PDF:
            graph_to_file(GF, OUT / "func.pdf")
        if WRITE_FUNC_FMT & ExportFormat.PNG:
            graph_to_file(GF, OUT / "func.png")


# TODO: move to helper and share code
with MeasureTime("Conversion to NX (candidates)", verbose=TIMES):
    G = memgraph_to_nx(results)


with MeasureTime("Subgraph Generation", verbose=TIMES):
    logger.info("Generating subgraphs...")
    subs = []
    for i, result in enumerate(tqdm(results, disable=not PROGRESS)):
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
            nodes_ |= {int(n.element_id) for n in nodes__}
        # print("nodes_", nodes_)
        G_ = G.subgraph(nodes_)
        # G_ = nx.subgraph_view(G, filter_node=lambda x: x in nodes_)
        # print("G_", G_)
        count = subs.count(G_)
        if count > 0:
            pass
        subs.append(G_)


# for i, result in enumerate(results):
#     print("result", result, i, dir(result), result.data())

with MeasureTime("Relabeling", verbose=TIMES):
    logger.info("Relabeling nodes...")
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
filtered_simple = set()
filtered_predicates = set()
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
with MeasureTime("I/O Analysis", verbose=TIMES):
    logger.info("Collecting I/O details...")
    for i, sub in enumerate(tqdm(subs, disable=not PROGRESS)):
        # if i in isos:
        #     continue
        # i = 3
        # sub = subs[i]
        # print("topo", topo)
        # print("===========================")
        # print("i, sub", i, sub)
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
        for inp in inputs:
            edges = list(io_sub.in_edges(inp))
            io_sub.remove_edges_from(edges)
        j = 0
        for inp in inputs:
            # TODO: physreg?
            if io_sub.nodes[inp]["label"] == "Const":
                io_sub.nodes[inp]["xlabel"] = "CONST"
                io_sub.nodes[inp]["fillcolor"] = "lightgray"
                io_sub.nodes[inp]["style"] = "filled"
                io_sub.nodes[inp]["shape"] = "box"
                io_sub.nodes[inp]["label"] = io_sub.nodes[inp]["properties"]["inst"][:-1]
            else:
                io_sub.nodes[inp]["xlabel"] = "IN"
                io_sub.nodes[inp]["label"] = f"src{j}"
                io_sub.nodes[inp]["fillcolor"] = "darkgray"
                io_sub.nodes[inp]["style"] = "filled"
                io_sub.nodes[inp]["shape"] = "box"
                j += 1
        # TODO: add out nodes to io_sub?
        # print("io_sub", io_sub)
        io_subs.append(io_sub)

with MeasureTime("Isomorphism Check", verbose=TIMES):
    logger.info("Checking isomorphism...")
    # print("io_subs", [str(x) for x in io_subs], len(io_subs))
    io_isos = set()
    for i, io_sub in enumerate(tqdm(io_subs, disable=not PROGRESS)):
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


with MeasureTime("Predicate Detection", verbose=TIMES):
    logger.info("Detecting Predicates...")
    subs_df["Predicates"] = InstrPredicate.NONE
    for i, sub in enumerate(tqdm(subs, disable=not PROGRESS)):
        if i in io_isos:
            continue
        pred = detect_predicates(sub)
        subs_df.loc[i, "Predicates"] = pred


with MeasureTime("Filtering subgraphs", verbose=TIMES):
    logger.info("Filtering subgraphs...")
    for i, sub in enumerate(tqdm(subs, disable=not PROGRESS)):
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
        elif MIN_INPUTS <= num_inputs_noconst <= MAX_INPUTS and MIN_OUTPUTS <= num_outputs <= MAX_OUTPUTS:
            pred = subs_df.loc[i, "Predicates"]
            if num_nodes > MAX_NODES:
                filtered_complex.add(i)
            elif num_nodes < MIN_NODES:
                filtered_simple.add(i)
            elif not check_predicates(pred, INSTR_PREDICATES):
                # TODO: add predicates details to df in prerequisite step
                filtered_predicates.add(i)
        else:
            filtered_io.add(i)

with MeasureTime("Generation", verbose=TIMES):
    logger.info("Generation...")
    for i, sub in enumerate(tqdm(subs, disable=not PROGRESS)):
        if i in io_isos or i in filtered_io or i in filtered_complex or i in filtered_predicates:
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
        if WRITE_GEN:
            if WRITE_GEN_FMT & ExportFormat.MIR:
                logger.info("Generating MIR")
                j = 0  # reg's
                # j_ = 0  # imm's
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
                """ ---
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
                with open(OUT / f"result{i}.mir", "w") as f:
                    f.write(mir_code)
            # TODO: split cdsl from gen_tree!
            xtrees = None
            if WRITE_GEN_FMT & ExportFormat.CDSL:
                logger.info("Generating CDSL")
                # TODO: tree to disk? (ExportFormat.TREE)
                try:
                    tree, xtrees, cdsl_code = gen_tree(GF, sub, inputs, outputs, xlen=XLEN)
                except AssertionError as e:
                    logger.exception(e)
                    errs.add(i)
                    continue
                # print("tree", tree)
                # print("cdsl_code", cdsl_code)
                # TODO: add encoding etc.!
                full_cdsl_code = wrap_cdsl(f"RESULT_{i}", cdsl_code)
                with open(OUT / f"result{i}.core_desc", "w") as f:
                    f.write(full_cdsl_code)
            if WRITE_GEN_FMT & ExportFormat.TXT:
                logger.info("Generating FLAT")
                assert xtrees is not None, ""
                header = f"Inputs (with imm): {num_inputs}, Inputs (without imm): {num_inputs_noconst}, Outputs: {num_outputs}"
                flat_code = gen_flat_code(xtrees, desc=desc)
                with open(OUT / f"result{i}.txt", "w") as f:
                    f.write(flat_code)

# TODO: loop multiple times (tree -> MIR -> CDSL -> FLAT) not interleaved

# if len(duplicate_counts) > 0:
#     print()
#     print("Duplicates:")
#     for orig, dups in duplicate_counts.items():
#         print(f"result{orig}:\t", dups)

with MeasureTime("Finish DF", verbose=TIMES):
    logger.info("Finalizing DataFrame...")

    # subs_df["Iso"] = subs_df["result"].apply(lambda x: x in isos)
    subs_df["Status"] = ExportFilter.SELECTED
    subs_df.loc[list(io_isos), "Status"] = ExportFilter.ISO
    subs_df.loc[list(filtered_io), "Status"] = ExportFilter.FILTERED_IO
    subs_df.loc[list(filtered_complex), "Status"] = ExportFilter.FILTERED_COMPLEX
    subs_df.loc[list(filtered_simple), "Status"] = ExportFilter.FILTERED_SIMPLE
    subs_df.loc[list(filtered_predicates), "Status"] = ExportFilter.FILTERED_PRED
    subs_df.loc[list(invalid), "Status"] = ExportFilter.INVALID
    subs_df.loc[list(errs), "Status"] = ExportFilter.ERROR
    subs_df["Status (str)"] = subs_df["Status"].apply(lambda x: str(ExportFilter(x)))
    subs_df["Predicates (str)"] = subs_df["Predicates"].apply(lambda x: str(InstrPredicate(x)))
    # print("subs_df")
    # print(subs_df)


if WRITE_SUB:
    with MeasureTime("Subgraph Export", verbose=TIMES):
        logger.info("Exporting subgraphs...")
        filtered_subs_df = subs_df[(subs_df["Status"] & WRITE_SUB_FLT) > 0].copy()
        for i, sub in enumerate(tqdm(subs, disable=not PROGRESS)):
            if i not in filtered_subs_df.index:
                continue
            if WRITE_SUB_FMT & ExportFormat.DOT:
                graph_to_file(G_, OUT / f"sub{i}.dot")
            if WRITE_SUB_FMT & ExportFormat.PDF:
                graph_to_file(G_, OUT / f"sub{i}.pdf")
            if WRITE_SUB_FMT & ExportFormat.PNG:
                graph_to_file(G_, OUT / f"sub{i}.png")
            if WRITE_SUB_FMT & ExportFormat.PKL:
                with open(OUT / f"sub{i}.pkl", "wb") as f:
                    pickle.dump(G_.copy(), f)


if WRITE_IO_SUB:
    with MeasureTime("Dumping I/O Subgraphs", verbose=TIMES):
        logger.info("Exporting I/O subgraphs...")
        filtered_subs_df = subs_df[(subs_df["Status"] & WRITE_IO_SUB_FLT) > 0].copy()
        for i, io_sub in enumerate(tqdm(io_subs, disable=not PROGRESS)):
            if i in io_isos or i not in filtered_subs_df.index:
                continue
            if WRITE_IO_SUB_FMT & ExportFormat.DOT:
                graph_to_file(io_sub, OUT / f"io_sub{i}.dot")
            if WRITE_IO_SUB_FMT & ExportFormat.PDF:
                graph_to_file(io_sub, OUT / f"io_sub{i}.pdf")
            if WRITE_IO_SUB_FMT & ExportFormat.PNG:
                graph_to_file(io_sub, OUT / f"io_sub{i}.png")


if WRITE_DF:
    with MeasureTime("Dump DF", verbose=TIMES):
        filtered_subs_df = subs_df[(subs_df["Status"] & WRITE_DF_FLT) > 0].copy()
        logger.info("Exporting DataFrame...")
        if WRITE_DF_FMT & ExportFormat.CSV:
            filtered_subs_df.to_csv(OUT / "subs.csv")
        elif WRITE_DF_FMT & ExportFormat.PKL:
            filtered_subs_df.to_pickle(OUT / "subs.pkl")


if WRITE_PIE:
    with MeasureTime("Generate Pie", verbose=TIMES):
        logger.info("Generating PieChart...")
        filtered_subs_df = subs_df[(subs_df["Status"] & WRITE_PIE_FLT) > 0].copy()

        def helper(x):
            if x & ExportFilter.SELECTED:
                return "Selected"
            if x & ExportFilter.ISO:
                return "Iso"
            if x & ExportFilter.FILTERED_IO:
                return "Filtered (I/O)"
            if x & ExportFilter.FILTERED_COMPLEX:
                return "Filtered (Complex)"
            if x & ExportFilter.FILTERED_SIMPLE:
                return "Filtered (Simple)"
            if x & ExportFilter.FILTERED_PRED:
                return "Filtered (Pred)"
            if x & ExportFilter.INVALID:
                return "Invalid"
            if x & ExportFilter.ERROR:
                return "ERROR"
            return "Unknown"

        filtered_subs_df["Label"] = filtered_subs_df["Status"].apply(helper)
        pie_df = filtered_subs_df.value_counts("Label").rename_axis("Label").reset_index(name="Count")
        # print("pie_df")
        # print(pie_df)
        fig = px.pie(pie_df, values="Count", names="Label", title="Candidates")
        fig.update_traces(hoverinfo="label+percent", textinfo="value")
        # fig.show()
    with MeasureTime("Dump Pie", verbose=TIMES):
        logger.info("Exporting PieChart...")
        if WRITE_PIE_FMT & ExportFormat.PDF:
            fig.write_image(OUT / "pie.pdf")
        if WRITE_PIE_FMT & ExportFormat.PNG:
            fig.write_image(OUT / "pie.png")
        if WRITE_PIE_FMT & ExportFormat.CSV:
            pie_df.to_csv(OUT / "pie.csv")

if TIMES:
    print(MeasureTime.summary())
    MeasureTime.write_csv(OUT / "times.csv")
