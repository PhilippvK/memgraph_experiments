import pickle
import logging
import argparse
from math import ceil, log2
from pathlib import Path
from datetime import datetime
from typing import Dict
from collections import defaultdict

# import concurrent.futures

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from anytree import RenderTree, AnyNode

# import matplotlib.pyplot as plt

# from anytree.iterators import AbstractIter

from .enums import ExportFormat, ExportFilter, InstrPredicate, CDFGStage, parse_enum_intflag
from .memgraph import connect_memgraph, run_query
from .iso import calc_io_isos
from .llvm_utils import parse_llvm_const_str
from .graph_utils import (
    graph_to_file,
    calc_inputs,
    calc_outputs,
    memgraph_to_nx,
    get_instructions,
    calc_weights,
    calc_weights_iso,
)
from .queries import generate_func_query, generate_candidates_query
from .pred import check_predicates, detect_predicates
from .timing import MeasureTime
from .pie import generate_pie_chart, generate_pie2_chart
from .index import write_index_file
from .gen.tree import generate_tree
from .gen.cdsl import generate_cdsl
from .gen.mir import generate_mir
from .gen.flat import generate_flat_code
from .gen.desc import generate_desc

logger = logging.getLogger("main")


# TODO: rename result to sub & gen
# TODO: actually implement filters
FUNC_FMT_DEFAULT = ExportFormat.DOT | ExportFormat.PKL
# FUNC_FLT_DEFAULT = ExportFilter.SELECTED
# SUB_FMT_DEFAULT = ExportFormat.DOT  # | ExportFormat.PDF | ExportFormat.PNG
# SUB_FMT_DEFAULT = ExportFormat.DOT | ExportFormat.PKL  # | ExportFormat.PNG | ExportFormat.PDF
SUB_FMT_DEFAULT = ExportFormat.DOT | ExportFormat.PKL | ExportFormat.PDF
SUB_FLT_DEFAULT = ExportFilter.SELECTED
# SUB_FLT_DEFAULT = ExportFilter.ALL
# IO_SUB_FMT_DEFAULT = ExportFormat.DOT | ExportFormat.PKL  # | ExportFormat.PNG | ExportFormat.PDF
IO_SUB_FMT_DEFAULT = ExportFormat.DOT | ExportFormat.PKL | ExportFormat.PDF
IO_SUB_FLT_DEFAULT = ExportFilter.SELECTED
# TODO: move Tree tro pre-gen
TREE_FMT_DEFAULT = ExportFormat.TXT | ExportFormat.PKL
TREE_FLT_DEFAULT = ExportFilter.SELECTED
# GEN_FMT_DEFAULT = ExportFormat.CDSL | ExportFormat.MIR | ExportFormat.FLAT
GEN_FMT_DEFAULT = ExportFormat.FLAT | ExportFormat.CDSL
GEN_FLT_DEFAULT = ExportFilter.SELECTED
PIE_FMT_DEFAULT = ExportFormat.PDF | ExportFormat.CSV
PIE_FLT_DEFAULT = ExportFilter.ALL
DF_FMT_DEFAULT = ExportFormat.CSV | ExportFormat.PKL
DF_FLT_DEFAULT = ExportFilter.ALL
INDEX_FMT_DEFAULT = ExportFormat.YAML
INDEX_FLT_DEFAULT = ExportFilter.SELECTED
INSTR_PREDICATES_DEFAULT = InstrPredicate.ALL
IGNORE_NAMES_DEFAULT = ["G_PHI", "PHI", "COPY", "PseudoCALLIndirect", "PseudoLGA", "Select_GPR_Using_CC_GPR"]
IGNORE_OP_TYPES_DEFAULT = ["input", "constant"]
ALLOWED_ENC_SIZES_DEFAULT = [32]
STAGE_DEFAULT = CDFGStage.STAGE_3
MAX_ENC_FOOTPRINT_DEFAULT = 1.0
MAX_ENC_WEIGHT_DEFAULT = 1.0
MIN_ENC_BITS_LEFT_DEFAULT = 5
MIN_ISO_WEIGHT_DEFAULT = 0.05
MAX_LOADS_DEFAULT = 1
MAX_STORES_DEFAULT = 1
MAX_MEMS_DEFAULT = 1
MAX_BRANCHES_DEFAULT = 1


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
    # TODO: --max-constants
    parser.add_argument("--max-nodes", type=int, default=5, help="TODO")
    parser.add_argument("--max-enc-footprint", type=float, default=MAX_ENC_FOOTPRINT_DEFAULT, help="TODO")
    parser.add_argument("--max-enc-weight", type=float, default=MAX_ENC_WEIGHT_DEFAULT, help="TODO")
    parser.add_argument("--min-enc-bits-left", type=float, default=MIN_ENC_BITS_LEFT_DEFAULT, help="TODO")
    parser.add_argument("--min-nodes", type=int, default=1, help="TODO")
    parser.add_argument("--min-path-length", type=int, default=1, help="TODO")
    parser.add_argument("--max-path-length", type=int, default=3, help="TODO")
    parser.add_argument("--max-path-width", type=int, default=2, help="TODO")
    parser.add_argument("--instr-predicates", type=int, default=INSTR_PREDICATES_DEFAULT, help="TODO")
    parser.add_argument("--function", "--func", default=None, help="TODO")
    parser.add_argument("--basic-block", "--bb", default=None, help="TODO")
    parser.add_argument("--stage", type=int, default=STAGE_DEFAULT, help="TODO")
    parser.add_argument("--ignore-names", default=",".join(IGNORE_NAMES_DEFAULT), help="TODO")
    parser.add_argument("--ignore-op-types", default=",".join(IGNORE_OP_TYPES_DEFAULT), help="TODO")
    parser.add_argument("--ignore-const-inputs", action="store_true", help="TODO")
    parser.add_argument("--xlen", type=int, default=64, help="TODO")
    parser.add_argument("--output-dir", "-o", default="./out", help="TODO")
    parser.add_argument("--write-func", action="store_true", help="TODO")
    parser.add_argument("--write-func-fmt", type=int, default=FUNC_FMT_DEFAULT, help="TODO")
    # parser.add_argument("--write-func-flt", type=int, default=FUNC_FLT_DEFAULT, help="TODO")
    parser.add_argument("--write-sub", action="store_true", help="TODO")
    parser.add_argument("--write-sub-fmt", type=int, default=SUB_FMT_DEFAULT, help="TODO")
    parser.add_argument("--write-sub-flt", type=int, default=SUB_FLT_DEFAULT, help="TODO")
    parser.add_argument("--write-io-sub", action="store_true", help="TODO")
    parser.add_argument("--write-io-sub-fmt", type=int, default=IO_SUB_FMT_DEFAULT, help="TODO")
    parser.add_argument("--write-io-sub-flt", type=int, default=IO_SUB_FLT_DEFAULT, help="TODO")
    parser.add_argument("--write-tree", action="store_true", help="TODO")
    parser.add_argument("--write-tree-fmt", type=int, default=TREE_FMT_DEFAULT, help="TODO")
    parser.add_argument("--write-tree-flt", type=int, default=TREE_FLT_DEFAULT, help="TODO")
    parser.add_argument("--write-gen", action="store_true", help="TODO")
    parser.add_argument("--write-gen-fmt", type=int, default=GEN_FMT_DEFAULT, help="TODO")
    parser.add_argument("--write-gen-flt", type=int, default=GEN_FLT_DEFAULT, help="TODO")
    parser.add_argument("--write-pie", action="store_true", help="TODO")
    parser.add_argument("--write-pie-fmt", type=int, default=PIE_FMT_DEFAULT, help="TODO")
    parser.add_argument("--write-pie-flt", type=int, default=PIE_FLT_DEFAULT, help="TODO")
    parser.add_argument("--write-df", action="store_true", help="TODO")
    parser.add_argument("--write-df-fmt", type=int, default=DF_FMT_DEFAULT, help="TODO")
    parser.add_argument("--write-df-flt", type=int, default=DF_FLT_DEFAULT, help="TODO")
    parser.add_argument("--write-index", action="store_true", help="TODO")
    parser.add_argument("--write-index-fmt", type=int, default=INDEX_FMT_DEFAULT, help="TODO")
    parser.add_argument("--write-index-flt", type=int, default=INDEX_FLT_DEFAULT, help="TODO")
    parser.add_argument("--write-queries", action="store_true", help="TODO")
    parser.add_argument("--allowed-enc-sizes", type=int, nargs="+", default=ALLOWED_ENC_SIZES_DEFAULT, help="TODO")
    parser.add_argument("--min-iso-weight", type=float, default=MIN_ISO_WEIGHT_DEFAULT, help="TODO")
    parser.add_argument("--max-loads", type=int, default=MAX_LOADS_DEFAULT, help="TODO")
    parser.add_argument("--max-stores", type=int, default=MAX_STORES_DEFAULT, help="TODO")
    parser.add_argument("--max-mems", type=int, default=MAX_MEMS_DEFAULT, help="TODO")
    parser.add_argument("--max-branches", type=int, default=MAX_BRANCHES_DEFAULT, help="TODO")
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
# TODO: MIN_FREQ, MIN_WEIGHT, MAX_INSTRS, MAX_UNIQUE_INSTRS
XLEN = args.xlen
OUT = Path(args.output_dir).resolve()
SESSION = args.session
HOST = args.host
PORT = args.port
FUNC = args.function
BB = args.basic_block
STAGE = CDFGStage(args.stage)
LIMIT_RESULTS = args.limit_results
MIN_PATH_LEN = args.min_path_length
MAX_PATH_LEN = args.max_path_length
MAX_PATH_WIDTH = args.max_path_width
INSTR_PREDICATES = parse_enum_intflag(args.instr_predicates, InstrPredicate)
IGNORE_NAMES = args.ignore_names.split(",")
IGNORE_OP_TYPES = args.ignore_op_types.split(",")
IGNORE_CONST_INPUTS = args.ignore_const_inputs
WRITE_FUNC = args.write_func
WRITE_FUNC_FMT = args.write_func_fmt
# WRITE_FUNC_FLT = args.write_func_flt
WRITE_SUB = args.write_sub
WRITE_SUB_FMT = parse_enum_intflag(args.write_sub_fmt, ExportFormat)
WRITE_SUB_FLT = parse_enum_intflag(args.write_sub_flt, ExportFilter)
WRITE_IO_SUB = args.write_io_sub
WRITE_IO_SUB_FMT = args.write_io_sub_fmt
WRITE_IO_SUB_FLT = args.write_io_sub_flt
WRITE_TREE = args.write_tree
WRITE_TREE_FMT = args.write_tree_fmt
WRITE_TREE_FLT = args.write_tree_flt
WRITE_GEN = args.write_gen
WRITE_GEN_FMT = args.write_gen_fmt
WRITE_GEN_FLT = args.write_gen_flt
WRITE_PIE = args.write_pie
WRITE_PIE_FMT = args.write_pie_fmt
WRITE_PIE_FLT = args.write_pie_flt
WRITE_DF = args.write_df
WRITE_DF_FMT = args.write_df_fmt
WRITE_DF_FLT = args.write_df_flt
WRITE_INDEX = args.write_index
WRITE_INDEX_FMT = args.write_index_fmt
WRITE_INDEX_FLT = args.write_index_flt
WRITE_QUERIES = args.write_queries
ALLOWED_ENC_SIZES = args.allowed_enc_sizes
MAX_ENC_FOOTPRINT = args.max_enc_footprint
MAX_ENC_WEIGHT = args.max_enc_weight
MIN_ENC_BITS_LEFT = args.min_enc_bits_left
MIN_ISO_WEIGHT = args.min_iso_weight
MAX_LOADS = args.max_loads
MAX_STORES = args.max_stores
MAX_MEMS = args.max_mems
MAX_BRANCHES = args.max_branches

with MeasureTime("Settings Validation", verbose=TIMES):
    logger.info("Validating settings...")
    assert OUT.is_dir(), f"OUT ({OUT}) is not a directory"
    assert FUNC is not None
    assert not IGNORE_CONST_INPUTS, "DEPRECTAED!"


logger.info("Running queries...")
with MeasureTime("Connect to DB", verbose=TIMES):
    driver = connect_memgraph(HOST, PORT, user="", password="")

with MeasureTime("Query func from DB", verbose=TIMES):
    query_func = generate_func_query(SESSION, FUNC, stage=STAGE)
    func_results = run_query(driver, query_func)
    if WRITE_QUERIES:
        logger.info("Exporting queries...")
        with open(OUT / "query_func.cypher", "w") as f:
            f.write(query_func)


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
        stage=STAGE,
        limit=LIMIT_RESULTS,
    )
    results = run_query(driver, query)
    if WRITE_QUERIES:
        logger.info("Exporting queries...")
        with open(OUT / "query_candidates.cypher", "w") as f:
            f.write(query)

# TODO: move to helper func
# TODO: print number of results
with MeasureTime("Conversion to NX (func)", verbose=TIMES):
    logger.info("Converting func graph to NX...")
    GF = memgraph_to_nx(func_results)
    # print("GF", GF)


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

index_artifacts = defaultdict(dict)


if WRITE_FUNC:
    with MeasureTime("Dumping GF graph", verbose=TIMES):
        logger.info("Exporting GF graph...")
        if WRITE_FUNC_FMT & ExportFormat.DOT:
            graph_to_file(GF, OUT / "func.dot")
        if WRITE_FUNC_FMT & ExportFormat.PDF:
            graph_to_file(GF, OUT / "func.pdf")
        if WRITE_FUNC_FMT & ExportFormat.PNG:
            graph_to_file(GF, OUT / "func.png")
        if WRITE_FUNC_FMT & ExportFormat.PKL:
            with open(OUT / "func.pkl", "wb") as f:
                pickle.dump(GF.copy(), f)
            index_artifacts[None]["func"] = OUT / "func.pkl"


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


global_df = pd.DataFrame()

now = datetime.now()
ts = now.strftime("%Y%m%dT%H%M%S")
global_df["timestamp"] = [ts]
global_df["min_inputs"] = [MIN_INPUTS]
global_df["max_inputs"] = [MAX_INPUTS]
global_df["min_outputs"] = [MIN_OUTPUTS]
global_df["max_outputs"] = [MAX_OUTPUTS]
global_df["min_nodes"] = [MIN_NODES]
global_df["max_nodes"] = [MAX_NODES]
global_df["xlen"] = [XLEN]
global_df["session"] = [SESSION]
global_df["func"] = [FUNC]
global_df["bb"] = [BB]
global_df["stage"] = [STAGE]
global_df["limit_results"] = [LIMIT_RESULTS]
global_df["min_path_len"] = [MIN_PATH_LEN]
global_df["max_path_len"] = [MAX_PATH_LEN]
global_df["max_path_width"] = [MAX_PATH_WIDTH]
global_df["instr_predicates"] = [INSTR_PREDICATES]
global_df["ignore_names"] = [IGNORE_NAMES]
global_df["ignore_op_types"] = [IGNORE_OP_TYPES]
global_df["allowed_enc_sizes"] = [ALLOWED_ENC_SIZES]
global_df["max_enc_footprint"] = [MAX_ENC_FOOTPRINT]
global_df["max_enc_weight"] = [MAX_ENC_WEIGHT]
global_df["min_enc_bits_left"] = [MIN_ENC_BITS_LEFT]
global_df["min_iso_weight"] = [MIN_ISO_WEIGHT]
global_df["max_loads"] = [MAX_LOADS]
global_df["max_stores"] = [MAX_STORES]
global_df["max_mems"] = [MAX_MEMS]
global_df["max_branches"] = [MAX_BRANCHES]
# TODO: MIN_FREQ, MAX_INSTRS, MAX_UNIQUE_INSTRS

subs_df = pd.DataFrame({"result": list(range(len(subs)))})
subs_df["DateTime"] = ts
subs_df["Parent"] = np.nan  # used to find the original sub for a variation
subs_df["Variations"] = np.nan  # used to specify applied variations for Children
subs_df["SubHash"] = np.nan
subs_df["IOSubHash"] = np.nan
subs_df["Isos"] = [np.array([])] * len(subs_df)
subs_df["#Isos"] = np.nan
subs_df["IsosNO"] = [np.array([])] * len(subs_df)
subs_df["#IsosNO"] = np.nan
subs_df["Nodes"] = [np.array([])] * len(subs_df)
# subs_df["#Nodes"] = np.nan
subs_df["InputNodes"] = [np.array([])] * len(subs_df)
subs_df["#InputNodes"] = np.nan
subs_df["InputNames"] = [np.array([])] * len(subs_df)
subs_df["ConstantNodes"] = [np.array([])] * len(subs_df)
subs_df["ConstantValues"] = [np.array([])] * len(subs_df)
subs_df["ConstantSigns"] = [np.array([])] * len(subs_df)
subs_df["ConstantMinBits"] = [np.array([])] * len(subs_df)
subs_df["#ConstantNodes"] = np.nan  # TODO: analyze constants
subs_df["OutputNodes"] = [np.array([])] * len(subs_df)
subs_df["OutputNames"] = [np.array([])] * len(subs_df)
subs_df["#OutputNodes"] = np.nan
subs_df["#Operands"] = np.nan
subs_df["OperandNames"] = [np.array([])] * len(subs_df)
subs_df["OperandNodes"] = [np.array([])] * len(subs_df)
subs_df["OperandDirs"] = [np.array([])] * len(subs_df)
subs_df["OperandTypes"] = [np.array([])] * len(subs_df)
subs_df["OperandRegClasses"] = [np.array([])] * len(subs_df)  # TODO
subs_df["OperandEncBits"] = [np.array([])] * len(subs_df)
subs_df["OperandEncBitsSum"] = np.nan
subs_df["Constraints"] = [np.array([])] * len(subs_df)
subs_df["Instrs"] = [np.array([])] * len(subs_df)
subs_df["#Instrs"] = np.nan
subs_df["#Loads"] = np.nan
subs_df["#Stores"] = np.nan
subs_df["#Mems"] = np.nan
subs_df["#Terminators"] = np.nan
subs_df["#Branches"] = np.nan
subs_df["LoadNodes"] = [np.array([])] * len(subs_df)
subs_df["StoreNodes"] = [np.array([])] * len(subs_df)
subs_df["TerminatorNodes"] = [np.array([])] * len(subs_df)
subs_df["BranchNodes"] = [np.array([])] * len(subs_df)
subs_df["UniqueInstrs"] = [np.array([])] * len(subs_df)
subs_df["#UniqueInstrs"] = np.nan
for enc_size in ALLOWED_ENC_SIZES:
    subs_df[f"EncodingBitsLeft ({enc_size} bits)"] = np.nan
    subs_df[f"EncodingWeight ({enc_size} bits)"] = np.nan
    subs_df[f"EncodingFootprint ({enc_size} bits)"] = np.nan
subs_df["Weight"] = np.nan
subs_df["Freq"] = np.nan
subs_df["IsoNodes"] = [np.array([])] * len(subs_df)
subs_df["IsoWeight"] = np.nan
subs_df["Status"] = ExportFilter.SELECTED  # TODO: init with UNKNOWN
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
        nodes = sub.nodes
        num_inputs, inputs, num_constants, constants = calc_inputs(GF, sub)
        num_outputs, outputs = calc_outputs(GF, sub)
        # if num_constants > 0:
        #     print("num_inputs", num_inputs)
        #     print("num_constants", num_constants)
        #     print("num_outputs", num_outputs)
        #     input(">")
        instrs = get_instructions(sub)
        unique_instrs = set(instrs)
        total_weight, freq = calc_weights(sub)
        subs_df.at[i, "Nodes"] = set(nodes)
        subs_df.at[i, "InputNodes"] = set(inputs)
        subs_df.loc[i, "#InputNodes"] = num_inputs
        subs_df.at[i, "ConstantNodes"] = set(constants)
        subs_df.loc[i, "#ConstantNodes"] = num_constants
        subs_df.at[i, "OutputNodes"] = set(outputs)
        subs_df.loc[i, "#OutputNodes"] = num_outputs
        subs_df.at[i, "Instrs"] = instrs
        subs_df.loc[i, "#Instrs"] = len(instrs)
        subs_df.at[i, "UniqueInstrs"] = unique_instrs
        subs_df.loc[i, "#UniqueInstrs"] = len(unique_instrs)
        subs_df.loc[i, "Weight"] = total_weight
        subs_df.loc[i, "Freq"] = freq
        # TODO: also sum up weight and freq for isos? (be careful with overlaps!)

        # print("num_inputs", num_inputs)
        # print("num_outputs", num_outputs)
        # print("inputs", [GF.nodes[inp] for inp in inputs])
        # print("outputs", [GF.nodes[outp] for outp in outputs])
        # TODO: copy fine?
        io_sub = GF.subgraph(list(sub.nodes) + inputs + constants).copy()
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


with MeasureTime("SubHash Creation", verbose=TIMES):
    logger.info("Creating SubHashes...")
    # Does not woek for MultiDiGraphs?
    # for i, io_sub in enumerate(tqdm(io_subs, disable=not PROGRESS)):
    #     sub = subs[i]
    #     def add_hash_attr(sub):
    #         for node in sub.nodes:
    #             temp = sub.nodes[node]["label"]
    #             if temp == "Const":
    #                 temp += "-" + sub.nodes[node]["properties"]["inst"]
    #             temp += "-" + str(sub.nodes[node]["properties"].get("alias", None))
    #             print("temp", temp)
    #             sub.nodes[node]["hash_attr"] = temp
    #         for edge in sub.edges:
    #             sub.edges[edge]["hash_attr"] = sub.edges[edge]["properties"]["op_idx"]
    #     add_hash_attr(sub)
    #     add_hash_attr(io_sub)
    #     edge_attr = "hash_attr"
    #     node_attr = "hash_attr"
    #     sub_hash = nx.weisfeiler_lehman_graph_hash(sub, edge_attr=edge_attr, node_attr=node_attr, iterations=3, digest_size=16)
    #     io_sub_hash = nx.weisfeiler_lehman_graph_hash(io_sub, edge_attr=edge_attr, node_attr=node_attr, iterations=3, digest_size=16)
    #     print("sub_hash", sub_hash)
    #     print("io_sub_hash", io_sub_hash)
    #     input("o")

with MeasureTime("Isomorphism Check", verbose=TIMES):
    logger.info("Checking isomorphism...")
    # print("io_subs", [str(x) for x in io_subs], len(io_subs))
    io_isos, sub_io_isos = calc_io_isos(io_subs, progress=PROGRESS)

    # print("subs_df")
    # print(subs_df)
    # print("io_isos", io_isos, len(io_isos))
    subs_df.loc[list(io_isos), "Status"] = ExportFilter.ISO
# print("sub_io_isos", sub_io_isos)
# for key in sorted(sub_io_isos, key=lambda k: len(sub_io_isos[k]), reverse=True):

with MeasureTime("Overlap Check", verbose=TIMES):
    logger.info("Checking overlaps...")
    for key in sub_io_isos:
        iso_nodes = set()
        iso_nodes |= set(subs[key].nodes)
        val = sub_io_isos[key]
        num_isos = len(val)
        non_overlapping = set()

        def check_overlap(x, y):
            # print("check_overlap", x, y)
            intersection = set(x.nodes) & set(y.nodes)
            # print("intersection", intersection)
            return len(intersection) > 0

        for iso in val:
            iso_nodes |= set(subs[iso].nodes)
            # print("iso", iso)
            ol = False
            ol |= check_overlap(subs[iso], subs[key])
            if not ol:
                for iso_ in non_overlapping:
                    # print("iso_", iso_)
                    ol |= check_overlap(subs[iso], subs[key])
                    if ol:
                        break
            if not ol:
                non_overlapping.add(iso)

        # print("num_isos", num_isos)
        # print("non_overlapping", non_overlapping)
        # print("len(non_overlapping)", len(non_overlapping))
        # input("LLL")
        subs_df.at[key, "IsoNodes"] = iso_nodes
        iso_weight, _ = calc_weights_iso(GF, iso_nodes)
        subs_df.loc[key, "IsoWeight"] = iso_weight
        subs_df.loc[key, "#IsosNO"] = len(non_overlapping)
        subs_df.at[key, "IsosNO"] = set(non_overlapping)
        subs_df.loc[key, "#Isos"] = num_isos
        subs_df.at[key, "Isos"] = set(val)
        if num_isos == 0:
            continue
        assert key not in io_isos
        # print(f"{key}: {num_isos}")


with MeasureTime("Predicate Detection", verbose=TIMES):
    logger.info("Detecting Predicates...")
    subs_df["Predicates"] = InstrPredicate.NONE
    subs_iter = [(i, sub) for i, sub in enumerate(subs) if i not in io_isos]
    # for i, sub in enumerate(tqdm(subs, disable=not PROGRESS)):
    for i, sub in tqdm(subs_iter, disable=not PROGRESS):
        # if i in io_isos:
        #     continue
        pred, num_loads, num_stores, num_branches = detect_predicates(sub)
        num_mems = num_loads + num_stores
        subs_df.loc[i, "Predicates"] = pred
        subs_df.loc[i, "#Loads"] = num_loads
        subs_df.loc[i, "#Stores"] = num_stores
        subs_df.loc[i, "#Mems"] = num_mems
        subs_df.loc[i, "#Branches"] = num_branches


# TODO: toggle on/off via cmdline?
with MeasureTime("Schedule Subs", verbose=TIMES):
    logger.info("Scheduling Subs...")
    # Very coarse measure to find longest path in subgraph (between inputs and outputs)
    subs_df["ScheduleLength"] = np.nan
    io_subs_iter = [(i, io_sub) for i, io_sub in enumerate(io_subs) if i not in io_isos]
    # for i, io_sub in enumerate(tqdm(io_subs, disable=not PROGRESS)):
    for i, io_sub in tqdm(io_subs_iter, disable=not PROGRESS):
        # if i in io_isos:
        #     continue
        print("i", i)
        print("io_sub", io_sub)
        sub_data = subs_df.iloc[i]
        print("sub_data", sub_data)
        inputs = subs_df.loc[i, "InputNodes"]
        # print("inputs", inputs)
        outputs = subs_df.loc[i, "OutputNodes"]
        terminators = subs_df.loc[i, "TerminatorNodes"]
        # print("outputs", outputs)

        def estimate_schedule_length(io_sub, ins, ends):
            # TODO: allow resource constraints (regfile ports, alus, ...)
            lengths = []
            for inp in ins:
                lengths_ = [
                    nx.shortest_path_length(io_sub, source=inp, target=outp)
                    for outp in ends
                    if nx.has_path(io_sub, inp, outp)
                ]
                # print("lengths_", lengths_)
                lengths += lengths_
            # print("lengths", lengths)
            # TODO: handle None?
            return max(lengths)

        ends = outputs | terminators
        length = estimate_schedule_length(io_sub, inputs, ends)
        # print("length", length)
        #  TODO
        subs_df.loc[i, "ScheduleLength"] = length


# TODO: filter before iso check?
with MeasureTime("Filtering subgraphs", verbose=TIMES):
    filtered_io = set()
    filtered_complex = set()
    filtered_simple = set()
    filtered_predicates = set()
    filtered_mem = set()
    filtered_branch = set()
    invalid = set()
    logger.info("Filtering subgraphs...")
    subs_iter = [(i, sub) for i, sub in enumerate(subs) if i not in io_isos]
    # for i, sub in enumerate(tqdm(subs, disable=not PROGRESS)):
    for i, sub in tqdm(subs_iter, disable=not PROGRESS):
        # if i in io_isos:
        #     continue
        # print("===========================")
        # print("i, sub", i, sub)
        num_nodes = len(sub.nodes)
        sub_data = subs_df.iloc[i]
        inputs = sub_data["InputNodes"]
        num_inputs = int(sub_data["#InputNodes"])
        outputs = sub_data["OutputNodes"]
        num_outputs = int(sub_data["#OutputNodes"])
        if num_inputs == 0:  # or num_outputs == 0:
            # TODO heck if branches and stores have outputs?
            invalid.add(i)
        elif MIN_INPUTS <= num_inputs <= MAX_INPUTS and MIN_OUTPUTS <= num_outputs <= MAX_OUTPUTS:
            pred = subs_df.loc[i, "Predicates"]
            if num_nodes > MAX_NODES:
                filtered_complex.add(i)
            elif num_nodes < MIN_NODES:
                filtered_simple.add(i)
            elif not check_predicates(pred, INSTR_PREDICATES):
                # TODO: add predicates details to df in prerequisite step
                filtered_predicates.add(i)
            else:
                num_loads = subs_df.loc[i, "#Loads"]
                num_stores = subs_df.loc[i, "#Stores"]
                num_mems = subs_df.loc[i, "#Mems"]
                if num_loads > MAX_LOADS or num_stores > MAX_STORES or num_mems > MAX_MEMS:
                    filtered_mem.add(i)
                else:
                    num_branches = subs_df.loc[i, "#Branches"]
                    if num_branches > MAX_BRANCHES:
                        filtered_branch.add(i)
        else:
            filtered_io.add(i)
    subs_df.loc[list(filtered_io), "Status"] = ExportFilter.FILTERED_IO
    subs_df.loc[list(filtered_complex), "Status"] = ExportFilter.FILTERED_COMPLEX
    subs_df.loc[list(filtered_simple), "Status"] = ExportFilter.FILTERED_SIMPLE
    subs_df.loc[list(filtered_predicates), "Status"] = ExportFilter.FILTERED_PRED
    subs_df.loc[list(filtered_mem), "Status"] = ExportFilter.FILTERED_MEM
    subs_df.loc[list(filtered_branch), "Status"] = ExportFilter.FILTERED_BRANCH
    subs_df.loc[list(invalid), "Status"] = ExportFilter.INVALID


# TODO: move to other file
def calc_encoding_footprint(enc_bits_sum, enc_size):
    if enc_size == 32:
        opcode_bits = 7
        remaining_bits = enc_size - opcode_bits
        enc_bits_left = remaining_bits - enc_bits_sum
        if enc_bits_left >= 0:
            enc_weight = 1 / (2**enc_bits_left)
        else:
            enc_weight = np.nan

        enc_footprint = enc_bits_sum / remaining_bits
    else:
        NotImplementedError(f"Encoding Size: {enc_size}")
    return enc_bits_left, enc_weight, enc_footprint


with MeasureTime("Determining IO names", verbose=TIMES):
    logger.info("Determining IO names...")
    filtered_subs_df = subs_df[(subs_df["Status"] & WRITE_GEN_FLT) > 0].copy()
    io_subs_iter = [(i, io_sub) for i, io_sub in enumerate(io_subs) if i in filtered_subs_df.index]
    # for i, sub in enumerate(tqdm(subs, disable=not PROGRESS)):
    for i, io_sub in tqdm(io_subs_iter, disable=not PROGRESS):
        sub = subs[i]
        sub_data = subs_df.iloc[i]
        inputs = sub_data["InputNodes"]
        outputs = sub_data["OutputNodes"]
        operand_names = []
        operand_nodes = []
        operand_types = []
        operand_reg_classes = []
        operand_dirs = []
        operand_enc_bits = []
        output_names = []
        for output_idx, j in enumerate(outputs):
            output_name = f"outp{output_idx}"
            output_node = io_sub.nodes[j]
            output_properties = output_node["properties"]
            op_type = output_properties["op_type"]
            output_names.append(output_name)
            # TODO: do not hardcode
            op_type_ = "REG"
            op_dir = "OUT"
            enc_bits = 5
            reg_class = "GPR"  # TODO: get from nodes!
            op_name = "rd" if output_idx == 0 else f"rd{output_idx+1}"
            operand_names.append(op_name)
            operand_nodes.append(j)
            operand_types.append(op_type_)
            operand_reg_classes.append(reg_class)
            operand_dirs.append(op_dir)
            operand_enc_bits.append(enc_bits)
        input_names = []
        for input_idx, j in enumerate(inputs):
            input_name = f"inp{input_idx}"
            input_node = io_sub.nodes[j]
            input_properties = input_node["properties"]
            op_type = input_properties["op_type"]
            input_names.append(input_name)
            # TODO: do not hardcode
            op_type_ = "REG"
            op_dir = "IN"
            enc_bits = 5
            reg_class = "GPR"
            op_name = f"rs{input_idx+1}"
            operand_names.append(op_name)
            operand_nodes.append(j)
            operand_types.append(op_type_)
            operand_reg_classes.append(reg_class)
            operand_dirs.append(op_dir)
            operand_enc_bits.append(enc_bits)
        subs_df.at[i, "OutputNames"] = output_names
        subs_df.at[i, "InputNames"] = input_names
        subs_df.at[i, "OperandNames"] = operand_names
        subs_df.at[i, "OperandTypes"] = operand_types
        subs_df.at[i, "OperandRegClasses"] = operand_reg_classes
        subs_df.at[i, "OperandNodes"] = operand_nodes
        subs_df.at[i, "OperandDirs"] = operand_dirs
        subs_df.at[i, "OperandEncBits"] = operand_enc_bits
        enc_bits_sum = sum(operand_enc_bits)
        subs_df.loc[i, "OperandEncBitsSum"] = enc_bits_sum
        for enc_size in ALLOWED_ENC_SIZES:
            enc_bits_left, enc_weight, enc_footprint = calc_encoding_footprint(enc_bits_sum, enc_size)
            subs_df.loc[i, f"EncodingBitsLeft ({enc_size} bits)"] = enc_bits_left
            subs_df.loc[i, f"EncodingWeight ({enc_size} bits)"] = enc_weight
            subs_df.loc[i, f"EncodingFootprint ({enc_size} bits)"] = enc_footprint


with MeasureTime("Analyze Constants", verbose=TIMES):
    logger.info("Analyzing constants...")
    filtered_subs_df = subs_df[(subs_df["Status"] & WRITE_GEN_FLT) > 0].copy()
    io_subs_iter = [(i, io_sub) for i, io_sub in enumerate(io_subs) if i in filtered_subs_df.index]
    # for i, sub in enumerate(tqdm(subs, disable=not PROGRESS)):
    for i, io_sub in tqdm(io_subs_iter, disable=not PROGRESS):
        sub = subs[i]
        sub_data = subs_df.iloc[i]
        constants = sub_data["ConstantNodes"]
        # print("constants")
        # constant_names = []
        constant_values = []
        constant_signs = []
        constant_min_bits = []
        for constant_idx, j in enumerate(constants):
            # print("j", j)
            constant_node = io_sub.nodes[j]
            # print("constant_node", constant_node)
            constant_properties = constant_node["properties"]
            # print("constant_properties", constant_properties)
            op_type = constant_properties["op_type"]
            assert op_type == "constant"
            # name = f"const{constant_idx}"
            val_str = constant_properties["inst"]
            assert val_str[-1] == "_"
            val = float(val_str[:-1])
            assert int(val) == val
            val = int(val)
            # print("val", val)
            sign = True  # For now handle all constants as signed
            # print("sign", sign)

            min_bits = 1 if val == 0 else (ceil(log2(abs(val))) + 1)
            # print("min_bits", min_bits)
            # TODO: handle float!!!
            # constant_names.append(name)
            constant_values.append(val)
            constant_signs.append(sign)
            constant_min_bits.append(min_bits)
            # input("##")
            # TODO: do not hardcode
        # subs_df.at[i, "ConstantNames"] = constant_names
        subs_df.at[i, "ConstantValues"] = constant_values
        subs_df.at[i, "ConstantSigns"] = constant_signs
        subs_df.at[i, "ConstantMinBits"] = constant_min_bits


with MeasureTime("Variation generation", verbose=TIMES):
    logger.info("Generating variations...")
    # 1. look for singleUse edges to reuse as output reg
    filtered_subs_df = subs_df[(subs_df["Status"] & WRITE_GEN_FLT) > 0].copy()
    io_subs_iter = [(i, io_sub) for i, io_sub in enumerate(io_subs) if i in filtered_subs_df.index]
    # for i, sub in enumerate(tqdm(subs, disable=not PROGRESS)):
    for i, io_sub in tqdm(io_subs_iter, disable=not PROGRESS):
        # print("i", i)
        # print("io_sub", io_sub)
        sub = subs[i]
        # print("sub", sub)
        sub_data = subs_df.iloc[i]
        # print("sub_data", sub_data)
        inputs = sub_data["InputNodes"]
        input_names = sub_data["InputNames"]
        num_inputs = int(sub_data["#InputNodes"])
        # print("inputs", inputs)
        # print("input_names", input_names)
        # print("num_inputs", num_inputs)
        operand_names = sub_data["OperandNames"]
        print("operand_names", operand_names)
        operand_nodes = sub_data["OperandNodes"]
        print("operand_nodes", operand_nodes)
        operand_dirs = sub_data["OperandDirs"]
        print("operand_dirs", operand_dirs)
        operand_types = sub_data["OperandTypes"]
        print("operand_types", operand_types)
        operand_enc_bits = sub_data["OperandEncBits"]
        print("operand_enc_bits", operand_enc_bits)
        operand_reg_classes = sub_data["OperandRegClasses"]
        print("operand_reg_classes", operand_reg_classes)
        outputs = sub_data["OutputNodes"]
        # print("outputs", outputs)
        output_names = sub_data["OutputNames"]
        # print("output_names", output_names)
        output_op_names = operand_names[: len(outputs)]
        # print("output_op_names", output_op_names)
        input_op_names = operand_names[len(outputs) :]
        # print("input_op_names", input_op_names)
        num_outputs = int(sub_data["#OutputNodes"])
        # print("num_outputs", num_outputs)
        if num_inputs < 1 or num_outputs < 1:
            continue
        # new = []
        for input_idx, j in enumerate(inputs):
            # print("input_idx", input_idx)
            # print("j", j)
            # TODO: reuse input node id for output or vice-versa?
            # Would create loop? Add constraint inp0 == outp0 to df?
            # TODO: check if input and output reg types match
            input_node_data = GF.nodes[j]
            # print("input_node_data", input_node_data)
            input_properties = input_node_data["properties"]
            # print("input_properties", input_properties)
            edge_count = 0
            single_use = None
            for src, dst, edge_data in GF.out_edges(j, data=True):
                # print("src", src)
                # print("dst", dst)
                # print("edge_data", edge_data)
                edge_properties = edge_data["properties"]
                # print("edge_properties", edge_properties)
                single_use_ = edge_properties.get("op_reg_single_use", None)
                # print("single_use_", single_use_)
                if single_use_ is not None:
                    single_use = single_use_
                edge_count += 1
            # print("edge_count", edge_count)
            # print("single_use", single_use)
            if single_use:
                assert edge_count == 1
                for output_idx, k in enumerate(outputs):
                    # print("output_idx", output_idx)
                    # print("k", k)
                    output_node_data = GF.nodes[j]
                    # print("output_node_data", output_node_data)
                    output_properties = output_node_data["properties"]
                    # print("output_properties", output_properties)
                    new_sub = sub.copy()
                    # print("new_sub", new_sub)
                    new_sub_data = sub_data.copy()
                    # print("new_sub_data", new_sub_data)
                    new_io_sub_ = io_sub.copy()
                    print("new_io_sub_", new_io_sub_)
                    print("new_io_sub_.nodes", new_io_sub_.nodes)
                    print("new_io_sub_.edges", new_io_sub_.edges)
                    new_io_sub_nodes = [x for x in io_sub.nodes if x != j]
                    new_input_node_data = input_node_data.copy()
                    new_input_node_data["alias"] = k
                    print("new_input_node_data", new_input_node_data)
                    new_input_node_id = max(GF.nodes) + 1
                    print("new_input_node_id", new_input_node_id)
                    GF.add_node(new_input_node_id, **new_input_node_data)
                    new_io_sub_nodes.append(new_input_node_id)
                    for src, dst, dat in io_sub.out_edges(j, data=True):
                        assert dst in io_sub.nodes
                        print("src", src)
                        print("dst", dst)
                        print("dat", dat)
                        GF.add_edge(new_input_node_id, dst, **dat)
                        print(f"{new_input_node_id} -> {dst}")
                    print("new_io_sub_nodes", new_io_sub_nodes)
                    new_io_sub = GF.subgraph(new_io_sub_nodes)
                    print("new_io_sub", new_io_sub)
                    print("new_io_sub.nodes", new_io_sub.nodes)
                    print("new_io_sub.edges", new_io_sub.edges)
                    # input("123")
                    input_op_name = input_op_names[input_idx]
                    output_op_name = output_op_names[output_idx]
                    new_constraint = f"{input_op_name} == {output_op_name}"
                    # print("new_constraint", new_constraint)
                    new_sub_data["Constraints"] = [new_constraint]
                    new_operand_names = [x for x in operand_names if x != input_op_name]
                    new_operand_dirs = [
                        operand_dirs[iii] if x != output_op_name else "INOUT"
                        for iii, x in enumerate(operand_names)
                        if x != input_op_name
                    ]
                    new_operand_nodes = [
                        # operand_nodes[iii] if x != input_op_name else new_input_node_id
                        operand_nodes[iii]
                        for iii, x in enumerate(operand_names)
                        if x != input_op_name
                    ]
                    new_operand_types = [
                        operand_types[iii] for iii, x in enumerate(operand_names) if x != input_op_name
                    ]
                    new_operand_reg_classes = [
                        operand_reg_classes[iii] for iii, x in enumerate(operand_names) if x != input_op_name
                    ]
                    new_operand_enc_bits = [
                        operand_enc_bits[iii] for iii, x in enumerate(operand_names) if x != input_op_name
                    ]
                    new_operand_enc_bits_sum = sum(new_operand_enc_bits)
                    parent = i
                    new_sub_data["Parent"] = parent
                    new_sub_data["Variations"] = ["ReuseIO"]
                    new_sub_data["OperandNames"] = new_operand_names
                    new_sub_data["OperandNodes"] = new_operand_nodes
                    new_input_nodes = [
                        inp if input_idx_ != input_idx else new_input_node_id for input_idx_, inp in enumerate(inputs)
                    ]
                    new_sub_data["InputNodes"] = new_input_nodes
                    new_sub_data["OperandDirs"] = new_operand_dirs
                    new_sub_data["OperandTypes"] = new_operand_types
                    new_sub_data["OperandRegClasses"] = new_operand_reg_classes
                    new_sub_data["OperandEncBits"] = new_operand_enc_bits
                    new_sub_data["OperandEncBitsSum"] = new_operand_enc_bits_sum
                    for enc_size in ALLOWED_ENC_SIZES:
                        enc_bits_left, enc_weight, enc_footprint = calc_encoding_footprint(enc_bits_sum, enc_size)
                        new_sub_data[f"EncodingBitsLeft ({enc_size} bits)"] = enc_bits_left
                        new_sub_data[f"EncodingWeight ({enc_size} bits)"] = enc_weight
                        new_sub_data[f"EncodingFootprint ({enc_size} bits)"] = enc_footprint
                    # TODO: re-calculate encoding footprint
                    new_sub_id = len(io_subs)
                    print("new_sub_id", new_sub_id)
                    new_sub_data["result"] = new_sub_id
                    # print("new_sub_data_", new_sub_data)
                    subs_df.loc[new_sub_id] = new_sub_data
                    subs.append(new_sub)
                    io_subs.append(new_io_sub)
                    # new.append(None)
                    # input("||")
        # print("new", new)

# TODO: if excoding space left (pre/post filter?) insert imm operands here where possible.
# (as new variation)

# TODO: filter before iso check?
with MeasureTime("Filtering subgraphs (Encoding)", verbose=TIMES):
    filtered_enc = set()
    logger.info("Filtering subgraphs (Encoding)...")
    filtered_subs_df = subs_df[(subs_df["Status"] & WRITE_GEN_FLT) > 0].copy()
    subs_iter = [(i, sub) for i, sub in enumerate(subs) if i in filtered_subs_df.index]
    # for i, sub in enumerate(tqdm(subs, disable=not PROGRESS)):
    for i, sub in tqdm(subs_iter, disable=not PROGRESS):
        sub_data = subs_df.iloc[i]
        valids = []
        for enc_size in ALLOWED_ENC_SIZES:
            enc_bits_left = sub_data[f"EncodingBitsLeft ({enc_size} bits)"]
            enc_weight = sub_data[f"EncodingWeight ({enc_size} bits)"]
            enc_footprint = sub_data[f"EncodingFootprint ({enc_size} bits)"]
            valid = True
            if enc_bits_left < MIN_ENC_BITS_LEFT:
                valid = False
            elif enc_weight > MAX_ENC_WEIGHT:
                valid = False
            elif enc_footprint > MAX_ENC_FOOTPRINT:
                valid = False
            valids.append(valid)
        # print("valids", valids)
        valid = any(valids)
        # print("valid", valid)
        if not valid:
            filtered_enc.add(i)
    # print("filtered_enc", filtered_enc)
    subs_df.loc[list(filtered_enc), "Status"] = ExportFilter.FILTERED_ENC


with MeasureTime("Filtering subgraphs (Weights)", verbose=TIMES):
    filtered_weights = set()
    logger.info("Filtering subgraphs (Weights)...")
    filtered_subs_df = subs_df[(subs_df["Status"] & WRITE_GEN_FLT) > 0].copy()
    subs_iter = [(i, sub) for i, sub in enumerate(subs) if i in filtered_subs_df.index]
    # for i, sub in enumerate(tqdm(subs, disable=not PROGRESS)):
    for i, sub in tqdm(subs_iter, disable=not PROGRESS):
        sub_data = subs_df.iloc[i]
        iso_weight = sub_data["IsoWeight"]
        if iso_weight < MIN_ISO_WEIGHT:
            filtered_weights.add(i)
    # print("filtered_weights", filtered_weights)
    subs_df.loc[list(filtered_weights), "Status"] = ExportFilter.FILTERED_WEIGHTS


sub_stmts: Dict[int, AnyNode] = {}
if WRITE_TREE_FMT:
    errs = set()
    filtered_subs_df = subs_df[(subs_df["Status"] & WRITE_GEN_FLT) > 0].copy()
    with MeasureTime("Tree Generation", verbose=TIMES):
        logger.info("Generation of Tree...")
        subs_iter = [(i, sub) for i, sub in enumerate(subs) if i in filtered_subs_df.index]
        # for i, sub in enumerate(tqdm(subs, disable=not PROGRESS)):
        for i, sub in tqdm(subs_iter, disable=not PROGRESS):
            sub_data = subs_df.iloc[i]
            try:
                stmts = generate_tree(
                    sub,
                    sub_data,
                    GF,
                    xlen=XLEN,
                )
                sub_stmts[i] = stmts
            except AssertionError as e:
                logger.exception(e)
                errs.add(i)
                continue
            if WRITE_TREE_FMT & ExportFormat.PKL:
                with open(OUT / f"tree{i}.pkl", "wb") as f:
                    pickle.dump(stmts, f)
                index_artifacts[i]["tree"] = OUT / f"tree{i}.pkl"
            if WRITE_TREE_FMT & ExportFormat.TXT:
                tree_txt = str(RenderTree(stmts))
                desc = generate_desc(i, sub_data, name=f"result{i}")
                tree_txt = f"// {desc}\n\n" + tree_txt
                with open(OUT / f"tree{i}.txt", "w") as f:
                    f.write(tree_txt)
                index_artifacts[i]["tree"] = OUT / f"tree{i}.pkl"
    subs_df.loc[list(errs), "Status"] = ExportFilter.ERROR


if WRITE_GEN:
    errs = set()
    filtered_subs_df = subs_df[(subs_df["Status"] & WRITE_GEN_FLT) > 0].copy()
    if WRITE_GEN_FMT & ExportFormat.MIR:
        with MeasureTime("MIR Generation", verbose=TIMES):
            logger.info("Generation of MIR...")
            subs_iter = [(i, sub) for i, sub in enumerate(subs) if i in filtered_subs_df.index]
            # for i, sub in enumerate(tqdm(subs, disable=not PROGRESS)):
            for i, sub in tqdm(subs_iter, disable=not PROGRESS):
                sub_data = subs_df.iloc[i]
                mir_code = generate_mir(sub, sub_data, topo, GF, name=f"result{i}")
                with open(OUT / f"result{i}.mir", "w") as f:
                    f.write(mir_code)
                index_artifacts[i]["mir"] = OUT / f"result{i}.mir"
    if WRITE_GEN_FMT & ExportFormat.CDSL:
        with MeasureTime("CDSL Generation", verbose=TIMES):
            logger.info("Generation of CDSL...")
            subs_iter = [(i, sub) for i, sub in enumerate(subs) if i in filtered_subs_df.index]
            # for i, sub in enumerate(tqdm(subs, disable=not PROGRESS)):
            for i, sub in tqdm(subs_iter, disable=not PROGRESS):
                sub_data = subs_df.iloc[i]
                stmts = sub_stmts.get(i, None)
                assert stmts is not None, "CDSL needs TREE"
                try:
                    desc = generate_desc(i, sub_data, name=f"result{i}")
                    cdsl_code = generate_cdsl(
                        stmts,
                        sub_data,
                        xlen=XLEN,
                        name=f"result{i}",
                        desc=desc,
                    )
                except AssertionError as e:
                    logger.exception(e)
                    errs.add(i)
                    continue
                with open(OUT / f"result{i}.core_desc", "w") as f:
                    f.write(cdsl_code)
                if index_artifacts is not None:
                    index_artifacts[i]["cdsl"] = OUT / f"result{i}.core_desc"
    if WRITE_GEN_FMT & ExportFormat.FLAT:
        with MeasureTime("FLAT Generation", verbose=TIMES):
            logger.info("Generation of FLAT...")
            subs_iter = [(i, sub) for i, sub in enumerate(subs) if i in filtered_subs_df.index]
            for i, sub in tqdm(subs_iter, disable=not PROGRESS):
                # if i not in filtered_subs_df.index:
                #     continue
                stmts = sub_stmts.get(i, None)
                assert stmts is not None, "FLAT needs TREE"
                sub_data = subs_df.iloc[i]
                try:
                    desc = generate_desc(i, sub_data, name=f"result{i}")
                    flat_code = generate_flat_code(stmts, desc=desc)
                except AssertionError as e:
                    logger.exception(e)
                    errs.add(i)
                    continue
                with open(OUT / f"result{i}.flat", "w") as f:
                    f.write(flat_code)
                index_artifacts[i]["flat"] = OUT / f"result{i}.flat"
    subs_df.loc[list(errs), "Status"] = ExportFilter.ERROR

# TODO: loop multiple times (tree -> MIR -> CDSL -> FLAT) not interleaved

# if len(duplicate_counts) > 0:
#     print()
#     print("Duplicates:")
#     for orig, dups in duplicate_counts.items():
#         print(f"result{orig}:\t", dups)

with MeasureTime("Finish DF", verbose=TIMES):
    logger.info("Finalizing DataFrame...")

    # subs_df["Iso"] = subs_df["result"].apply(lambda x: x in isos)
    subs_df["Status (str)"] = subs_df["Status"].apply(lambda x: str(ExportFilter(x)))
    subs_df["Predicates (str)"] = subs_df["Predicates"].apply(lambda x: str(InstrPredicate(x)))
    # print("subs_df")
    # print(subs_df)


# TODO: add helper to share code here!


if WRITE_SUB:
    with MeasureTime("Subgraph Export", verbose=TIMES):
        logger.info("Exporting subgraphs...")
        filtered_subs_df = subs_df[(subs_df["Status"] & WRITE_SUB_FLT) > 0].copy()
        subs_iter = [(i, sub) for i, sub in enumerate(subs) if i in filtered_subs_df.index]
        for i, sub in tqdm(subs_iter, disable=not PROGRESS):
            # if i not in filtered_subs_df.index:
            #     continue
            if WRITE_SUB_FMT & ExportFormat.DOT:
                graph_to_file(G_, OUT / f"sub{i}.dot")
            if WRITE_SUB_FMT & ExportFormat.PDF:
                graph_to_file(G_, OUT / f"sub{i}.pdf")
            if WRITE_SUB_FMT & ExportFormat.PNG:
                graph_to_file(G_, OUT / f"sub{i}.png")
            if WRITE_SUB_FMT & ExportFormat.PKL:
                with open(OUT / f"sub{i}.pkl", "wb") as f:
                    pickle.dump(G_.copy(), f)
                index_artifacts[i]["sub"] = OUT / f"sub{i}.pkl"


if WRITE_IO_SUB:
    with MeasureTime("Dumping I/O Subgraphs", verbose=TIMES):
        logger.info("Exporting I/O subgraphs...")
        filtered_subs_df = subs_df[(subs_df["Status"] & WRITE_IO_SUB_FLT) > 0].copy()
        io_subs_iter = [(i, io_sub) for i, io_sub in enumerate(io_subs) if i in filtered_subs_df.index]
        # for i, io_sub in enumerate(tqdm(io_subs, disable=not PROGRESS)):
        for i, io_sub in tqdm(io_subs_iter, disable=not PROGRESS):
            # if i in io_isos or i not in filtered_subs_df.index:
            #     continue
            if WRITE_IO_SUB_FMT & ExportFormat.DOT:
                graph_to_file(io_sub, OUT / f"io_sub{i}.dot")
            if WRITE_IO_SUB_FMT & ExportFormat.PDF:
                graph_to_file(io_sub, OUT / f"io_sub{i}.pdf")
            if WRITE_IO_SUB_FMT & ExportFormat.PNG:
                graph_to_file(io_sub, OUT / f"io_sub{i}.png")
            if WRITE_SUB_FMT & ExportFormat.PKL:
                with open(OUT / f"io_sub{i}.pkl", "wb") as f:
                    pickle.dump(io_sub.copy(), f)
                index_artifacts[i]["io_sub"] = OUT / f"io_sub{i}.pkl"


if WRITE_DF:
    with MeasureTime("Dump DFs", verbose=TIMES):
        filtered_subs_df = subs_df[(subs_df["Status"] & WRITE_DF_FLT) > 0].copy()
        logger.info("Exporting Global DataFrame...")
        if WRITE_DF_FMT & ExportFormat.CSV:
            global_df.to_csv(OUT / "global.csv")
        if WRITE_DF_FMT & ExportFormat.PKL:
            global_df.to_pickle(OUT / "global.pkl")
        logger.info("Exporting Subs DataFrame...")
        if WRITE_DF_FMT & ExportFormat.CSV:
            filtered_subs_df.to_csv(OUT / "subs.csv")
        if WRITE_DF_FMT & ExportFormat.PKL:
            filtered_subs_df.to_pickle(OUT / "subs.pkl")
            index_artifacts[None]["subs"] = OUT / "subs.pkl"


if WRITE_PIE:
    with MeasureTime("Generate Pie", verbose=TIMES):
        logger.info("Generating PieChart...")
        filtered_subs_df = subs_df[(subs_df["Status"] & WRITE_PIE_FLT) > 0].copy()
        pie_df, pie_fig = generate_pie_chart(filtered_subs_df)
        pie2_df, pie2_fig = generate_pie2_chart(filtered_subs_df)

    with MeasureTime("Dump Pie", verbose=TIMES):
        logger.info("Exporting PieChart...")
        if WRITE_PIE_FMT & ExportFormat.PDF:
            pie_fig.write_image(OUT / "pie.pdf")
            pie2_fig.write_image(OUT / "pie2.pdf")
        if WRITE_PIE_FMT & ExportFormat.PNG:
            pie_fig.write_image(OUT / "pie.png")
            pie2_fig.write_image(OUT / "pie2.png")
        if WRITE_PIE_FMT & ExportFormat.HTML:
            pie_fig.write_html(OUT / "pie.html")
            pie2_fig.write_html(OUT / "pie2.html")
        if WRITE_PIE_FMT & ExportFormat.CSV:
            pie_df.to_csv(OUT / "pie.csv")
            pie2_df.to_csv(OUT / "pie2.csv")
        if WRITE_PIE_FMT & ExportFormat.PKL:
            pie_df.to_pickle(OUT / "pie.pkl")
            pie2_df.to_pickle(OUT / "pie2.pkl")
            index_artifacts[None]["pie"] = OUT / "pie.pkl"
            index_artifacts[None]["pie2"] = OUT / "pie2.pkl"


if WRITE_INDEX:
    with MeasureTime("Write Index", verbose=TIMES):
        filtered_subs_df = subs_df[(subs_df["Status"] & WRITE_INDEX_FLT) > 0].copy()
        logger.info("Writing Index File...")
        if WRITE_INDEX_FMT & ExportFormat.YAML:
            write_index_file(OUT / "index.yml", filtered_subs_df, global_df, index_artifacts)

if TIMES:
    print(MeasureTime.summary())
    MeasureTime.write_csv(OUT / "times.csv")

# TODO: estimate encoding usage (free bits, rel. for enc_size 16/32/48)
# TODO: for all constants query isomorph subs and count the different values
# TODO: pass allowed imm width (5bit, 12bits)
# TODO: also allow non-sequential imm ranges? (i.e. 1,2,4,8,...)?
# TODO: check if STAGE_6 (64, post regalloc) works?
# TODO: figure out if rd_wb = rs1 can be detected automatically? -> check single_use edge property
# TODO: Merge equivalent subs with respect to isCommutable
# TODO: Generate variations (generalize/specialize)
