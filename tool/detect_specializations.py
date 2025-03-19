import pickle
import logging
import argparse
# from pathlib import Path
from collections import defaultdict

import yaml
import pandas as pd
from tqdm import tqdm
import networkx as nx
import networkx.algorithms.isomorphism as iso


# from .iso import calc_io_isos
from .graph_utils import graph_to_file

logger = logging.getLogger("combine_index")

# TODO: expose via python (via __main__)

NEUTRAL_ELEMENTS = {
    "ADD": [(None, 0), (0, None)],
    "ADDI": [(None, 0), (0, None)],
    "SUB": [(None, 0)],
    "MUL": [(None, 1), (1, None)],
    "OR": [(None, 0), (0, None)],
    "ORI": [(None, 0), (0, None)],
    "SLL": [(None, 0)],
    "SLLI": [(None, 0)],
    # "SRL"
    # "SRLI"
    # "SRA"
    # "SRAI"
    # "XOR"
    # "XOR"
    # ...
}


def handle_cmdline():
    # TODO: add help messages
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("index", nargs="+", help="TODO")  # print if None
    parser.add_argument("--log", default="info", choices=["critical", "error", "warning", "info", "debug"], help="TODO")
    parser.add_argument("--output", "-o", default=None, help="TODO")  # print if None
    parser.add_argument("--drop", action="store_true", help="TODO")
    parser.add_argument("--graph", default=None, help="TODO")  # print if None
    parser.add_argument("--progress", action="store_true", help="TODO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()))
    return args


args = handle_cmdline()
INS = args.index
assert len(INS) == 1
in_path = INS[0]
OUT = args.output
PROGRESS = args.progress
DROP = args.drop
GRAPH = args.graph

candidates = []
candidate_io_subs = []
global_properties = []

logger.info("Loading input %s", in_path)
with open(in_path, "r") as f:
    yaml_data = yaml.safe_load(f)
global_data = yaml_data["global"]
global_properties_ = global_data["properties"]
candidates_data = yaml_data["candidates"]
path_ids = set()
for candidate_data in tqdm(candidates_data, disable=not args.progress):
    artifacts = candidate_data["artifacts"]
    properties = candidate_data["properties"]
    candidates.append(candidate_data)
    io_sub_path = artifacts.get("io_sub", None)
    assert io_sub_path is not None, "PKL for io_sub not found in artifacts"
    with open(io_sub_path, "rb") as f:
        io_sub = pickle.load(f)
    candidate_io_subs.append(io_sub)
global_properties_[0]["candidate_count"] = len(path_ids)
global_properties += global_properties_

total_count = len(candidates)
logger.info(f"Loaded {total_count} candidates from {len(INS)} files...")


sub_specializations = {}
# WARN: symmetry can not be used here!
for i, c in tqdm(enumerate(candidates), disable=not PROGRESS):
    print("i", i)
    # print("c", c)
    sub_data = c["properties"]
    num_nodes = sub_data["#Nodes"]
    # print("sub_data", sub_data)
    io_sub = candidate_io_subs[i]
    # print("io_sub", io_sub)
    artifacts = c["artifacts"]
    flat_artifact = artifacts["flat"]
    # print("flat_artifact", flat_artifact)
    # TODO: speedup using hashes

    specs = []
    for j, c_ in tqdm(enumerate(candidates), disable=not PROGRESS):
        if i == j:
            continue
        # TODO: skip if num consts does not match?
        sub_data_ = c_["properties"]
        num_nodes_ = sub_data_["#Nodes"]
        if (num_nodes - num_nodes_) != 1:
            continue
        instrs = sub_data["Instrs"]
        nodes = sub_data["Nodes"]
        instrs_ = sub_data_["Instrs"]
        # print("1", instrs_-instrs)
        # print("2", instrs-instrs_)
        from collections import Counter

        def is_sublist(x, y):
            cx, cy = Counter(x), Counter(y)
            cy.subtract(cx)
            dropped = [instruction for instruction, count in cy.items() if count > 0]
            return min(cy.values()) >= 0, dropped

        is_sublist_, dropped_instr = is_sublist(instrs_, instrs)
        if not is_sublist_:
            continue

        assert len(dropped_instr) == 1
        dropped_instr = dropped_instr[0]
        print("instrs", instrs)
        print("instrs_", instrs_)
        print("dropped_instr", dropped_instr)
        possible_dropped_nodes = [nodes[idx] for idx, instruction in enumerate(instrs) if instruction == dropped_instr]
        print("num_nodes", num_nodes)
        print("num_nodes_", num_nodes_)
        print("possible_dropped_nodes", possible_dropped_nodes)
        io_sub_ = candidate_io_subs[j]
        artifacts_ = c_["artifacts"]
        flat_artifact_ = artifacts_["flat"]
        with open(flat_artifact, "r") as f:
            flat = f.read()
        with open(flat_artifact_, "r") as f:
            flat_ = f.read()
        print("flat", flat)
        print("flat_", flat_)
        for dropped_node in possible_dropped_nodes:
            print("dropped_node", dropped_node)

            # def try_remove_node(ios_sub, to_remove):
            #     raise NotImplementedError

            # temp_io_sub = try_remove_node(io_sub, dropped_node)
            # if temp_io_sub is None:
            #     print("cont (illegal)")
            #     continue

            # def drop_io_nodes(ios_sub):
            #     raise NotImplementedError

            # sub_ = drop_io_nodes(io_sub_)
            # assert sub_ is not None

            # temp_sub = drop_io_nodes(temp_io_sub)
            # assert temp_sub is not None
            # TODO: (earlier) check if dropped node is a source or sink (ignoring I/Os)
            # TODO: check if temp_sub and sub_ are isomorph
            # TODO: normalize IO
            # TODO: check if temp_io_sub and io_sub_ are isomorph
            # TODO: define neural_elements (if required)

        # input(">>>")
        spec = (j, f"{dropped_instr} [{dropped_node}] -> ?")
        specs.append(spec)

        nm = iso.categorical_node_match("hash_attr_ignore_const", None)
        em = iso.categorical_edge_match("hash_attr", None)
        matcher = nx.algorithms.isomorphism.DiGraphMatcher(io_sub, io_sub_, node_match=nm, edge_match=em)
        check = matcher.is_isomorphic()
        if check:
            # print("check", check)
            print("======")
            # print("j", j)
            print("i,j", i, j)
            # print("c_", c_)
            # print("sub_data_", sub_data_)
            # print("flat_artifact_", flat_artifact_)
            mapping = matcher.mapping
            print("mapping", mapping)
    if len(specs) > 0:
        sub_specializations[i] = specs
print("sub_specializations", sub_specializations)
for i, specs in sub_specializations.items():
    print(f"{i}\t: {len(specs)}")

spec_graph = nx.DiGraph()
for i, c in enumerate(candidates):
    spec_graph.add_node(i, label=f"c{i}")
for i, specs in sub_specializations.items():
    for spec in specs:
        j, label = spec
        spec_graph.add_edge(i, j, label=label)
sources = [x for x in spec_graph.nodes() if spec_graph.in_degree(x) == 0 ]
for source in sources:
    spec_graph.nodes[source].update({"style": "filled", "fillcolor": "lightgray"})
print("sources", sources, len(sources))
print("spec_graph", spec_graph)
if GRAPH is not None:
    graph_to_file(spec_graph, GRAPH)


if DROP:
    candidates = [x for i, x in enumerate(candidates) if i in sources]

temp = {"global": {"artifacts": [], "properties": global_properties}, "candidates": candidates}
if OUT:
    with open(OUT, "w") as f:
        yaml.dump(temp, f)
else:
    yaml_str = yaml.dump(temp)
    print(yaml_str)
