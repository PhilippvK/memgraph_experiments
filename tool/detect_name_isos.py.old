import pickle
import logging
import argparse
# from pathlib import Path
# from collections import defaultdict

import yaml
# import pandas as pd
from tqdm import tqdm
import networkx as nx
import networkx.algorithms.isomorphism as iso


from .hash import add_hash_attr

logger = logging.getLogger("detect_name_isos")

# TODO: expose via python (via __main__)


def handle_cmdline():
    # TODO: add help messages
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("index", nargs="+", help="TODO")  # print if None
    parser.add_argument("--log", default="info", choices=["critical", "error", "warning", "info", "debug"], help="TODO")
    parser.add_argument("--drop", action="store_true", help="TODO")
    parser.add_argument("--output", "-o", default=None, help="TODO")  # print if None
    parser.add_argument("--progress", action="store_true", help="TODO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()))
    return args


args = handle_cmdline()
INS = args.index
assert len(INS) == 1
in_path = INS[0]
OUT = args.output
DROP = args.drop
PROGRESS = args.progress

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

all_name_isos = set()
sub_name_isos = {}
for i, c in tqdm(enumerate(candidates), disable=not PROGRESS):
    print("i", i)
    print("c", c)
    sub_data = c["properties"]
    print("sub_data", sub_data)
    input_nodes = sub_data["InputNodes"]
    print("input_nodes", input_nodes)
    input_names = sub_data["InputNames"]
    print("input_names", input_names)
    sub_hash = sub_data["SubHash"]
    print("sub_hash", sub_hash)
    io_sub_hash = sub_data["IOSubHash"]
    print("io_sub_hash", io_sub_hash)
    io_sub = candidate_io_subs[i]
    print("io_sub", io_sub)
    # artifacts = c["artifacts"]
    # flat_artifact = artifacts["flat"]
    # print("flat_artifact", flat_artifact)
    # TODO: speedup using hashes

    name_isos = set()
    for j, c_ in tqdm(enumerate(candidates), disable=not PROGRESS):
        if j <= i:
            continue
        if j in all_name_isos:
            continue
        sub_data_ = c_["properties"]
        print("sub_data_", sub_data_)
        input_nodes_ = sub_data_["InputNodes"]
        print("input_nodes_", input_nodes_)
        input_names_ = sub_data_["InputNames"]
        print("input_names_", input_names_)
        sub_hash_ = sub_data_["SubHash"]
        print("sub_hash_", sub_hash_)
        io_sub_hash_ = sub_data_["IOSubHash"]
        print("io_sub_hash_", io_sub_hash_)
        io_sub_ = candidate_io_subs[j]
        print("io_sub_", io_sub_)
        if sub_hash != sub_hash_:
            continue
        assert io_sub_hash != io_sub_hash_

        # TODO: add automatically
        add_hash_attr(io_sub, attr_name="hash_attr2", ignore_const=False, ignore_names=True)
        add_hash_attr(io_sub_, attr_name="hash_attr2", ignore_const=False, ignore_names=True)
        nm = iso.categorical_node_match("hash_attr2", None)
        em = iso.categorical_edge_match("hash_attr2", None)
        matcher = nx.algorithms.isomorphism.DiGraphMatcher(io_sub, io_sub_, node_match=nm, edge_match=em)
        check = matcher.is_isomorphic()
        # print("check", check)
        mapping = matcher.mapping
        # print("mapping", mapping)
        if check:
            ins_mapping = {k: v for k, v in mapping.items() if k in input_nodes}
            # print("ins_mapping", ins_mapping)
            # TODO: outs_mapping!
            names = [input_names[input_nodes.index(k)] for k in ins_mapping.keys()]
            # print("names", names)
            names_ = [input_names_[input_nodes_.index(v)] for v in ins_mapping.values()]
            # print("names_", names_)
            if names == names_:
                continue
            name_isos.add(j)
            # print("======")
            # print("j", j)
            # print("i,j", i, j)
            # print("c_", c_)
            # print("sub_data_", sub_data_)
            # print("flat_artifact_", flat_artifact_)
            # input(">")
    if len(name_isos) > 0:
        sub_name_isos[i] = name_isos
    all_name_isos |= name_isos

print("all_name_isos", all_name_isos, len(all_name_isos))
print("sub_name_isos", sub_name_isos, len(sub_name_isos))
# for i, specs in sub_specializations.items():
#     print(f"{i}\t: {len(specs)}")

if DROP:
    candidates = [x for i, x in enumerate(candidates) if i not in all_name_isos]

temp = {"global": {"artifacts": [], "properties": global_properties}, "candidates": candidates}
if OUT:
    with open(OUT, "w") as f:
        yaml.dump(temp, f)
else:
    yaml_str = yaml.dump(temp)
    print(yaml_str)
