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
from .iso import categorical_edge_match_multidigraph

logger = logging.getLogger("filter_isos")


def handle_cmdline():
    # TODO: add help messages
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("index", nargs="+", help="TODO")  # print if None
    parser.add_argument("--log", default="info", choices=["critical", "error", "warning", "info", "debug"], help="TODO")
    # parser.add_argument("--drop", action="store_true", help="TODO")
    parser.add_argument("--output", "-o", default=None, help="TODO")  # print if None
    parser.add_argument("--progress", action="store_true", help="TODO")
    parser.add_argument("--ignore-const", action="store_true", help="TODO")
    parser.add_argument("--ignore-names", action="store_true", help="TODO")
    parser.add_argument("--ignore-alias", action="store_true", help="TODO")
    parser.add_argument("--handle-commutable", action="store_true", help="TODO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()))
    return args


def load_candidates(in_path, progress: bool = False):
    candidates = []
    candidate_io_subs = []
    logger.info("Loading input %s", in_path)
    with open(in_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    # global_data = yaml_data["global"]
    candidates_data = yaml_data["candidates"]
    for candidate_data in tqdm(candidates_data, disable=not progress):
        artifacts = candidate_data["artifacts"]
        # properties = candidate_data["properties"]
        candidates.append(candidate_data)
        io_sub_path = artifacts.get("io_sub", None)
        assert io_sub_path is not None, "PKL for io_sub not found in artifacts"
        with open(io_sub_path, "rb") as f:
            io_sub = pickle.load(f)
        candidate_io_subs.append(io_sub)

    total_count = len(candidates)
    logger.info(f"Loaded {total_count} candidates...")
    return yaml_data, candidates, candidate_io_subs


def collect_isos(
    candidates,
    candidate_io_subs,
    ignore_const: bool = False,
    ignore_names: bool = False,
    handle_commutable: bool = True,
    ignore_alias: bool = False,
    progress: bool = False,
):
    all_isos = set()
    sub_isos = {}
    for i, c in tqdm(enumerate(candidates), disable=not progress):
        # print("i", i)
        # print("c", c)
        sub_data = c["properties"]
        # print("sub_data", sub_data)
        # input_nodes = sub_data["InputNodes"]
        # print("input_nodes", input_nodes)
        # input_names = sub_data["InputNames"]
        # print("input_names", input_names)
        # output_nodes = sub_data["OutputNodes"]
        # print("output_nodes", output_nodes)
        # output_names = sub_data["OutputNames"]
        # print("output_names", output_names)
        sub_hash = sub_data["SubHash"]
        # print("sub_hash", sub_hash)
        # io_sub_hash = sub_data["IOSubHash"]
        # print("io_sub_hash", io_sub_hash)
        io_sub = candidate_io_subs[i]
        # print("io_sub", io_sub)
        # artifacts = c["artifacts"]
        # flat_artifact = artifacts["flat"]
        # print("flat_artifact", flat_artifact)
        # TODO: speedup using hashes

        isos = set()
        # STOP = False
        for j, c_ in tqdm(enumerate(candidates), disable=not progress):
            # if STOP:
            #     input(">")
            #     STOP = False
            if j <= i:
                continue
            if j in all_isos:
                continue
            sub_data_ = c_["properties"]
            # print("sub_data_", sub_data_)
            # input_nodes_ = sub_data_["InputNodes"]
            # print("input_nodes_", input_nodes_)
            # input_names_ = sub_data_["InputNames"]
            # print("input_names_", input_names_)
            # output_nodes_ = sub_data_["OutputNodes"]
            # print("output_nodes_", output_nodes_)
            # output_names_ = sub_data_["OutputNames"]
            # print("output_names_", output_names_)
            sub_hash_ = sub_data_["SubHash"]
            # print("sub_hash_", sub_hash_)
            # io_sub_hash_ = sub_data_["IOSubHash"]
            # print("io_sub_hash_", io_sub_hash_)
            io_sub_ = candidate_io_subs[j]
            # print("io_sub_", io_sub_)
            if sub_hash != sub_hash_:
                continue
            # assert io_sub_hash != io_sub_hash_

            # TODO: add automatically
            # ignore_alias = False
            add_hash_attr(
                io_sub,
                attr_name="hash_attr2",
                ignore_const=ignore_const,
                ignore_names=ignore_names,
                handle_commutable=handle_commutable,
                ignore_alias=ignore_alias,
            )
            add_hash_attr(
                io_sub_,
                attr_name="hash_attr2",
                ignore_const=ignore_const,
                ignore_names=ignore_names,
                handle_commutable=handle_commutable,
                ignore_alias=ignore_alias,
            )
            # TODO: share code with iso.py
            nm = iso.categorical_node_match("hash_attr2", None)
            # em = iso.categorical_edge_match("hash_attr2", None)
            em = categorical_edge_match_multidigraph("hash_attr2", None)
            matcher = nx.algorithms.isomorphism.DiGraphMatcher(io_sub, io_sub_, node_match=nm, edge_match=em)
            check = matcher.is_isomorphic()
            # print("check", check)
            if check:
                isos.add(j)
        if len(isos) > 0:
            sub_isos[i] = isos
        all_isos |= isos

    print("all_isos", all_isos, len(all_isos))
    print("sub_isos", sub_isos, len(sub_isos))
    return all_isos, sub_isos


def main():
    args = handle_cmdline()
    INS = args.index
    assert len(INS) == 1
    in_path = INS[0]
    OUT = args.output

    yaml_data, candidates, candidate_io_subs = load_candidates(in_path, progress=args.progress)

    all_isos, sub_isos = collect_isos(
        candidates,
        candidate_io_subs,
        ignore_const=args.ignore_const,
        ignore_names=args.ignore_names,
        handle_commutable=args.handle_commutable,
        ignore_alias=args.ignore_alias,
        progress=args.progress,
    )
    # for i, specs in sub_specializations.items():
    #     print(f"{i}\t: {len(specs)}")

    DROP = True
    if DROP:
        candidates = [x for i, x in enumerate(candidates) if i not in all_isos]
        UPDATE_IDS = True
        if UPDATE_IDS:
            for i, candidate in enumerate(candidates):
                candidate["id"] = i

    yaml_data["candidates"] = candidates
    if OUT:
        with open(OUT, "w") as f:
            yaml.dump(yaml_data, f)
    else:
        yaml_str = yaml.dump(yaml_data)
        print(yaml_str)


if __name__ == "__main__":
    main()
