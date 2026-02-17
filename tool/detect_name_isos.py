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

logger = logging.getLogger("detect_name_isos")


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


def name_iso_helper(io_sub, io_sub_, sub_data, sub_data_, i, j):
    input_nodes = sub_data["InputNodes"]
    # print("input_nodes", input_nodes)
    input_names = sub_data["InputNames"]
    # print("input_names", input_names)
    output_nodes = sub_data["OutputNodes"]
    # print("output_nodes", output_nodes)
    output_names = sub_data["OutputNames"]
    # print("output_names", output_names)
    # sub_hash = sub_data["SubHash"]
    # print("sub_hash", sub_hash)
    # io_sub_hash = sub_data["IOSubHash"]
    # print("io_sub_hash", io_sub_hash)
    # print("sub_data_", sub_data_)
    input_nodes_ = sub_data_["InputNodes"]
    # print("input_nodes_", input_nodes_)
    input_names_ = sub_data_["InputNames"]
    # print("input_names_", input_names_)
    output_nodes_ = sub_data_["OutputNodes"]
    # print("output_nodes_", output_nodes_)
    output_names_ = sub_data_["OutputNames"]
    # print("output_names_", output_names_)
    # sub_hash_ = sub_data_["SubHash"]
    # print("sub_hash_", sub_hash_)
    # io_sub_hash_ = sub_data_["IOSubHash"]
    # # print("io_sub_hash_", io_sub_hash_)
    # io_sub_ = candidate_io_subs[j]
    # print("io_sub_", io_sub_)
    # if sub_hash != sub_hash_:  # TODO: check if Subhash is already comm aware?
    #     continue
    # assert io_sub_hash != io_sub_hash_

    # TODO: add automatically
    # ignore_alias = False
    ignore_alias = True
    add_hash_attr(
        io_sub,
        attr_name="hash_attr2",
        ignore_const=False,
        ignore_names=True,
        handle_commutable=True,
        ignore_alias=ignore_alias,
    )
    add_hash_attr(
        io_sub_,
        attr_name="hash_attr2",
        ignore_const=False,
        ignore_names=True,
        handle_commutable=True,
        ignore_alias=ignore_alias,
    )
    nm = iso.categorical_node_match("hash_attr2", None)
    # em = iso.categorical_edge_match("hash_attr2", None)
    em = categorical_edge_match_multidigraph("hash_attr2", None)
    matcher = nx.algorithms.isomorphism.DiGraphMatcher(io_sub, io_sub_, node_match=nm, edge_match=em)
    check = matcher.is_isomorphic()
    # print("check", check)
    mapping = matcher.mapping
    # print("mapping", mapping)
    # if i == 11 and j == 32:
    #     STOP = True
    if check:
        ins_mapping = {k: v for k, v in mapping.items() if k in input_nodes}
        # print("ins_mapping", ins_mapping)
        outs_mapping = {k: v for k, v in mapping.items() if k in output_nodes}
        # print("outs_mapping", outs_mapping)
        # print("?", [(io_sub.nodes[k]["hash_attr2"], io_sub_.nodes[v]["hash_attr2"]) for k, v in ins_mapping.items()])
        # print("?", [(io_sub.nodes[k].get("alias"), io_sub_.nodes[v].get("alias")) for k, v in ins_mapping.items()])
        alias_constraints = {
            k: io_sub.nodes[k].get("alias")
            for k in ins_mapping.keys()
            if io_sub.nodes[k].get("alias") is not None
        }
        alias_constraints_ = {
            v: io_sub_.nodes[v].get("alias")
            for v in ins_mapping.values()
            if io_sub_.nodes[v].get("alias") is not None
        }
        # print("alias_constraints", alias_constraints)
        # print("alias_constraints_", alias_constraints_)
        mapped_alias_constraints = {ins_mapping[k]: outs_mapping[v] for k, v in alias_constraints.items()}
        # print("mapped_alias_constraints", mapped_alias_constraints)
        if len(alias_constraints) > 0 or len(alias_constraints_) > 0:
            if len(alias_constraints_) != len(alias_constraints):
                # if len(alias_constraints) == 0:
                #     parent2variations[i].add(j)
                # elif len(alias_constraints_) == 0:
                #     parent2variations[j].add(i)
                # print("cont1")
                return False
            # assert len(alias_constraints) == len(alias_constraints_)
            if len(alias_constraints) != 1:
                raise NotImplementedError("multiple aliases not supported")
            have_same_constraints = (
                list(mapped_alias_constraints.items())[0] == list(alias_constraints_.items())[0]
            )
            # print("have_same_constraints", have_same_constraints)
            # print("i", i)
            # print("j", j)
            # input("?")
            if not have_same_constraints:
                # group = None
                # for group_ in missmatched_constraints:
                #     if i in group_:
                #         group = group_
                #         break
                #     if j in group_:
                #         group = group_
                #         break
                # if group is None:
                #     group = {i, j}
                #     missmatched_constraints.append(group)
                # else:
                #     group.update({i, j})
                # print("cont2")
                return False
        in_names = [input_names[input_nodes.index(k)] for k in ins_mapping.keys()]
        # print("in_names", in_names)
        in_names_ = [input_names_[input_nodes_.index(v)] for v in ins_mapping.values()]
        # print("in_names_", in_names_)
        out_names = [output_names[output_nodes.index(k)] for k in outs_mapping.keys()]
        # print("out_names", out_names)
        out_names_ = [output_names_[output_nodes_.index(v)] for v in outs_mapping.values()]
        # print("out_names_", out_names_)
        if in_names == in_names_ and out_names == out_names_:
            # print("cont3")
            logger.warning("Subs %d & %d are fully-isomorph!", i, j)
            return False
        if out_names != out_names:
            raise NotImplementedError("Out name isos")
        return True
    return False


def collect_name_isos(candidates, candidate_io_subs, progress: bool = False):
    all_name_isos = set()
    sub_name_isos = {}
    missmatched_constraints = []
    from collections import defaultdict
    parent2variations = defaultdict(set)
    for i, c in tqdm(enumerate(candidates), disable=not progress):
        # print("i", i)
        # print("c", c)
        sub_data = c["properties"]
        # print("sub_data", sub_data)
        # input_nodes = sub_data["InputNodes"]
        # # print("input_nodes", input_nodes)
        # input_names = sub_data["InputNames"]
        # # print("input_names", input_names)
        # output_nodes = sub_data["OutputNodes"]
        # # print("output_nodes", output_nodes)
        # output_names = sub_data["OutputNames"]
        # # print("output_names", output_names)
        # sub_hash = sub_data["SubHash"]
        # # print("sub_hash", sub_hash)
        # io_sub_hash = sub_data["IOSubHash"]
        # print("io_sub_hash", io_sub_hash)
        io_sub = candidate_io_subs[i]
        # print("io_sub", io_sub)
        # artifacts = c["artifacts"]
        # flat_artifact = artifacts["flat"]
        # print("flat_artifact", flat_artifact)
        # TODO: speedup using hashes

        name_isos = set()
        # STOP = False
        for j, c_ in tqdm(enumerate(candidates), disable=not progress):
            # if STOP:
            #     input(">")
            #     STOP = False
            # if j <= i:
            if j <= i:
                continue
            # if j in all_name_isos:
            #     continue
            sub_data_ = c_["properties"]
            io_sub_ = candidate_io_subs[j]
            is_name_iso = name_iso_helper(io_sub, io_sub_, sub_data, sub_data_, i, j)
            if is_name_iso:
                # print("i", i)
                # print("j", j)
                # input("!")
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

    # print("all_name_isos", all_name_isos, len(all_name_isos))
    # print("sub_name_isos", sub_name_isos, len(sub_name_isos))
    # print("missmatched_constraints", missmatched_constraints)
    # print("p2v", parent2variations)
    return all_name_isos, sub_name_isos, missmatched_constraints


def main():
    args = handle_cmdline()
    INS = args.index
    assert len(INS) == 1
    in_path = INS[0]
    OUT = args.output
    DROP = args.drop

    yaml_data, candidates, candidate_io_subs = load_candidates(in_path, progress=args.progress)

    all_name_isos, sub_name_isos, missmatched_constraints = collect_name_isos(candidates, candidate_io_subs, progress=args.progress)
    # for i, specs in sub_specializations.items():
    #     print(f"{i}\t: {len(specs)}")

    if DROP:
        candidates = [x for i, x in enumerate(candidates) if i not in all_name_isos]
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
