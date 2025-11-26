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
from .hash import add_hash_attr
from .iso import categorical_edge_match_multidigraph

logger = logging.getLogger("combine_index")

# TODO: expose via python (via __main__)


def handle_cmdline():
    # TODO: add help messages
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("index", nargs="+", help="TODO")  # print if None
    parser.add_argument("--log", default="info", choices=["critical", "error", "warning", "info", "debug"], help="TODO")
    parser.add_argument("--output", "-o", default=None, help="TODO")  # print if None
    parser.add_argument("--drop-duplicates", action="store_true", help="TODO")
    parser.add_argument("--overlaps", default=None, help="TODO")
    # parser.add_argument("--venn", default=None, help="TODO")
    # parser.add_argument("--sankey", default=None, help="TODO")
    parser.add_argument("--progress", action="store_true", help="TODO")
    # parser.add_argument("--sort-by", type=str, default=None, help="TODO")
    # parser.add_argument("--sort-asc", action="store_true", help="TODO")
    # parser.add_argument("--topk", type=int, default=None, help="Only keep the k first items")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()))
    return args


def load_candidates(in_path, progress: bool = False):
    candidates = []
    candidate_io_subs = []

    logger.info("Loading input %s", in_path)
    with open(in_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    candidates_data = yaml_data["candidates"]
    for candidate_data in tqdm(candidates_data, disable=not progress):
        # candidate_data["id"] = i
        artifacts = candidate_data["artifacts"]
        # properties = candidate_data["properties"]
        # path_ids.add(i)
        candidates.append(candidate_data)
        # if DROP_DUPLICATES:
        if True:
            io_sub_path = artifacts.get("io_sub", None)
            assert io_sub_path is not None, "PKL for io_sub not found in artifacts"
            with open(io_sub_path, "rb") as f:
                io_sub = pickle.load(f)
            candidate_io_subs.append(io_sub)
        # i += 1

    total_count = len(candidates)
    logger.info(f"Loaded {total_count} candidates...")
    return candidates, candidate_io_subs


def get_const_value_subs(candidates, candidate_io_subs, ignore_covered: bool = True, progress: bool = False):
    sub_const_value_subs = {}
    covered = set()
    for i, c in tqdm(enumerate(candidates), disable=not progress):
        # print("i", i)
        # print("c", c)
        sub_data = c["properties"]
        # print("sub_data", sub_data)
        constants = sub_data["ConstantNodes"]
        # print("constants", constants)
        constant_values = sub_data["ConstantValues"]
        # print("constant_values", constant_values)
        io_sub = candidate_io_subs[i]
        # print("io_sub", io_sub)
        # artifacts = c["artifacts"]
        # flat_artifact = artifacts["flat"]
        # print("flat_artifact", flat_artifact)

        const_value_subs = {}
        for c, constant in enumerate(constants):
            const_value_subs[c] = defaultdict(list)
            constant_value = constant_values[c]
            const_value_subs[c][constant_value].append(i)

        STOP = False
        for j, c_ in tqdm(enumerate(candidates), disable=not progress):
            if j <= i:
                # print("cont (sym.)")
                continue
            if not ignore_covered and j in covered:
                # print("cont (cov.)")
                continue
            # TODO: skip if num consts does not match?
            sub_data_ = c_["properties"]
            constants_ = sub_data_["ConstantNodes"]
            constant_values_ = sub_data_["ConstantValues"]
            io_sub_ = candidate_io_subs[j]
            # artifacts_ = c_["artifacts"]
            # flat_artifact_ = artifacts_["flat"]

            add_hash_attr(io_sub)
            add_hash_attr(io_sub_)
            add_hash_attr(io_sub, attr_name="hash_attr_ignore_const", ignore_const=True, handle_commutable=True)
            add_hash_attr(io_sub_, attr_name="hash_attr_ignore_const", ignore_const=True, handle_commutable=True)

            nm = iso.categorical_node_match("hash_attr_ignore_const", None)
            # em = iso.categorical_edge_match("hash_attr", None)
            em = categorical_edge_match_multidigraph("hash_attr", None)
            matcher = nx.algorithms.isomorphism.DiGraphMatcher(io_sub, io_sub_, node_match=nm, edge_match=em)
            check = matcher.is_isomorphic()
            if check:
                STOP = True
                # print("check", check)
                # print("======")
                # print("j", j)
                # print("i,j", i, j)
                # print("c_", c_)
                # print("sub_data_", sub_data_)
                # print("constants_", constants_)
                # print("constant_values_", constant_values_)
                # print("io_sub_", io_sub_)
                # print("flat_artifact_", flat_artifact_)
                # with open(flat_artifact, "r") as f:
                # flat = f.read()
                # with open(flat_artifact_, "r") as f:
                # flat_ = f.read()
                # print("flat", flat)
                # print("flat_", flat_)
                mapping = matcher.mapping
                # print("mapping", mapping)
                const_mapping = {}
                for c, constant in enumerate(constants):
                    constant_value = constant_values[c]
                    matching_constant = mapping[constant]
                    assert matching_constant in constants_
                    c_ = constants_.index(matching_constant)
                    matching_constant_value = constant_values_[c_]
                    if matching_constant_value == constant_value:
                        # print("cont (full iso)")
                        continue
                    const_value_subs[c][matching_constant_value].append(j)
                    const_mapping[constant] = f"{constant_value} -> {matching_constant_value}"
                # if len(const_mapping) > 0:
                #     print("const_mapping", const_mapping)
                #     # input(">")
                covered.add(j)
        if STOP:
            sub_const_value_subs[i] = const_value_subs
            # print("const_value_subs", const_value_subs)
            # input("2>")
    return sub_const_value_subs


def get_const_values_df(sub_const_value_subs, candidates):
    df_data = []
    # print("sub_const_value_subs", sub_const_value_subs)
    for i, const_value_subs in sub_const_value_subs.items():
        # print("i", i)
        # print("const_value_subs", const_value_subs)
        idx = candidates[i]["id"]
        num_constants = len(const_value_subs)
        # print("num_constants", num_constants)
        # values = [list(x.keys()) for x in const_value_subs.values()]
        diff_values = [list(x.keys()) for x in const_value_subs.values() if len(x) > 1]
        # print("diff_values", diff_values)
        num_diff_constants = len(diff_values)
        # print("num_diff_constants", num_diff_constants)
        if num_diff_constants == 0:
            continue
        signed = list(map(lambda x: any([y < 0 for y in x]), diff_values))
        # print("signed", signed)
        diff_abs_values = [[abs(y) for y in x] for x in diff_values]
        # print("diff_abs_values", diff_abs_values)
        diff_max_abs_values = list(map(max, diff_abs_values))
        # print("diff_max_abs_values", diff_max_abs_values)
        from math import ceil, log2

        # diff_min_bits = list(map(lambda x: ceil(log2(x + 1)), diff_max_values))
        diff_min_bits_ = list(map(lambda x: ceil(log2(x + 1)), diff_max_abs_values))
        # print("diff_min_bits_", diff_min_bits_)
        diff_sign_bits = [1 if s else 0 for s in signed]
        # print("diff_sign_bits", diff_sign_bits)
        diff_min_bits = [sum(x) for x in zip(diff_min_bits_, diff_sign_bits)]
        # print("diff_min_bits", diff_min_bits)
        extra_imm_bits = sum(diff_min_bits)
        # print("extra_imm_bits", extra_imm_bits)
        any_signed = any(signed)
        # print("any_signed", any_signed)
        # if any_signed:
        #     input("!!!")
        covered = set()
        for x in const_value_subs.values():
            for nodes in list(map(set, x.values())):
                covered.update(nodes)
        # print("covered", covered)
        covered_idx = set(map(lambda x: candidates[x]["id"], covered))
        num_covered = len(covered)
        # print("num_covered", num_covered)
        data = {
            "sub": i,
            "idx": idx,
            "num_constants": num_constants,
            "num_diff_constants": num_diff_constants,
            "diff_values": diff_values,
            "diff_signed": signed,
            "diff_abs_values": diff_abs_values,
            "diff_max_abs_values": diff_max_abs_values,
            "diff_min_bits": diff_min_bits,
            "extra_imm_bits": extra_imm_bits,
            "covered": covered,
            "covered_idx": covered_idx,
            "num_covered": num_covered,
        }
        # print("data", data)
        df_data.append(data)
    df = pd.DataFrame(df_data)
    return df


def main():
    args = handle_cmdline()
    INS = args.index
    assert len(INS) == 1
    in_path = INS[0]
    # OVERLAPS_OUT = args.overlaps
    # DROP_DUPLICATES = args.drop_duplicates
    IGNORE_COVERED = False
    # Not meaningful:
    # IGNORE_COVERED = True
    candidates, candidate_io_subs = load_candidates(in_path, progress=args.progress)

    sub_const_value_subs = get_const_value_subs(
        candidates, candidate_io_subs, ignore_covered=IGNORE_COVERED, progress=args.progress
    )
    df = get_const_values_df(sub_const_value_subs, candidates)
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df)
    input("3>")

    # temp = {"global": {"artifacts": [], "properties": global_properties}, "candidates": candidates}
    # if OUT:
    #     with open(OUT, "w") as f:
    #         yaml.dump(temp, f)
    # else:
    #     yaml_str = yaml.dump(temp)
    #     print(yaml_str)


if __name__ == "__main__":
    main()
