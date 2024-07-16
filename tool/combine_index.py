import pickle
import logging
import argparse
from collections import defaultdict

import yaml
import networkx as nx

logger = logging.getLogger("combine_index")


def handle_cmdline():
    # TODO: add help messages
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("index", nargs="+", help="TODO")  # print if None
    parser.add_argument("--log", default="info", choices=["critical", "error", "warning", "info", "debug"], help="TODO")
    parser.add_argument("--output", "-o", default=None, help="TODO")  # print if None
    parser.add_argument("--drop-duplicates", action="store_true", help="TODO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()))
    return args


args = handle_cmdline()
INS = args.index
OUT = args.output
DROP_DUPLICATES = args.drop_duplicates

i = 0

candidates = []
candidate_io_subs = []

for in_path in INS:
    print("in_path", in_path)
    with open(in_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    candidates_data = yaml_data["candidates"]
    # print("candidates_data", candidates_data, len(candidates_data))
    for candidate_data in candidates_data:
        candidate_data["id"] = i
        candidates.append(candidate_data)
        if DROP_DUPLICATES:
            io_sub_path = candidate_data["artifacts"].get("io_sub", None)
            assert io_sub_path is not None, "PKL for io_sub not found in artifacts"
            with open(io_sub_path, "rb") as f:
                io_sub = pickle.load(f)
            candidate_io_subs.append(io_sub)
        i += 1

if DROP_DUPLICATES:
    duplicates = defaultdict(set)
    duplicate_count = 0
    # print("candidate_io_subs", candidate_io_subs)
    # print("candidate_io_subs", [(str(x), x.nodes) for x in candidate_io_subs])
    # input("A")
    for i, io_sub in enumerate(candidate_io_subs):
        def node_match(x, y):
            # print("node_match")
            # print("x", x)
            # print("y", y)
            # input("?")
            # TODO: check xlabel? (Const replaced with val!)
            return x["label"] == y["label"] and (
                x["label"] != "Const" or x["properties"]["inst"] == y["properties"]["inst"]
            )

        io_isos_ = set(
            j for j, io_sub_ in enumerate(candidate_io_subs) if j > i and nx.is_isomorphic(io_sub, io_sub_, node_match=node_match)
        )
        # TODO: do not check in same index?
        # print("io_isos_", io_isos_, len(io_isos_))
        if len(io_isos_) > 0:
            duplicates[i] = io_isos_
            duplicate_count += len(io_isos_)
        # input("@@")
    print("duplicates", duplicates)
    print("duplicate_count", duplicate_count)
    # TODO: fix ids?
    input(">>")
# print("candidates", candidates, len(candidates))

# with MeasureTime("Isomorphism Check", verbose=TIMES):
#     logger.info("Checking isomorphism...")
#     # print("io_subs", [str(x) for x in io_subs], len(io_subs))
#     io_isos = set()
#     for i, io_sub in enumerate(tqdm(io_subs, disable=not PROGRESS)):
#         # break  # TODO
#         # print("io_sub", i, io_sub, io_sub.nodes)
#         # print("io_sub nodes", [GF.nodes[n] for n in io_sub.nodes])
#         # print("io_isos_", io_isos_)
#         io_iso_count = len(io_isos_)
#         # print("io_iso_count", io_iso_count)
#         io_isos |= io_isos_
#     # print("subs_df")
#     # print(subs_df)
#     # print("io_isos", io_isos, len(io_isos))
#     subs_df.loc[list(io_isos), "Status"] = ExportFilter.ISO

# logger.info("Writing Combined Index File...")
# write_index_file(OUT / "index.yml", filtered_subs_df, index_data)
temp = {"candidates": candidates}
if OUT:
    with open(OUT, "w") as f:
        yaml.dump(temp, f)
else:
    yaml_str = yaml.dump(temp)
    print(yaml_str)
