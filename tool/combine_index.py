import pickle
import logging
import argparse
from collections import defaultdict

import yaml
from tqdm import tqdm


from .iso import calc_io_isos

logger = logging.getLogger("combine_index")


def handle_cmdline():
    # TODO: add help messages
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("index", nargs="+", help="TODO")  # print if None
    parser.add_argument("--log", default="info", choices=["critical", "error", "warning", "info", "debug"], help="TODO")
    parser.add_argument("--output", "-o", default=None, help="TODO")  # print if None
    parser.add_argument("--drop-duplicates", action="store_true", help="TODO")
    parser.add_argument("--venn", default=None, help="TODO")
    parser.add_argument("--progress", action="store_true", help="TODO")
    parser.add_argument("--sort-by", type=str, default=None, help="TODO")
    parser.add_argument("--sort-asc", action="store_true", help="TODO")
    parser.add_argument("--topk", type=int, default=None, help="Only keep the k first items")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()))
    return args


args = handle_cmdline()
INS = args.index
OUT = args.output
DROP_DUPLICATES = args.drop_duplicates
VENN_OUT = args.venn

i = 0

candidates = []
candidate_io_subs = []
venn_data = []
global_properties = []

for j, in_path in enumerate(INS):
    logger.info("Loading input %s", in_path)
    with open(in_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    global_data = yaml_data["global"]
    global_properties_ = global_data["properties"]
    global_properties += global_properties_
    candidates_data = yaml_data["candidates"]
    path_ids = set()
    for candidate_data in tqdm(candidates_data, disable=not args.progress):
        candidate_data["id"] = i
        artifacts = candidate_data["artifacts"]
        properties = candidate_data["properties"]
        path_ids.add(i)
        candidates.append(candidate_data)
        if DROP_DUPLICATES:
            io_sub_path = artifacts.get("io_sub", None)
            assert io_sub_path is not None, "PKL for io_sub not found in artifacts"
            with open(io_sub_path, "rb") as f:
                io_sub = pickle.load(f)
            candidate_io_subs.append(io_sub)
        i += 1
    venn_data.append(path_ids)

if DROP_DUPLICATES:
    duplicates = defaultdict(set)
    duplicate_count = 0
    logger.info("Detecting duplicates...")
    _, sub_io_isos = calc_io_isos(candidate_io_subs, progress=args.progress)
    for sub, io_isos_ in sub_io_isos.items():
        if len(io_isos_) > 0:
            duplicates[sub] = io_isos_
            duplicate_count += len(io_isos_)
            for k in range(len(venn_data)):
                venn_data[k] = set(i if k2 in io_isos_ else k2 for k2 in venn_data[k])
        # input("@@")
    all_duplicates = set(sum(map(list, duplicates.values()), []))
    candidates = [x for i, x in enumerate(candidates) if i not in all_duplicates]
    # TODO: fix ids?

if VENN_OUT is not None:
    from matplotlib_venn import venn3, venn2
    from matplotlib_venn.layout.venn3 import DefaultLayoutAlgorithm
    from matplotlib import pyplot as plt

    assert len(venn_data) > 0, "--venn needs --drop"
    assert len(venn_data) in [2, 3], "--venn only works for 2 or 3 inputs"
    if len(venn_data) == 3:
        fig = venn3(
            venn_data,
            ("Set1", "Set2", "Set3"),
            layout_algorithm=DefaultLayoutAlgorithm(fixed_subset_sizes=(1, 1, 1, 1, 1, 1, 1)),
        )
    elif len(venn_data) == 2:
        fig = venn2(venn_data, ("Set1", "Set2"))
    plt.savefig(VENN_OUT)

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

if args.sort_by:
    assert isinstance(args.sort_by, str)
    candidates = sorted(candidates, key=lambda x: x["properties"][args.sort_by], reverse=not args.sort_asc)

if args.topk:
    assert args.sort_by, "--topk is only meaningful on sorted list"
    assert isinstance(args.topk, int)
    candidates = candidates[:args.topk]

temp = {"global": {"artifacts": [], "properties": global_properties}, "candidates": candidates}
if OUT:
    with open(OUT, "w") as f:
        yaml.dump(temp, f)
else:
    yaml_str = yaml.dump(temp)
    print(yaml_str)
