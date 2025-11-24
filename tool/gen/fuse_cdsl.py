import logging
from typing import Union, List
from pathlib import Path
import argparse

import yaml
import pandas as pd
from tqdm import tqdm

from .flat_utils import FlatCodeEmitter
from .tree_utils import tree_from_pkl
from .desc import generate_desc
from .cdsl import generate_operands, generate_complete_cdsl
from .gen_utils import gen_helper

logger = logging.getLogger("fuse_cdsl")


def generate_fuse_cdsl(stmts_root, sub_data, xlen: int, name="result", desc=None):
    codes = []
    emitter = FlatCodeEmitter()
    emitter.visit(stmts_root)
    output = emitter.output
    codes += output.split("\n")
    operands = generate_operands(sub_data, xlen=xlen, with_attrs=True)

    return generate_complete_cdsl(codes, operands=operands, xlen=xlen, name=name, desc=desc)


def get_global_df(global_properties: List[dict]):
    return pd.DataFrame(global_properties)


def process_candidate_fuse_cdsl(idx, candidate_data, xlen=None, out_dir=None):
    candidate_properties = candidate_data["properties"]
    candidate_artifacts = candidate_data["artifacts"]
    name = candidate_properties.get("InstrName")
    if name is None:
        name = f"name{i}"
    sub_data = candidate_properties
    desc = generate_desc(idx, sub_data, name=name)
    tree_pkl = candidate_artifacts.get("tree", None)
    assert tree_pkl is not None
    stmts = tree_from_pkl(tree_pkl)
    # TODO: make sure that result/sub col in combined index is unique
    cdsl_code = generate_fuse_cdsl(stmts, sub_data, xlen=xlen, name=name, desc=desc)
    out_file = out_dir / f"{name}.fuse_core_desc"
    with open(out_file, "w") as f:
        f.write(cdsl_code)
    # TODO: Status = GENERATED?
    return out_file


def process(
    index_path,
    out_dir: Union[str, Path],
    inplace: bool = True,
    split: bool = True,
    split_files: bool = True,
    progress: bool = False,
    n_parallel: int = 1,
):
    gen_helper(
        "fuse_cdsl",
        process_candidate_fuse_cdsl,
        index_path,
        out_dir=out_dir,
        inplace=inplace,
        split=split,
        split_files=split_files,
        progress=progress,
        n_parallel=n_parallel,
    )


def handle_cmdline():
    # TODO: add help messages
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("index", help="TODO")  # print if None
    parser.add_argument("--log", default="info", choices=["critical", "error", "warning", "info", "debug"], help="TODO")
    parser.add_argument("--output", "-o", default=None, help="TODO")  # print if None
    parser.add_argument("--inplace", action="store_true", help="TODO")
    parser.add_argument("--split", action="store_true", help="TODO")  # one instr per set
    parser.add_argument("--split-files", action="store_true", help="TODO")  # one file per set
    parser.add_argument("--progress", action="store_true", help="TODO")
    parser.add_argument("--parallel", type=int, default=1, help="TODO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()))
    return args


def main():
    args = handle_cmdline()
    process(
        args.index,
        args.output,
        inplace=args.inplace,
        split=args.split,
        split_files=args.split_files,
        progress=args.progress,
        n_parallel=args.parallel,
    )


if __name__ == "__main__":
    main()
