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


def process(
    index_path,
    out_dir: Union[str, Path],
    inplace: bool = True,
    split: bool = True,
    split_files: bool = True,
    progress: bool = False,
):
    if not split:
        raise NotImplementedError("--no-split")
    if not split_files:
        raise NotImplementedError("--no-split-files")
    if isinstance(out_dir, str):
        out_dir = Path(out_dir)
    assert isinstance(out_dir, Path)
    assert out_dir.is_dir()
    logger.info("Loading input %s", index_path)
    with open(index_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    global_data = yaml_data["global"]
    global_properties = global_data["properties"]
    # print("global_properties", global_properties)
    global_df = get_global_df(global_properties)
    global_artifacts = global_data["artifacts"]
    # print("global_artifacts", global_artifacts)
    xlens = global_df["xlen"].unique()
    assert len(xlens) == 1
    xlen = int(xlens[0])
    candidates_data = yaml_data["candidates"]
    for i, candidate_data in tqdm(enumerate(candidates_data), disable=not progress):
        candidate_properties = candidate_data["properties"]
        # print("candidate_properties", candidate_properties)
        candidate_artifacts = candidate_data["artifacts"]
        # print("candidate_artifacts", candidate_artifacts)
        name = f"name{i}"
        print("name", name)
        sub_data = candidate_properties
        desc = generate_desc(i, sub_data, name=name)
        tree_pkl = candidate_artifacts.get("tree", None)
        assert tree_pkl is not None
        # print("tree_pkl", tree_pkl)
        stmts = tree_from_pkl(tree_pkl)
        # print("stmts", stmts)
        # TODO: make sure that result/sub col in combined index is unique
        cdsl_code = generate_fuse_cdsl(stmts, sub_data, xlen=xlen, name=name, desc=desc)
        print("cdsl_code")
        print(cdsl_code)
        with open(out_dir / f"{name}.fuse_core_desc", "w") as f:
            f.write(cdsl_code)
        candidate_artifacts["cdsl"] = str(out_dir / f"{name}.core_desc")
        yaml_data["candidates"][i]["artifacts"] = candidate_artifacts
        # TODO: Status = GENERATED?
    if inplace:
        out_index_path = index_path
    else:
        out_index_path = out_dir / "index.yml"
    with open(out_index_path, "w") as f:  # TODO: reuse code from index.py
        yaml.dump(yaml_data, f, default_flow_style=False)


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
    )


if __name__ == "__main__":
    main()
