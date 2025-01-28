import pickle
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import yaml
import pandas as pd
from tqdm import tqdm


from .iso import calc_io_isos

logger = logging.getLogger("filter_index")


def handle_cmdline():
    # TODO: add help messages
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("index", help="Index yaml file")
    parser.add_argument("--log", default="info", choices=["critical", "error", "warning", "info", "debug"], help="TODO")
    parser.add_argument("--output", "-o", default=None, help="Output index yaml file. Use '-' to print")
    parser.add_argument("--drop", "-d", default=None, nargs="+", help="Candidate ids/names to drop")
    parser.add_argument("--keep", "-k", default=None, nargs="+", help="Candidate ids/names to keep")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()))
    return args


def main():
    args = handle_cmdline()
    IN = args.index
    DROP = args.drop
    KEEP = args.keep
    OUT = args.output

    logger.info("Loading input %s", IN)
    with open(IN, "r") as f:
        yaml_data = yaml.safe_load(f)
    candidates_data = yaml_data["candidates"]

    def resolve_idxs(data):
        ret = set()

        def convert(x):
            if isinstance(x, int):
                return x
            if isinstance(x, str):
                x = x.lower()
                if x.isdigit():
                    return int(x)
                if x.startswith("custom"):
                    rest = x[6:]
                    return convert(rest)
                if x.startswith("name"):
                    rest = x[4:]
                    return convert(rest)
                raise ValueError(f"Unkown identifier: {x}")
            raise ValueError(f"Unsupported type: {type(x)}")

        for x in data:
            new = convert(x)
            ret.add(new)
        return ret

    idxs = set(range(len(candidates_data)))

    if KEEP:
        keep_idxs = resolve_idxs(KEEP)
        idxs = keep_idxs

    if DROP:
        drop_idxs = resolve_idxs(DROP)
        idxs -= drop_idxs

    candidates_data = [data for i, data in enumerate(candidates_data) if i in idxs]

    yaml_data["candidates"] = candidates_data

    if OUT is None:
        OUT = IN
    if OUT != "-":
        with open(OUT, "w") as f:
            yaml.dump(yaml_data, f)
    else:
        yaml_str = yaml.dump(yaml_data)
        print(yaml_str)

if __name__ == "__main__":
    main()
