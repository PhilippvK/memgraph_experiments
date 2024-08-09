import logging
from typing import Union, List
from pathlib import Path
import argparse

import yaml
import pandas as pd
from tqdm import tqdm

from .cdsl_utils import CDSLEmitter, wrap_cdsl
from .tree_utils import tree_from_pkl
from .desc import generate_desc

logger = logging.getLogger("cdsl")


def generate_operands(sub_data: dict, xlen: int, with_attrs: bool = True):
    operands = {}
    operand_names = sub_data["OperandNames"]
    # operand_nodes = sub_data["OperandNodes"]
    operand_dirs = sub_data["OperandDirs"]
    operand_types = sub_data["OperandTypes"]  # TODO: rename to Classes?
    operand_enc_bits = sub_data["OperandEncBits"]
    operand_reg_classes = sub_data["OperandRegClasses"]
    for i, operand_name in enumerate(operand_names):
        operand_dir = operand_dirs[i]
        operand_type = operand_types[i]
        operand_reg_class = operand_reg_classes[i]
        is_output = operand_dir in ["OUT", "INOUT"]
        enc_bits = operand_enc_bits[i]
        if operand_type == "REG":
            if operand_reg_class == "GPR":
                operand_code = f"unsigned<{enc_bits}> {operand_name}"
                if with_attrs:
                    attrs = {
                        "is_reg": None,
                        "reg_class": operand_reg_class.upper(),
                        "reg_type": f"s{xlen}",
                        operand_dir.lower(): None,
                    }
                    attr_strs = [
                        f"[[{key}={value}]]" if value is not None else f"[[{key}]]" for key, value in attrs.items()
                    ]
                    attrs_str = " ".join(attr_strs)
                    operand_code += " " + attrs_str
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        operand = (operand_code, operand_reg_class.lower(), is_output)
        operands[operand_name] = operand
    return operands


def generate_cdsl(stmts_root, sub_data, xlen: int, name="result", desc=None):
    emitter = CDSLEmitter(xlen)
    emitter.visit(stmts_root)
    output = emitter.output
    codes = []
    codes += output.split("\n")
    operands = generate_operands(sub_data, xlen=xlen, with_attrs=True)

    # print("CDSL Code:")
    codes = ["    " + code for code in codes]
    asm_ins = []
    asm_outs = []
    mnemonic = "myinst"
    operands_code = "operands: {\n"
    for operand_name, operand in operands.items():
        operand_code, reg_class, is_output = operand
        asm_str = operand_name
        if reg_class == "gpr":
            asm_str = f"name({operand_name})"
        elif reg_class == "gpr":
            asm_str = f"fname({operand_name})"
        asm_str = f"{{{asm_str}}}"
        if is_output:
            asm_outs.append(asm_str)
        else:
            asm_ins.append(asm_str)
        operands_code += "    " + operand_code + ";\n"
    asm_all = asm_outs + asm_ins
    asm_syntax = ", ".join(asm_all)
    operands_code += "}"
    codes = (
        [operands_code, "encoding: auto;", f'assembly: {{"{mnemonic}", "{asm_syntax}"}};', "behavior: {"]
        + codes
        + ["}"]
    )
    codes = wrap_cdsl("MyInst", "\n".join(codes)).split("\n")
    codes = ["    " * 2 + code for code in codes]
    code = "\n".join(codes) + "\n"
    prefix = f'import "RV{xlen}I.core_desc"\n\n'
    code = (
        prefix
        + f"""InstructionSet MySet extends RV{xlen}I {{
    instructions {{
{code}
    }}
}}
"""
    )
    if desc:
        code = f"// {desc}\n\n" + code
    return code.strip()


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
        cdsl_code = generate_cdsl(stmts, sub_data, xlen=xlen, name=name, desc=desc)
        print("cdsl_code")
        print(cdsl_code)
        with open(out_dir / f"{name}.core_desc", "w") as f:
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
