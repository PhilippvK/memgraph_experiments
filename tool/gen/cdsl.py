import logging
from typing import Union, List
from pathlib import Path
import argparse

import pandas as pd
from anytree import AnyNode, RenderTree

from .cdsl_utils import CDSLEmitter, wrap_cdsl
from .tree_utils import tree_from_pkl
from .desc import generate_desc
from .gen_utils import gen_helper

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

        def convert_val(x):
            if isinstance(x, str):
                return f'"{x}"'
            return x

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
                        f"[[{key}={convert_val(value)}]]" if value is not None else f"[[{key}]]"
                        for key, value in attrs.items()
                    ]
                    attrs_str = " ".join(attr_strs)
                    operand_code += " " + attrs_str
            else:
                raise NotImplementedError
        elif operand_type == "IMM":
            # TODO: store sign info
            signed = operand_name[0] == "s"
            dtype = "signed" if signed else "unsigned"
            operand_code = f"{dtype}<{enc_bits}> {operand_name}"
            if with_attrs:
                attrs = {
                    "is_imm": None,
                    operand_dir.lower(): None,
                }

                attr_strs = [
                    f"[[{key}={convert_val(value)}]]" if value is not None else f"[[{key}]]"
                    for key, value in attrs.items()
                ]
                attrs_str = " ".join(attr_strs)
                operand_code += " " + attrs_str
            operand_reg_class = "IMM"
        else:
            raise NotImplementedError

        operand = (operand_code, operand_reg_class.lower(), is_output)
        operands[operand_name] = operand
    return operands


def generate_complete_cdsl(codes: List[str], operands, xlen: int, name="result", desc=None):
    # print("CDSL Code:")
    codes = ["    " + code for code in codes]
    asm_ins = []
    asm_outs = []
    if name is None:
        name = "MyInst"
    mnemonic = name.lower()
    operands_code = "operands: {\n"
    for operand_name, operand in operands.items():
        operand_code, reg_class, is_output = operand
        asm_str = operand_name
        if reg_class == "imm":
            pass
        elif reg_class == "gpr":
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
    codes = wrap_cdsl(name, "\n".join(codes)).split("\n")
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


def generate_cdsl(stmts_root, sub_data, xlen: int, name="result", desc=None):
    print("stmts_root", stmts_root)
    print(RenderTree(stmts_root))
    # input(">>>")
    emitter = CDSLEmitter(xlen)
    emitter.visit(stmts_root)
    output = emitter.output
    codes = []
    codes += output.split("\n")
    operands = generate_operands(sub_data, xlen=xlen, with_attrs=True)
    return generate_complete_cdsl(codes, operands=operands, xlen=xlen, name=name, desc=desc)


def process_candidate_cdsl(idx, candidate_data, xlen=None, out_dir=None):
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
    cdsl_code = generate_cdsl(stmts, sub_data, xlen=xlen, name=name, desc=desc)
    out_file = out_dir / f"{name}.core_desc"
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
        "cdsl",
        process_candidate_cdsl,
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
