import logging
import argparse

from .cdsl_utils import CDSLEmitter, wrap_cdsl

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

        operand_code = ""
        operand = (operand_code, operand_reg_class.lower(), is_output)
        operands[operand_name] = operand
    print("operands123", operands)
    input(">>>")
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
        code = "// {desc}\n\n" + code
    return code


def handle_cmdline():
    # TODO: add help messages
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("index", help="TODO")  # print if None
    parser.add_argument("--log", default="info", choices=["critical", "error", "warning", "info", "debug"], help="TODO")
    parser.add_argument("--output", "-o", default=None, help="TODO")  # print if None
    parser.add_argument("--split", action="store_true", help="TODO")  # one instr per set
    parser.add_argument("--split-files", action="store_true", help="TODO")  # one file per set
    parser.add_argument("--progress", action="store_true", help="TODO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()))
    return args


def main():
    args = handle_cmdline()
    print("args", args)
    input(">>>")


if __name__ == "__main__":
    main()
