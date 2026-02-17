from enum import IntEnum, auto

from .tree_utils import Ref, Cast, Constant, Register, Operation, Statements, Declaration, Assignment
from .utils import mem_lookup


def wrap_cdsl(name, code):
    ret = f"{name} {{\n"
    ret += "\n".join(["    " + line for line in code.splitlines()]) + "\n"
    ret += "}\n"
    return ret


class ExtendMode(IntEnum):
    UNDEFINED = auto()
    SIGN_EXTEND = auto()
    ZERO_EXTEND = auto()


class CDSLEmitter:

    def __init__(self, xlen):
        assert xlen in [32, 64]
        self.xlen = xlen
        self.output = ""

    def write(self, text):
        if not isinstance(text, str):
            text = str(text)
        self.output += text

    def visit_ref(self, node):
        assert isinstance(node, Ref)
        self.write(node.name)

    def visit_constant(self, node):
        assert isinstance(node, Constant)
        self.write("(")
        self.write(node.value)  # TODO: dtype
        self.write(")")

    def visit_constant_generic(self, node):
        assert isinstance(node, Constant)
        assert len(node.children) == 1
        # print("node", node)
        # print("node.name", node.name)
        # print("node.in_types", node.in_types)
        # print("node.out_types", node.out_types)
        # input("!!")
        self.visit(node.children[0])

    def visit_trunc_generic(self, node):
        assert isinstance(node, Operation)
        # print("visit_trunc_generic")
        # print("node", node)
        # print("node.children", node.children)
        out_types = node.out_types
        assert len(out_types) == 1
        out_type = out_types[0]
        assert out_type is not None
        self.write("(")
        self.write(out_type)  # TODO: do explicit trunc here with slice?
        self.write(")")
        assert len(node.children) == 1
        self.visit(node.children[0])

    def visit_ext_generic(self, node):
        assert isinstance(node, Operation)
        # print("visit_ext_generic")
        # print("node", node)
        # print("node.children", node.children)
        lookup = {
            "G_ANYEXT": (ExtendMode.UNDEFINED,),
            "G_SEXT": (ExtendMode.SIGN_EXTEND,),
            "G_ZEXT": (ExtendMode.ZERO_EXTEND,),
        }
        res = lookup.get(node.name)
        assert res is not None
        (mode,) = res
        in_types = node.in_types
        assert len(in_types) == 1
        in_type = in_types[0]  # TODO: need to get the bit size here...
        if mode == ExtendMode.SIGN_EXTEND:
            if in_type.startswith("unsigned"):
                in_type = in_type.replace("unsigned", "signed")
        elif mode == ExtendMode.ZERO_EXTEND:
            if in_type.startswith("signed"):
                in_type = in_type.replace("signed", "unsigned")
        out_types = node.out_types
        assert len(out_types) == 1
        out_type = out_types[0]
        assert out_type is not None
        self.write("(")
        self.write(out_type)  # TODO: do explicit trunc here with slice?
        self.write(")")
        self.write("(")
        self.write("(")
        self.write(in_type)
        self.write(")")
        assert len(node.children) == 1
        self.visit(node.children[0])
        self.write(")")

    def visit_register(self, node):
        assert isinstance(node, Register)
        assert len(node.children) == 1
        idx = node.children[0]
        reg_class = node.reg_class
        mem_name = mem_lookup.get(reg_class, None)
        assert mem_name is not None, f"Unable to find mem_name for reg_class: {reg_class}"
        self.write(mem_name)
        self.write("[")
        self.visit(idx)
        self.write("]")

    def visit_statements(self, node):
        assert isinstance(node, Statements)
        # print("visit_statements", node, node.children)
        for children in node.children:
            self.visit(children)
            self.write("\n")

    def visit_assignment(self, node):
        assert isinstance(node, (Assignment, Declaration))
        # print("visit_assignment", node, node.children)
        assert len(node.children) == 2
        lhs, rhs = node.children
        self.visit(lhs)
        self.write("=")
        self.visit(rhs)
        self.write(";")

    def visit_declaration(self, node):
        assert isinstance(node, Declaration)
        # print("visit_declaration", node, node.children)
        assert len(node.children) == 2
        lhs, rhs = node.children
        decl_type = node.decl_type
        self.write(f"{decl_type} ")
        self.visit_assignment(node)

    def visit_cast(self, node):
        assert isinstance(node, Cast)
        # print("visit_assignment", node, node.children)
        # print("node", node, dir(node))
        assert len(node.children) == 1
        lhs = node.children[0]
        to_type = node.to
        self.write(f"({to_type})")
        self.write("(")
        self.visit(lhs)
        self.write(")")

    def visit_branch_riscv(self, node):
        assert isinstance(node, Operation)
        lookup = {
            "BNE": ("!=", False),
            "BEQ": ("==", False),
        }
        res = lookup.get(node.name)
        assert res is not None
        pred, imm = res
        assert not imm, "Immediated branching not supported"
        # print("node.children", node.children)
        assert len(node.children) == 2  # TODO: fix missing offset label
        lhs, rhs = node.children
        # TODO: check alignment
        self.write("if(")
        self.visit(lhs)
        self.write(pred)
        self.visit(rhs)
        self.write(")")
        self.write("{")
        self.write("PC")
        self.write("=")
        self.write("PC")
        self.write("+")
        self.write("(signed)")
        # self.visit(offset)
        self.write("TODO")
        self.write("}")

    def visit_load_generic(self, node):
        assert isinstance(node, Operation)

        lookup = {
            "G_LOAD": (ExtendMode.UNDEFINED,),
            "G_SEXTLOAD": (ExtendMode.SIGN_EXTEND,),
            "G_ZEXTLOAD": (ExtendMode.ZERO_EXTEND,),
        }
        res = lookup.get(node.name)
        assert res is not None
        (mode,) = res
        # TODO: dtypes?
        # in_types = node.in_types
        # assert len(in_types) == 1
        # in_type = in_types[0]  # TODO: need to get the bit size here...
        mem_type = "unsigned<?>"
        out_types = node.out_types
        assert len(out_types) == 1
        out_type = out_types[0]
        assert out_type is not None
        # print("node.children", node.children)
        assert len(node.children) == 1
        (base,) = node.children
        self.write(f"({out_type})")
        self.write("(")
        if mode == ExtendMode.SIGN_EXTEND:
            if mem_type.startswith("unsigned"):
                mem_type_ = mem_type.replace("unsigned", "signed")
                self.write(f"({mem_type_})")
        elif mode == "ExtendMode.ZERO_EXTEND":
            if mem_type.startswith("signed"):
                mem_type_ = mem_type.replace("signed", "unsigned")
                self.write(f"({mem_type_})")
        else:
            pass  # TODO: ok?
        self.write(f"({mem_type})")
        self.write("MEM[")
        self.visit(base)
        self.write("+")
        # self.visit(offset)
        self.write("?")
        self.write("]")
        self.write(")")

    def visit_load_riscv(self, node):
        assert isinstance(node, Operation)
        lookup = {
            "LH": (True, 16, True),
            "LW": (True, 32, True),
        }
        res = lookup.get(node.name)
        assert res is not None
        signed, sz, imm = res
        assert imm, "reg-reg loads not supported"
        assert len(node.children) == 2
        base, offset = node.children
        self.write("(unsigned<XLEN>)")
        self.write("(")
        if signed:
            self.write(f"(signed<{sz}>)")
        else:
            self.write(f"(unsigned<{sz}>)")
        self.write("MEM[")
        self.visit(base)
        self.write("+")
        self.visit(offset)
        self.write("]")
        self.write(")")

    def visit_store_riscv(self, node):
        assert isinstance(node, Operation)
        lookup = {
            "SW": (32, True),
            "SH": (16, True),
        }
        res = lookup.get(node.name)
        assert res is not None
        sz, imm = res
        assert imm, "reg-reg stores not supported"
        assert len(node.children) == 3
        value, base, offset = node.children
        self.write("MEM[")
        self.visit(base)
        self.write("+")
        self.visit(offset)
        self.write("]")
        self.write("=")
        self.write("(")
        self.write(f"(signed<{sz}>)")
        self.visit(value)
        self.write(")")
        self.write(";")

    def visit_lui_riscv(self, node):
        assert isinstance(node, Operation)
        raise NotImplementedError("LUI")
        self.write("lui(TODO)")

    def visit_binop_generic(self, node):
        assert isinstance(node, Operation)
        # TODO: imm needed?
        # print("visit_binop_generic", node)
        lookup = {
            "G_ADD": ("+", True),  # %dst:_(s32) = G_ADD %src0:_(s32), %src1:_(s32)
            "G_MUL": ("*", True),
            "G_PTR_ADD": ("+", True),  # %1:_(p0) = G_PTR_ADD %0:_(p0), %1:_(s32)
        }
        res = lookup.get(node.name)
        assert res is not None
        op, signed = res
        # TODO: dtypes
        self.write("(")
        assert len(node.children) == 2
        lhs, rhs = node.children
        # print("lhs", lhs)
        # print("rhs", rhs)
        # TODO: add size only of != xlen?
        assert len(node.in_types) == 2
        assert len(node.out_types) == 1
        assert node.in_types[0] == node.in_types[1]
        # print("node", node)
        assert node.in_types[0] == node.out_types[0]
        sz = 0
        if signed:
            self.write(f"(signed<{sz}>)")
        else:
            self.write(f"(unsigned<{sz}>)")
        self.visit(lhs)
        self.write(op)
        if signed:
            self.write(f"(signed<{sz}>)")
        else:
            self.write(f"(unsigned<{sz}>)")
        self.visit(rhs)
        self.write(")")

    def visit_binop_riscv(self, node):
        assert isinstance(node, Operation)
        # TODO: imm needed?
        lookup = {
            "ADD": ("+", True, True, self.xlen, False, 0),
            "ADDW": ("+", True, True, 32, False, 0),
            "ADDI": ("+", True, True, self.xlen, True, 0),
            "ADDIW": ("+", True, True, 32, True, 0),
            "SUB": ("-", True, True, self.xlen, False, 0),
            "SUBW": ("-", True, True, 32, False, 0),
            "SRA": (">>", True, True, self.xlen, False, 0),
            "SRAI": (">>", True, True, self.xlen, True, 0),
            "SRLI": (">>", False, False, self.xlen, True, 0),
            "SRL": (">>", False, False, self.xlen, False, 0),
            "SLL": ("<<", False, False, self.xlen, False, 0),
            "SLLI": ("<<", False, False, self.xlen, True, 0),
            "AND": ("&", False, False, self.xlen, False, 0),
            "OR": ("|", False, False, self.xlen, False, 0),
            "XOR": ("^", False, False, self.xlen, False, 0),
            "XORI": ("^", False, False, self.xlen, True, 0),
            "ANDI": ("&", False, False, self.xlen, True, 0),
            "ORI": ("|", False, False, self.xlen, True, 0),
            "MULW": ("*", True, True, 32, False, 0),
            "MUL": ("*", True, True, self.xlen, False, 0),
            "MULH": ("*", True, True, self.xlen, False, self.xlen),
            "MULHSU": ("*", True, False, self.xlen, False, self.xlen),
            "MULHU": ("*", False, False, self.xlen, False, self.xlen),
            "DIVU": ("/", False, False, self.xlen, False, 0),
            "REMU": ("%", False, False, self.xlen, False, 0),
        }
        res = lookup.get(node.name)
        assert res is not None
        op, lhs_signed, rhs_signed, sz, imm, shift = res
        # TODO: check imm (sign/sz?)
        # TODO: dtype
        self.write("(")
        assert len(node.children) == 2
        lhs, rhs = node.children
        # TODO: add size only of != xlen?
        if shift > 0:
            assert shift == 32
            if lhs_signed:
                self.write(f"(signed<{sz}>)")
            else:
                self.write(f"(unsigned<{sz}>)")
            self.write("(")
            if lhs_signed:
                self.write(f"(signed<{sz * 2}>)")
            else:
                self.write(f"(unsigned<{sz * 2}>)")
            self.write("(")
        if lhs_signed:
            self.write(f"(signed<{sz}>)")
        else:
            self.write(f"(unsigned<{sz}>)")
        self.visit(lhs)
        self.write(op)
        if rhs_signed:
            self.write(f"(signed<{sz}>)")
        else:
            self.write(f"(unsigned<{sz}>)")
        self.visit(rhs)
        if shift > 0:
            self.write(")")
            self.write(">>")
            self.write(shift)
            self.write(")")
        self.write(")")

    def visit_pseudo_rvv_instr(self, node):
        assert isinstance(node, Operation)
        instr = node.name
        # print("instr", instr)
        lookup = {
            "PseudoVXOR_VV_M1": ("vxor_vv", 2),
            "PseudoVAND_VV_M1": ("vand_vv", 2),
            "PseudoVSLIDEUP_VI_M1": ("vslideup_vi", 2),
            "PseudoVSLIDE1DOWN_VX_M1": ("vslideup_vx", 2),
            "PseudoVSRL_VX_M1": ("vslr_vx", 2),
            "PseudoVMV_X_S": ("PseudoVMV_X_S", 1),
        }
        res = lookup.get(node.name)
        assert res is not None
        (func, num_inputs) = res
        # print("func", func)
        # print("node.children", node.children, len(node.children))
        children = node.children
        # print("children", children, len(children))
        assert len(children) == num_inputs
        self.write(func)
        self.write("(")
        for i, child in enumerate(children):
            if i > 0:
                self.write(", ")
            self.visit(child)
        # input(">>>")
        # TODO: how how to access destination register?
        # TODO: use vreg idx instead of real data?
        # TODO: handle status
        # TODO: support vi and vx
        # TODO: extract vtype?
        # self.visit_rvv_reg(lhs)
        # self.visit_rvv_reg(rhs)
        self.write(")")

    def visit_binop_riscv_divu_remu(self, node):
        lhs, rhs = node.children
        self.write("(")
        self.write("(")
        self.visit(rhs)
        self.write("!=")
        self.write("0")
        self.write(")")
        self.write("?")
        self.visit_binop_riscv(node)
        self.write(":")
        if node.name == "DIVU":
            self.write(-1)
        elif node.name == "REMU":
            self.visit(lhs)
        else:
            raise NotImplementedError(f"Op: {node.name}")
        self.write(")")
        assert len(node.children) == 2

    def visit_binop_riscv_div_rem(self, node):
        raise NotImplementedError("Op: DIV/REM")

    def visit_cond_set_riscv(self, node):
        # raise NotImplementedError("SLTU/SLTIU breaks hls flow")
        assert isinstance(node, Operation)
        lookup = {"SLT": ("<", True), "SLTU": ("<", False), "SLTIU": ("<", False), "SLTI": ("<", "True")}
        res = lookup.get(node.name)
        assert res is not None
        pred, signed = res
        assert len(node.children) == 2
        lhs, rhs = node.children
        self.write("(")
        if signed:
            self.write("(signed)")
        self.visit(lhs)
        self.write(pred)
        if signed:
            self.write("(signed)")
        self.visit(rhs)
        self.write("?")
        self.write(1)
        self.write(":")
        self.write(0)
        self.write(")")

    def visit_operation(self, node):
        assert isinstance(node, Operation)
        name = node.name
        # print("name", name)
        if name in [
            "ADDIW",
            "SRLI",
            "SLLI",
            "AND",
            "ANDI",
            "ORI",
            "OR",
            "XOR",
            "XORI",
            "ADD",
            "ADDI",
            "ADDW",
            "MULW",
            "MUL",
            "MULH",
            "MULHSU",
            "MULHU",
            "SRA",
            "SRAI",
            "SRL",
            "SLL",
            "SUB",
            "SUBW",
        ]:
            self.visit_binop_riscv(node)
        elif name in [
            "DIVU",
            "REMU",
        ]:
            self.visit_binop_riscv_divu_remu(node)
        elif name in [
            "DIV",
            "REM",
        ]:
            self.visit_binop_riscv_div_rem(node)
        elif name in ["SLT", "SLTU", "SLTIU", "SLTI"]:
            self.visit_cond_set_riscv(node)
        elif name in ["BNE", "BEQ"]:
            self.visit_branch_riscv(node)
        elif name in ["LH", "LW"]:
            self.visit_load_riscv(node)
        elif name in ["SW", "SH"]:
            self.visit_store_riscv(node)
        elif name in ["LUI"]:
            self.visit_lui_riscv(node)
        elif name in ["G_CONSTANT"]:
            self.visit_constant_generic(node)
        elif name in ["G_TRUNC"]:
            self.visit_trunc_generic(node)
        elif name in ["G_ANYEXT", "G_SEXT", "G_ZEXT"]:
            self.visit_ext_generic(node)
        elif name in [
            "G_ADD",
            "G_PTR_ADD",
            "G_MUL",
        ]:
            self.visit_binop_generic(node)
        elif name in ["G_LOAD", "G_SEXTLOAD", "G_ZEXTLOAD"]:
            self.visit_load_generic(node)
        elif name in [
            "PseudoVXOR_VV_M1",
            "PseudoVAND_VV_M1",
            "PseudoVSLIDEUP_VI_M1",
            "PseudoVSLIDE1DOWN_VX_M1",
            "PseudoVSRL_VX_M1",
            "PseudoVMV_X_S",
        ]:
            self.visit_pseudo_rvv_instr(node)
        else:
            raise NotImplementedError(f"Unhandled: {name}")

    def visit(self, node):
        # print("visit", node)
        # op_type = node.op_type
        # print("op_type", op_type)
        if isinstance(node, Statements):
            self.visit_statements(node)
        elif isinstance(node, Assignment):
            self.visit_assignment(node)
        elif isinstance(node, Declaration):
            self.visit_declaration(node)
        elif isinstance(node, Cast):
            self.visit_cast(node)
        elif isinstance(node, Ref):
            self.visit_ref(node)
        elif isinstance(node, Constant):
            self.visit_constant(node)
        elif isinstance(node, Register):
            self.visit_register(node)
        elif isinstance(node, Operation):
            self.visit_operation(node)
        else:
            raise RuntimeError(f"Unhandled tree node type: {node}")
