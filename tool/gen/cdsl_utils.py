from enum import IntEnum, auto


def wrap_cdsl(name, code):
    ret = f"{name} {{\n"
    ret += "\n".join(["    " + line for line in code.splitlines()]) + "\n"
    ret += "}\n"
    return ret


mem_lookup = {
    "gpr": "X",
    "fpr": "F",
    "csr": "CSR",
}


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
        self.write(node.name)

    def visit_constant(self, node):
        self.write("(")
        self.write(node.value)  # TODO: dtype
        self.write(")")

    def visit_constant_generic(self, node):
        print("visit_constant_generic")
        print("node", node)
        print("node.children", node.children)
        raise NotImplementedError("G_CONSTANT (missing children)")

    def visit_trunc_generic(self, node):
        print("visit_trunc_generic")
        print("node", node)
        print("node.children", node.children)
        raise NotImplementedError("G_CONSTANT (missing children)")

    def visit_register(self, node):
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
        # print("visit_statements", node, node.children)
        for children in node.children:
            self.visit(children)
            self.write("\n")

    def visit_assignment(self, node):
        # print("visit_assignment", node, node.children)
        assert len(node.children) == 2
        lhs, rhs = node.children
        self.visit(lhs)
        self.write("=")
        self.visit(rhs)
        self.write(";")

    def visit_declaration(self, node):
        # print("visit_declaration", node, node.children)
        assert len(node.children) == 2
        lhs, rhs = node.children
        decl_type = node.decl_type
        self.write(f"{decl_type} ")
        self.visit_assignment(node)

    def visit_cast(self, node):
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

        lookup = {
            "G_LOAD": (ExtendMode.UNDEFINED,),
            "G_SEXTLOAD": (ExtendMode.SIGN_EXTEND,),
            "G_ZEXTLOAD": (ExtendMode.ZERO_EXTEND,),
        }
        res = lookup.get(node.name)
        assert res is not None
        (mode,) = res
        # TODO: dtypes?
        sz = 0
        print("node.children", node.children)
        assert len(node.children) == 1
        (base,) = node.children
        self.write("(unsigned<XLEN>)")
        self.write("(")
        if mode == ExtendMode.SIGN_EXTEND:
            self.write(f"(signed<{sz}>)")
        elif mode == "ExtendMode.ZERO_EXTEND":
            self.write(f"(unsigned<{sz}>)")
        else:
            pass  # TODO: ok?
        self.write("MEM[")
        self.visit(base)
        self.write("+")
        # self.visit(offset)
        self.write("?")
        self.write("]")
        self.write(")")

    def visit_load_riscv(self, node):
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
        self.write("lui(TODO)")

    def visit_binop_generic(self, node):
        # TODO: imm needed?
        print("visit_binop_generic", node)
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
        print("lhs", lhs)
        print("rhs", rhs)
        # TODO: add size only of != xlen?
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
        # TODO: imm needed?
        lookup = {
            "ADD": ("+", True, self.xlen, False),
            "ADDW": ("+", True, 32, False),
            "ADDI": ("+", True, self.xlen, True),
            "ADDIW": ("+", True, 32, True),
            "SUBW": ("-", True, 32, False),
            "SRA": (">>", True, self.xlen, False),
            "SRAI": (">>", True, self.xlen, True),
            "SRLI": (">>", False, self.xlen, True),
            "SLL": ("<<", False, self.xlen, False),
            "SLLI": ("<<", False, self.xlen, True),
            "AND": ("&", False, self.xlen, False),
            "OR": ("|", False, self.xlen, False),
            "XOR": ("^", False, self.xlen, False),
            "ANDI": ("&", False, self.xlen, True),
            "ORI": ("|", False, self.xlen, True),
            "MULW": ("*", True, 32, False),
            "MUL": ("*", True, self.xlen, False),
        }
        res = lookup.get(node.name)
        assert res is not None
        op, signed, sz, imm = res
        # TODO: check imm (sign/sz?)
        # TODO: dtype
        self.write("(")
        assert len(node.children) == 2
        lhs, rhs = node.children
        # TODO: add size only of != xlen?
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

    def visit_cond_set_riscv(self, node):
        lookup = {"SLT": ("<", True), "SLTU": ("<", False)}
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

    def visit(self, node):
        print("visit", node)
        op_type = node.op_type
        # print("op_type", op_type)
        if op_type == "statements":
            self.visit_statements(node)
        elif op_type == "assignment":
            self.visit_assignment(node)
        elif op_type == "declaration":
            self.visit_declaration(node)
        elif op_type == "cast":
            self.visit_cast(node)
        elif op_type == "ref":
            self.visit_ref(node)
        elif op_type == "constant":
            self.visit_constant(node)
        # TODO: implement this
        elif op_type == "register":
            self.visit_register(node)
        else:
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
                "ADD",
                "ADDI",
                "ADDW",
                "MULW",
                "MUL",
                "SRA",
                "SRAI",
                "SLL",
                "SUBW",
            ]:
                self.visit_binop_riscv(node)
            elif name in ["SLT", "SLTU"]:
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
            elif name in [
                "G_ADD",
                "G_PTR_ADD",
                "G_MUL",
            ]:
                self.visit_binop_generic(node)
            elif name in ["G_LOAD", "G_SEXTLOAD", "G_ZEXTLOAD"]:
                self.visit_load_generic(node)
            else:
                raise NotImplementedError(f"Unhandled: {name}")
