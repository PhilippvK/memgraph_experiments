def wrap_cdsl(name, code):
    ret = f"{name} {{\n"
    ret += "\n".join(["    " + line for line in code.splitlines()]) + "\n"
    ret += "}\n"
    return ret


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

    def visit_register(self, node):
        assert len(node.children) == 1
        idx = node.children[0]
        self.write("X")
        self.write("[")
        self.visit(idx)
        self.write("]")

    def visit_assignment(self, node):
        # print("visit_assignment", node, node.children)
        assert len(node.children) == 2
        lhs, rhs = node.children
        self.visit(lhs)
        self.write("=")
        self.visit(rhs)
        self.write(";")

    def visit_branch(self, node):
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

    def visit_load(self, node):
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
            self.write(f"(unsigned<{sz}>)")
        else:
            self.write(f"(signed<{sz}>)")
        self.write("MEM[")
        self.visit(base)
        self.write("+")
        self.visit(offset)
        self.write("]")
        self.write(")")

    def visit_store(self, node):
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

    def visit_lui(self, node):
        self.write("lui(TODO)")

    def visit_binop(self, node):
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
            "XOR": ("^", False, self.xlen, False),
            "ANDI": ("&", False, self.xlen, True),
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
        self.visit(lhs)
        self.write(op)
        self.visit(rhs)
        self.write(")")

    def visit_cond_set(self, node):
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
        # print("visit", node)
        op_type = node.op_type
        # print("op_type", op_type)
        if op_type == "assignment":
            self.visit_assignment(node)
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
                self.visit_binop(node)
            elif name in ["SLT", "SLTU"]:
                self.visit_cond_set(node)
            elif name in ["BNE", "BEQ"]:
                self.visit_branch(node)
            elif name in ["LH", "LW"]:
                self.visit_load(node)
            elif name in ["SW", "SH"]:
                self.visit_store(node)
            elif name in ["LUI"]:
                self.visit_lui(node)
            else:
                raise NotImplementedError(f"Unhandled: {name}")
