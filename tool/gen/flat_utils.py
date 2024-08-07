from .cdsl_utils import mem_lookup


class FlatCodeEmitter:

    def __init__(self):
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
        reg_class = node.reg_class
        mem_name = mem_lookup.get(reg_class, None)
        assert mem_name is not None, f"Unable to find mem_name for reg_class: {reg_class}"
        self.write(mem_name)
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

    def visit_call(self, node):
        # print("visit_assignment", node, node.children)
        name = node.name
        args = node.children
        assert len(args) > 0
        self.write(name)
        self.write("(")
        for i, arg in enumerate(args):
            if i > 0:
                self.write(",")
            self.visit(arg)
        self.write(")")

    def visit(self, node):
        # print("visit", node)
        op_type = node.op_type
        # print("op_type", op_type)
        if op_type == "assignment":
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
            # print("op_type", op_type)
            assert op_type in ["operator", "output"]
            self.visit_call(node)
