from .utils import mem_lookup


from .tree_utils import Ref, Cast, Constant, Register, Operation, Statements, Declaration, Assignment


class FlatCodeEmitter:

    def __init__(self):
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
        for children in node.children:
            self.visit(children)
            self.write("\n")

    def visit_assignment(self, node):
        assert isinstance(node, (Assignment, Declaration))
        assert len(node.children) == 2
        lhs, rhs = node.children
        self.visit(lhs)
        self.write("=")
        self.visit(rhs)
        self.write(";")

    def visit_declaration(self, node):
        assert isinstance(node, Declaration)
        assert len(node.children) == 2
        lhs, rhs = node.children
        decl_type = node.decl_type
        self.write(f"{decl_type} ")
        self.visit_assignment(node)

    def visit_cast(self, node):
        assert isinstance(node, Cast)
        # print("node", node, dir(node))
        assert len(node.children) == 1
        lhs = node.children[0]
        to_type = node.to
        self.write(f"({to_type})")
        self.write("(")
        self.visit(lhs)
        self.write(")")

    def visit_call(self, node):
        assert isinstance(node, Operation)
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
            self.visit_call(node)
        else:
            raise RuntimeError(f"Unhandled tree node type: {node}")
