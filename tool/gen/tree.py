import networkx as nx
from anytree import AnyNode

from .tree_utils import TreeGenContext
from .cdsl_utils import mem_lookup


def generate_tree(sub, sub_data, GF, xlen=None):
    # ret = {}
    stmts = []
    inputs = sub_data["InputNodes"]
    # num_inputs = int(sub_data["#InputNodes"])
    outputs = sub_data["OutputNodes"]
    # num_outputs = int(sub_data["#OutputNodes"])
    # print("gen_tree", GF, sub, inputs, outputs)
    topo = list(nx.topological_sort(GF))
    inputs = sorted(inputs, key=lambda x: topo.index(x))
    outputs = sorted(outputs, key=lambda x: topo.index(x))
    # treegen = TreeGenContext(sub)
    treegen = TreeGenContext(GF, sub, inputs=inputs)
    # i = 0  # reg
    j = 0  # imm
    for i, inp in enumerate(inputs):
        # print("i", i)
        # print("inp", inp)
        node_data = GF.nodes[inp]
        node_alias = node_data.get("alias", None)
        if node_alias:
            print("ALIAS", node_alias)
            input("!!!")
        # print("node_data", node_data)
        node_properties = node_data["properties"]
        # print("node_properties", node_properties)
        op_type = node_properties["op_type"]
        # print("op_type", op_type)
        reg_class = node_properties.get("out_reg_class", None)
        reg_size = node_properties.get("out_reg_size", None)
        if op_type == "constant":
            continue
        if op_type != "input":
            assert reg_class in ["gpr"], f"Unexpected reg_class: {reg_class}"
            if isinstance(reg_size, str):
                reg_size = int(reg_size)
            assert reg_size in [32, 64], f"Unexpected reg_size: {reg_size}"
            if xlen is not None:
                assert reg_size == xlen, f"reg_size ({reg_size}) does not match xlen ({xlen})"
        res = treegen.visit(inp)
        # print("res", res)
        name = f"inp{j}"
        # print("name", name)
        # input("<>")
        treegen.defs[inp] = name
        # ret[name] = res
        if res.name[:2] == "$x":
            idx = int(res.name[2:])
            # TODO: make more generic to also work for assignments
            ref_ = AnyNode(id=-1, name=res.name, op_type="constant", value=idx)
            res = AnyNode(id=-1, name="X[?]", children=[ref_], op_type="register", reg_class=reg_class)
        else:
            name_ = f"rs{j+1}"
            ref_ = AnyNode(id=-1, name=name_, op_type="ref")
            mem_name = mem_lookup.get(reg_class, None)
            assert mem_name is not None, f"Unable to find mem_name for reg_class: {reg_class}"
            res = AnyNode(id=-1, name=f"{mem_name}[?]", children=[ref_], op_type="register", reg_class=reg_class)
        ref = AnyNode(id=-1, name=name, op_type="ref")
        decl_type = f"unsigned<{reg_size}>"
        root = AnyNode(id=-1, name="ASSIGN1", children=[ref, res], op_type="declaration", decl_type=decl_type)
        stmts.append(root)
        j += 1
        # print(f"{name}:")
        # print(RenderTree(res))
    j = 0
    for i, outp in enumerate(outputs):
        # print("i", i)
        # print("outp", outp)
        node_data = GF.nodes[outp]
        node_alias = node_data.get("alias", None)
        if node_alias:
            print("ALIAS2", node_alias)
            input("!!!2")
        # print("node_data", node_data)
        node_properties = node_data["properties"]
        # print("node_properties", node_properties)
        op_type = node_properties["op_type"]
        # print("op_type", op_type)
        reg_class = node_properties.get("out_reg_class", None)
        assert reg_class in ["gpr"], f"Unexpected reg_class: {reg_class}"
        reg_size = node_properties.get("out_reg_size", None)
        if isinstance(reg_size, str):
            reg_size = int(reg_size)
        assert reg_size in [32, 64], f"Unexpected reg_size: {reg_size}"
        if xlen is not None:
            assert reg_size == xlen, f"reg_size ({reg_size}) does not match xlen ({xlen})"
        res = treegen.visit(outp)
        # print("res", res)
        # TODO: check for may_store, may_branch
        name = f"outp{j}"
        # print("name", name)
        # input("<>")
        treegen.defs[outp] = name
        # ret[name] = root
        # ret[name] = res
        # print("res", res)
        # print(RenderTree(res))
        if res.name in ["SD", "SW", "SH", "SB", "BEQ", "BNE"]:
            root = res
            stmts.append(root)
        else:
            ref = AnyNode(id=-1, name=name, op_type="ref")
            ref_ = AnyNode(id=-1, name=name, op_type="ref")
            # import pdb; pdb.set_trace()
            # print("ref", ref, ref.children)
            # print("res", res, res.children)
            decl_type = f"unsigned<{reg_size}>"
            root = AnyNode(id=-1, name="ASSIGN2", children=[ref, res], op_type="declaration", decl_type=decl_type)
            # print("root", root, root.children)
            stmts.append(root)
            idx = j + 1
            name_ = "rd" if idx == 1 else f"rd{idx}"
            ref2 = AnyNode(id=-1, name=name_, op_type="ref")
            reg = AnyNode(id=-1, name=f"{mem_name}[?]", children=[ref2], op_type="register", reg_class=reg_class)
            # cast_ = AnyNode(id=-1, name="CAST3", children=[ref_], op_type="cast", to=f"unsigned<{reg_size}>")
            # root2 = AnyNode(id=-1, name="ASSIGN3", children=[reg, cast_], op_type="assignment")
            root2 = AnyNode(id=-1, name="ASSIGN3", children=[reg, ref_], op_type="assignment")
            stmts.append(root2)
        j += 1

        # print(f"{name}:")
        # print(RenderTree(res))
    # print("Generating CDSL...")
    codes = []
    header = "// TODO"
    codes.append(header)
    stmts_root = AnyNode(id=-1, children=stmts, op_type="statements")
    return stmts_root
