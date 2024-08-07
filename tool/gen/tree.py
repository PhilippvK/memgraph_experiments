import networkx as nx
from anytree import AnyNode

from .tree_utils import TreeGenContext
from .cdsl_utils import mem_lookup


def generate_tree(sub, sub_data, GF, xlen=None):
    # print("sub", sub)
    print("sub_data", sub_data)
    # print("GF", GF)
    # ret = {}
    stmts = []
    sub_id = sub_data["result"]
    print("sub_id", sub_id, type(sub_id))
    if int(sub_id) == 2574:
        input("!3574!")
    operand_names = sub_data["OperandNames"]
    print("operand_names", operand_names)
    operand_nodes = sub_data["OperandNodes"]
    print("operand_nodes", operand_nodes)
    operand_dirs = sub_data["OperandDirs"]
    print("operand_dirs", operand_dirs)
    operand_types = sub_data["OperandTypes"]  # TODO: rename to Classes?
    # operand_enc_bits = sub_data["OperandEncBits"]
    operand_reg_classes = sub_data["OperandRegClasses"]
    assert (
        len(operand_names) == len(operand_nodes) == len(operand_dirs) == len(operand_types) == len(operand_reg_classes)
    )
    inputs = [operand_nodes[i] for i, operand_dir in enumerate(operand_dirs) if operand_dir in ["IN", "INOUT"]]
    outputs = [operand_nodes[i] for i, operand_dir in enumerate(operand_dirs) if operand_dir in ["OUT", "INOUT"]]
    # inputs = sub_data["InputNodes"]
    # num_inputs = int(sub_data["#InputNodes"])
    # outputs = sub_data["OutputNodes"]
    # num_outputs = int(sub_data["#OutputNodes"])
    # print("gen_tree", GF, sub, inputs, outputs)
    topo = list(nx.topological_sort(GF))
    inputs = sorted(inputs, key=lambda x: topo.index(x))
    outputs = sorted(outputs, key=lambda x: topo.index(x))

    def find_aliases(GF, sub_data):
        ret = {}
        input_nodes = sub_data["InputNodes"]
        for input_node in input_nodes:
            # print("input_node", input_node)
            input_node_data = GF.nodes[input_node]
            # print("input_node_data", input_node_data)
            input_node_alias = input_node_data.get("alias", None)
            if input_node_alias is not None:
                ret[input_node] = input_node_alias
        return ret

    node_aliases = find_aliases(GF, sub_data)
    print("node_aliases", node_aliases)
    all_inputs = []
    all_inputs += inputs
    for k, v in node_aliases.items():
        all_inputs.append(k)
        all_inputs.append(v)
        assert v in outputs
    # treegen = TreeGenContext(sub)
    # list(node_aliases.keys())
    treegen = TreeGenContext(GF, sub, inputs=all_inputs, outputs=outputs)
    # i = 0  # reg
    j = 0  # imm
    for i, inp in enumerate(inputs):
        op_idx = operand_nodes.index(inp)
        print("op_idx", op_idx)
        print("i", i)
        print("inp", inp)
        operand_name = operand_names[op_idx]
        operand_dir = operand_dirs[op_idx]
        operand_type = operand_types[op_idx]
        if operand_dir == "INOUT":
            print("INOUT")

            assert len(node_aliases) > 0
            assert inp in node_aliases.values()
            alias = list(node_aliases.keys())[list(node_aliases.values()).index(inp)]
            inp = alias
            # input("!!!")
        node_data = GF.nodes[inp]
        print("node_data", node_data)
        # print("node_data", node_data)
        node_properties = node_data["properties"]
        # print("node_properties", node_properties)
        op_type = node_properties["op_type"]
        # print("op_type", op_type)

        if op_type == "constant":
            continue

        assert operand_type == "REG"
        # reg_class = node_properties.get("out_reg_class", None)
        reg_class = operand_reg_classes[op_idx].lower()
        reg_size = node_properties.get("out_reg_size", None)
        # enc_bits = int(operand_enc_bits[op_idx])

        if op_type != "input":
            assert reg_class in ["gpr"], f"Unexpected reg_class: {reg_class}"
            if isinstance(reg_size, str):
                reg_size = int(reg_size)
            assert reg_size in [32, 64], f"Unexpected reg_size: {reg_size}"
            if xlen is not None:
                assert reg_size == xlen, f"reg_size ({reg_size}) does not match xlen ({xlen})"
        res = treegen.visit(inp)
        # print("res", res)
        # name = f"inp{j}"
        name = f"{operand_name}_val"
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
            # name_ = f"rs{j+1}"
            name_ = operand_name
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
        print("i", i)
        print("outp", outp)
        if outp in operand_nodes:
            print("Found!")
            op_idx = operand_nodes.index(outp)
        else:
            print("Not Found!")
            assert outp in node_aliases
            outp = node_aliases[outp]
            op_idx = operand_nodes.index(outp)
        print("op_idx", op_idx)
        operand_name = operand_names[op_idx]
        operand_dir = operand_dirs[op_idx]
        operand_type = operand_types[op_idx]
        node_data = GF.nodes[outp]
        # node_alias = node_data.get("alias", None)
        # if node_alias:
        #     print("ALIAS2", node_alias)
        #     input("!!!2")
        # print("node_data", node_data)
        node_properties = node_data["properties"]
        # print("node_properties", node_properties)
        op_type = node_properties["op_type"]
        print("op_type", op_type)
        operand_name = operand_names[op_idx]
        operand_dir = operand_dirs[op_idx]
        if operand_dir == "INOUT":
            print("INOUT2!")
            input("INOUT2")
        operand_type = operand_types[op_idx]
        assert operand_type == "REG"
        # reg_class = node_properties.get("out_reg_class", None)
        reg_class = operand_reg_classes[op_idx].lower()
        assert reg_class in ["gpr"], f"Unexpected reg_class: {reg_class}"
        reg_size = node_properties.get("out_reg_size", None)
        # enc_bits = int(operand_enc_bits[op_idx])
        # reg_size = int(2**enc_bits)
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
            # name_ = "rd" if idx == 1 else f"rd{idx}"
            name_ = operand_name
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
    stmts_root = AnyNode(id=-1, name="statements", children=stmts, op_type="statements")
    if int(sub_id) == 2574:
        input("!3574!")
    return stmts_root
