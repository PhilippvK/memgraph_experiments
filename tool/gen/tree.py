import networkx as nx

from .tree_utils import TreeGenContext, Constant, Ref, Declaration, Assignment, Register, Statements
from .cdsl_utils import mem_lookup
from ..llvm_utils import llvm_type_to_cdsl_type


def generate_tree(sub, sub_data, GF, xlen=None):
    # print("sub", sub)
    # print("sub_data", sub_data)
    # print("GF", GF)
    # ret = {}
    stmts = []
    constants = sub_data["ConstantNodes"]
    operand_names = sub_data["OperandNames"]
    # print("operand_names", operand_names)
    operand_nodes = sub_data["OperandNodes"]
    # print("operand_nodes", operand_nodes)
    operand_dirs = sub_data["OperandDirs"]
    # print("operand_dirs", operand_dirs)
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
    # print("node_aliases", node_aliases)
    all_inputs = []
    all_inputs += inputs
    for k, v in node_aliases.items():
        all_inputs.append(k)
        all_inputs.append(v)
        assert v in outputs
    # treegen = TreeGenContext(sub)
    # list(node_aliases.keys())
    treegen = TreeGenContext(GF, sub, inputs=all_inputs, outputs=outputs, constants=constants)
    # i = 0  # reg
    j = 0  # imm
    for i, inp in enumerate(inputs):
        op_idx = operand_nodes.index(inp)
        # print("op_idx", op_idx)
        # print("i", i)
        # print("inp", inp)
        operand_name = operand_names[op_idx]
        operand_dir = operand_dirs[op_idx]
        operand_type = operand_types[op_idx]
        if operand_dir == "INOUT":
            # print("INOUT")

            assert len(node_aliases) > 0
            assert inp in node_aliases.values()
            alias = list(node_aliases.keys())[list(node_aliases.values()).index(inp)]
            inp = alias
        node_data = GF.nodes[inp]
        # print("node_data", node_data)
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
                if reg_size == "unknown":
                    reg_size = None
                else:
                    reg_size = int(reg_size)
            if reg_size is not None:
                assert reg_size in [32, 64], f"Unexpected reg_size: {reg_size}"
                if xlen is not None:
                    pass  # Ignore XLEN for now: check if casts are added automatically
                    # assert reg_size == xlen, f"reg_size ({reg_size}) does not match xlen ({xlen})"
        res = treegen.visit(inp)
        # print("res", res)
        # name = f"inp{j}"
        name = f"{operand_name}_val"
        # print("name", name)
        treegen.defs[inp] = name
        # ret[name] = res
        if hasattr(res, "name") and res.name[:2] == "$x":
            idx = int(res.name[2:])
            # TODO: make more generic to also work for assignments
            ref_ = Constant(value=idx, in_types=[], out_types=[None])
            signed = False
            cdsl_type = llvm_type_to_cdsl_type(None, signed, reg_size=reg_size)
            res = Register(
                name="X[?]",
                children=[ref_],
                reg_class=reg_class,
                in_types=ref_.out_types,
                out_types=[cdsl_type],
            )
        else:
            # name_ = f"rs{j+1}"
            name_ = operand_name
            ref_ = Ref(name=name_, in_types=[], out_types=[None])
            mem_name = mem_lookup.get(reg_class, None)
            assert mem_name is not None, f"Unable to find mem_name for reg_class: {reg_class}"
            signed = False
            cdsl_type = llvm_type_to_cdsl_type(None, signed, reg_size=reg_size)
            res = Register(
                name=f"{mem_name}[?]",
                children=[ref_],
                reg_class=reg_class,
                in_types=ref_.out_types,
                out_types=[cdsl_type],
            )
        ref = Ref(name=name, in_types=[], out_types=[None])
        decl_type = f"unsigned<{reg_size}>"
        root = Declaration(
            children=[ref, res],
            decl_type=decl_type,
            in_types=[ref.out_types[0], res.out_types[0]],
            out_types=[decl_type],
        )
        stmts.append(root)
        j += 1
        # print(f"{name}:")
        # print(RenderTree(res))
    j = 0
    for i, outp in enumerate(outputs):
        # print("i", i)
        # print("outp", outp)
        if outp in operand_nodes:
            # print("Found!")
            op_idx = operand_nodes.index(outp)
        else:
            # print("Not Found!")
            assert outp in node_aliases
            outp = node_aliases[outp]
            op_idx = operand_nodes.index(outp)
        # print("op_idx", op_idx)
        operand_name = operand_names[op_idx]
        operand_dir = operand_dirs[op_idx]
        operand_type = operand_types[op_idx]
        node_data = GF.nodes[outp]
        # print("node_data", node_data)
        node_properties = node_data["properties"]
        # print("node_properties", node_properties)
        op_type = node_properties["op_type"]
        # print("op_type", op_type)
        operand_name = operand_names[op_idx]
        operand_dir = operand_dirs[op_idx]
        operand_type = operand_types[op_idx]
        assert operand_type == "REG"
        # reg_class = node_properties.get("out_reg_class", None)
        reg_class = operand_reg_classes[op_idx].lower()
        assert reg_class in ["gpr"], f"Unexpected reg_class: {reg_class}"
        reg_size = node_properties.get("out_reg_size", None)
        # enc_bits = int(operand_enc_bits[op_idx])
        # reg_size = int(2**enc_bits)
        if isinstance(reg_size, str):
            if reg_size == "unknown":
                reg_size = None
            else:
                reg_size = int(reg_size)
        if reg_size is not None:
            assert reg_size in [32, 64], f"Unexpected reg_size: {reg_size}"
            if xlen is not None:
                pass  # Ignore XLEN for now: check if casts are added automatically
                # assert reg_size == xlen, f"reg_size ({reg_size}) does not match xlen ({xlen})"
        res = treegen.visit(outp)
        # print("res", res)
        # TODO: check for may_store, may_branch
        name = f"outp{j}"
        # print("name", name)
        treegen.defs[outp] = name
        # ret[name] = root
        # ret[name] = res
        # print("res", res)
        # print(RenderTree(res))
        if hasattr(res, "name") and res.name in ["SD", "SW", "SH", "SB", "BEQ", "BNE"]:  # TODO: DETECT via predicates or zero outputs!
            root = res
            stmts.append(root)
        else:
            ref = Ref(name=name, in_types=[], out_types=[None])
            ref_ = Ref(name=name, in_types=[], out_types=[None])
            # import pdb; pdb.set_trace()
            # print("ref", ref, ref.children)
            # print("res", res, res.children)
            decl_type = f"unsigned<{reg_size}>"
            root = Declaration(
                children=[ref, res],
                decl_type=decl_type,
                in_types=[ref.out_types[0], res.out_types[0]],
                out_types=[decl_type],
            )
            # print("root", root, root.children)
            stmts.append(root)
            idx = j + 1
            # name_ = "rd" if idx == 1 else f"rd{idx}"
            name_ = operand_name
            ref2 = Ref(name=name_, in_types=[], out_types=[None])
            reg = Register(
                name=f"{mem_name}[?]",
                children=[ref2],
                reg_class=reg_class,
                in_types=[ref2.out_types[0]],
                out_types=[None],
            )
            # cast_ = AnyNode(children=[ref_], op_type="cast", to=f"unsigned<{reg_size}>")
            # root2 = AnyNode(children=[reg, cast_], op_type="assignment")
            root2 = Assignment(
                children=[reg, ref_],
                in_types=[reg.out_types[0], ref_.out_types[0]],
                out_types=[None],
            )
            stmts.append(root2)
        j += 1

        # print(f"{name}:")
        # print(RenderTree(res))
    # print("Generating CDSL...")
    codes = []
    header = "// TODO"
    codes.append(header)
    stmts_root = Statements(
        name="statements",
        children=stmts,
        op_type="statements",
        in_types=[None] * len(stmts),
        out_types=[None],
    )
    return stmts_root
