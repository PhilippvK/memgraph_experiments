import networkx as nx

from .tree_utils import TreeGenContext, Constant, Ref, Declaration, Assignment, Register, Statements
from ..llvm_utils import llvm_type_to_cdsl_type


# def generate_tree(sub, sub_data, GF, xlen=None):
def generate_tree(sub, sub_data, io_sub, xlen=None):
    # print("generate_tree", sub, sub_data, GF, xlen)
    # print("generate_tree", sub, sub_data, io_sub, xlen)

    constants = sub_data["ConstantNodes"]
    # input_nodes = sub_data["InputNodes"]
    # output_nodes = sub_data["OutputNodes"]
    # print("input_nodes", input_nodes)
    # print("output_nodes", output_nodes)
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

    # topo = list(nx.topological_sort(GF))
    topo = list(nx.topological_sort(io_sub))
    inputs = sorted(inputs, key=lambda x: topo.index(x))
    # print("inputs", inputs)
    outputs = sorted(outputs, key=lambda x: topo.index(x))
    # print("outputs", outputs)

    def find_aliases(graph, sub_data):
        ret = {}
        input_nodes = sub_data["InputNodes"]
        for input_node in input_nodes:
            # print("input_node", input_node)
            input_node_data = graph.nodes[input_node]
            # print("input_node_data", input_node_data)
            input_node_alias = input_node_data.get("alias", None)
            if input_node_alias is not None:
                ret[input_node] = input_node_alias
        return ret

    # node_aliases = find_aliases(GF, sub_data)
    node_aliases = find_aliases(io_sub, sub_data)
    # print("node_aliases", node_aliases)

    def get_all_inputs(inputs, node_aliases, outputs):
        all_inputs = []
        all_inputs += inputs
        for k, v in node_aliases.items():
            all_inputs.append(k)
            # all_inputs.append(v)
            all_inputs.remove(v)
            assert v in outputs
        return all_inputs

    all_inputs = get_all_inputs(inputs, node_aliases, outputs)
    # all_inputs = inputs
    # print("all_inputs", all_inputs)

    # TODO: io_sub for edges?
    treegen = TreeGenContext(
        # GF, sub, sub_data, node_aliases=node_aliases, inputs=all_inputs, outputs=outputs, constants=constants
        io_sub,
        sub,
        io_sub,
        sub_data,
        node_aliases=node_aliases,
        inputs=all_inputs,
        outputs=outputs,
        constants=constants,
    )
    tree = treegen.generate_tree()
    return tree
