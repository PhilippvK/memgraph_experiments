import logging

from tqdm import tqdm
import networkx as nx

from ..graph_utils import (
    calc_inputs,
    calc_outputs,
    get_instructions,
    calc_weights,
)

logger = logging.getLogger("analyze_io")


def analyze_io(settings, GF, subs, subs_df):
    logger.info("Collecting I/O details...")
    io_subs = []
    for i, sub in enumerate(tqdm(subs, disable=not settings.progress)):
        # if i in isos:
        #     continue
        # i = 3
        # sub = subs[i]
        # print("topo", topo)
        # print("===========================")
        # print("i, sub", i, sub)
        nodes = sub.nodes
        num_nodes = len(nodes)
        num_inputs, inputs, num_constants, constants = calc_inputs(GF, sub)
        num_outputs, outputs = calc_outputs(GF, sub)
        instrs = get_instructions(sub)
        unique_instrs = list(set(instrs))
        total_weight, freq = calc_weights(sub)
        subs_df.at[i, "Nodes"] = list(nodes)
        subs_df.at[i, "#Nodes"] = num_nodes
        subs_df.at[i, "InputNodes"] = list(inputs)
        subs_df.loc[i, "#InputNodes"] = num_inputs
        subs_df.at[i, "ConstantNodes"] = list(constants)
        subs_df.loc[i, "#ConstantNodes"] = num_constants
        subs_df.at[i, "OutputNodes"] = list(outputs)
        subs_df.loc[i, "#OutputNodes"] = num_outputs
        subs_df.at[i, "Instrs"] = instrs
        subs_df.loc[i, "#Instrs"] = len(instrs)
        subs_df.at[i, "UniqueInstrs"] = unique_instrs
        subs_df.loc[i, "#UniqueInstrs"] = len(unique_instrs)
        subs_df.loc[i, "Weight"] = total_weight
        subs_df.loc[i, "Freq"] = freq
        # TODO: also sum up weight and freq for isos? (be careful with overlaps!)

        # print("num_inputs", num_inputs)
        # print("num_outputs", num_outputs)
        # print("inputs", [GF.nodes[inp] for inp in inputs])
        # print("outputs", [GF.nodes[outp] for outp in outputs])
        # TODO: copy fine?
        io_sub = GF.subgraph(list(sub.nodes) + inputs + constants).copy()
        for inp in inputs:
            edges = list(io_sub.in_edges(inp))
            io_sub.remove_edges_from(edges)
        # TODO: drop this (use apply style stage!)
        # SHOW_NODE_IDS = True
        # SHOW_NODE_IDS = False
        # io_sub_topo = list(reversed(list(nx.topological_sort(io_sub))))
        # print("io_sub_topo", io_sub_topo)
        # input(">")
        # inputs_sorted = sorted(inputs, key=lambda x: io_sub_topo.index(x))
        inputs_sorted = sorted(inputs)
        # print("inputs_sorted", inputs_sorted)
        # Normalize io_subs
        input_node_mapping = {n: f"src{ii}" for ii, n in enumerate(inputs_sorted)}
        output_node_mapping = {n: (len(GF.nodes) + ii, f"dst{ii}") for ii, n in enumerate(outputs)}
        # print("output_node_mapping", output_node_mapping)
        outputs_ = []
        for out, temp in output_node_mapping.items():
            out_, out_label = temp
            old_properties = io_sub.nodes[out]["properties"]
            new_properties = {
                "basic_block": old_properties["basic_block"],
                "bb_id": old_properties["bb_id"],
                "func_name": old_properties["func_name"],
                "session": old_properties["session"],
                "module_name": old_properties["module_name"],
                # "name": old_properties["name"],
                "name": out_label,
                "out_reg_class": old_properties["out_reg_class"],
                "out_reg_name": old_properties["out_reg_name"],
                "out_reg_size": old_properties["out_reg_size"],
                "out_reg_type": old_properties["out_reg_type"],
                "op_type": "output",
            }
            node_data = {
                "label": out_label,
                "properties": new_properties,
            }
            edge_data = {
                "properties": {"op_idx": 0, "out_idx": 0},
            }
            io_sub.add_node(out_, **node_data)
            GF.add_node(out_, **node_data)
            io_sub.add_edge(out, out_, **edge_data)
            outputs_.append(out_)
        subs_df.at[i, "OutputNodes"] = outputs_

        # if len(outputs) > 1:
        #     input("yyy")
        # Update subs_df
        subs_df.at[i, "InputNodes"] = list(inputs_sorted)
        subs_df.at[i, "InputNames"] = list(input_node_mapping.values())

        def canonicalize_node(node, graph):
            """Recursively generate a canonical key for a node"""
            node_data = graph.nodes[node]
            if node in constants:
                val = node_data["properties"]["inst"][:-1]
                return f"C:{val}"

            elif node in inputs:
                input_label = input_node_mapping[node]
                return f"I:{input_label}"
            elif node in outputs_:
                output_label = output_node_mapping[outputs[outputs_.index(node)]][1]
                return f"O:{output_label}"

            else:  # Computational node
                operand_hashes = [canonicalize_node(pred, graph) for pred in graph.predecessors(node)]

                COMMUTATIVE_OPS = ["ADD", "MUL"]
                op = node_data["label"]
                # print("operand_hashes", operand_hashes)
                if op in COMMUTATIVE_OPS:
                    # print("COMM")
                    operand_hashes.sort()  # Sort operands to make commutative ops unique
                    # print("sorted_operand_hashes", operand_hashes)

                return f"{op}({','.join(operand_hashes)})"

        new_graph = nx.MultiDiGraph()
        for n, data in io_sub.nodes(data=True):
            new_label = input_node_mapping.get(n, n)  # Replace input nodes, keep others unchanged
            op_hash = canonicalize_node(n, io_sub)
            # print("op_hash", op_hash)
            data_new = {
                # "label": new_label,
                # "label": n,
                "new_label": new_label,
                **data,
                "op_hash": op_hash,
            }
            # new_graph.add_node(new_label, **data_new)
            new_graph.add_node(n, **data_new)

        # Copy edges, ensuring commutative nodes have sorted inputs
        def is_commutative(node_data):
            """Returns True if the node represents a commutative operation."""
            return node_data.get("label") in {"ADD", "MUL", "AND", "OR", "XOR"}  # TODO: Extend

        for dst in io_sub.nodes:
            # print("dst", dst)
            preds = list(io_sub.predecessors(dst))
            # print("preds", preds)

            if preds:
                if is_commutative(io_sub.nodes[dst]):
                    # print("comm")
                    # preds.sort()  # Sort predecessors for commutative operations
                    # preds = sorted(preds, key=lambda x: new_graph.nodes[input_node_mapping.get(x, x)]["op_hash"])
                    preds = sorted(preds, key=lambda x: new_graph.nodes[x]["op_hash"])
                    # print("preds_sorted", preds)

                for k, src in enumerate(preds):
                    # print("src", src)
                    edge_data = io_sub[src][dst]
                    # TODO: generalize
                    assert len(edge_data.keys()) == 1
                    edge_data = list(edge_data.values())[0]
                    edge_properties = edge_data["properties"]
                    op_idx = edge_properties["op_idx"]
                    # print("k", k)
                    # print("op_idx", op_idx)
                    if op_idx != k:
                        edge_data["properties"]["op_idx"] = k
                    # input(">>>")
                    # print("edge_data", edge_data)
                    # new_src = input_node_mapping.get(src, src)
                    # new_dst = input_node_mapping.get(dst, dst)
                    # print("new_src", new_src)
                    # print("new_dst", new_dst)
                    # input("?!")
                    # TODO: add key label type & properties
                    # new_graph.add_edge(new_src, new_dst, **edge_data)
                    new_graph.add_edge(src, dst, **edge_data)
        io_sub = new_graph

        for inp in inputs_sorted:
            # TODO: physreg?
            input_label = input_node_mapping[inp]
            # inp = input_label
            io_sub.nodes[inp]["node_type"] = "IN"
            io_sub.nodes[inp]["label"] = input_label
        for const in set(constants):
            assert io_sub.nodes[const]["label"] == "Const"
            io_sub.nodes[inp]["node_type"] = "CONST"
            label = io_sub.nodes[const]["properties"]["inst"][:-1]
            io_sub.nodes[const]["label"] = label
            value = int(label)  # TODO: handle floating point!
            io_sub.nodes[inp]["value"] = value
        for outp in outputs:
            outp_, output_label = output_node_mapping[outp]
            io_sub.nodes[outp_]["node_type"] = "OUT"
            io_sub.nodes[outp_]["label"] = output_label

        # TODO: add out nodes to io_sub?
        # print("io_sub", io_sub)
        io_subs.append(io_sub)
    return io_subs
