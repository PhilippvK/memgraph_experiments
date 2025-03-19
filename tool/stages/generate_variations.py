import logging

from tqdm import tqdm

from ..hash import add_hash_attr
from ..enum import Variation
from ..encoding import calc_encoding_footprint

logger = logging.getLogger("generate_variations")


def generate_variations(settings, subs, GF, io_subs, subs_df):
    logger.info("Generating variations...")
    # 1. look for singleUse edges to reuse as output reg
    if settings.enable_variation_reuse_io:
        filtered_subs_df = subs_df[(subs_df["Status"] & settings.write.gen_flt) > 0].copy()
        io_subs_iter = [(i, io_sub) for i, io_sub in enumerate(io_subs) if i in filtered_subs_df.index]
        # for i, sub in enumerate(tqdm(subs, disable=not settings.progress)):
        for i, io_sub in tqdm(io_subs_iter, disable=not settings.progress):
            # print("i", i)
            # print("io_sub", io_sub)
            sub = subs[i]
            # print("sub", sub)
            sub_data = subs_df.iloc[i]
            # print("sub_data", sub_data)
            inputs = sub_data["InputNodes"]
            # input_names = sub_data["InputNames"]
            num_inputs = int(sub_data["#InputNodes"])
            # print("inputs", inputs)
            # print("input_names", input_names)
            # print("num_inputs", num_inputs)
            operand_names = sub_data["OperandNames"]
            # print("operand_names", operand_names)
            operand_nodes = sub_data["OperandNodes"]
            # print("operand_nodes", operand_nodes)
            operand_dirs = sub_data["OperandDirs"]
            # print("operand_dirs", operand_dirs)
            operand_types = sub_data["OperandTypes"]
            # print("operand_types", operand_types)
            operand_enc_bits = sub_data["OperandEncBits"]
            # print("operand_enc_bits", operand_enc_bits)
            operand_reg_classes = sub_data["OperandRegClasses"]
            # print("operand_reg_classes", operand_reg_classes)
            outputs = sub_data["OutputNodes"]
            # print("outputs", outputs)
            # output_names = sub_data["OutputNames"]
            # print("output_names", output_names)
            output_op_names = operand_names[: len(outputs)]
            # print("output_op_names", output_op_names)
            input_op_names = operand_names[len(outputs) :]
            # print("input_op_names", input_op_names)
            num_outputs = int(sub_data["#OutputNodes"])
            # print("num_outputs", num_outputs)
            if num_inputs < 1 or num_outputs < 1:
                continue
            # new = []
            for input_idx, j in enumerate(inputs):
                # print("input_idx", input_idx)
                # print("j", j)
                # TODO: reuse input node id for output or vice-versa?
                # Would create loop? Add constraint inp0 == outp0 to df?
                # TODO: check if input and output reg types match
                input_node_data = GF.nodes[j]
                # print("input_node_data", input_node_data)
                # input_properties = input_node_data["properties"]
                # print("input_properties", input_properties)
                edge_count = 0
                single_use = None
                for src, dst, edge_data in GF.out_edges(j, data=True):
                    # print("src", src)
                    # print("dst", dst)
                    # print("edge_data", edge_data)
                    edge_properties = edge_data["properties"]
                    # print("edge_properties", edge_properties)
                    single_use_ = edge_properties.get("op_reg_single_use", None)
                    # print("single_use_", single_use_)
                    if single_use_ is not None:
                        single_use = single_use_
                    edge_count += 1
                # print("edge_count", edge_count)
                # print("single_use", single_use)
                if single_use and edge_count == 1:
                    # assert edge_count == 1
                    for output_idx, k in enumerate(outputs):
                        # print("output_idx", output_idx)
                        # print("k", k)
                        # output_node_data = GF.nodes[j]
                        # print("output_node_data", output_node_data)
                        # output_properties = output_node_data["properties"]
                        # print("output_properties", output_properties)
                        new_sub = sub.copy()
                        # print("new_sub", new_sub)
                        new_sub_data = sub_data.copy()
                        # print("new_sub_data", new_sub_data)
                        # new_io_sub_ = io_sub.copy()
                        # print("new_io_sub_", new_io_sub_)
                        # print("new_io_sub_.nodes", new_io_sub_.nodes)
                        # print("new_io_sub_.edges", new_io_sub_.edges)
                        new_io_sub_nodes = [x for x in io_sub.nodes if x != j]
                        new_input_node_data = input_node_data.copy()
                        new_input_node_data["alias"] = k
                        # print("new_input_node_data", new_input_node_data)
                        new_input_node_id = max(GF.nodes) + 1
                        # print("new_input_node_id", new_input_node_id)
                        GF.add_node(new_input_node_id, **new_input_node_data)
                        new_io_sub_nodes.append(new_input_node_id)
                        for src, dst, dat in io_sub.out_edges(j, data=True):
                            assert dst in io_sub.nodes
                            # print("src", src)
                            # print("dst", dst)
                            # print("dat", dat)
                            GF.add_edge(new_input_node_id, dst, **dat)
                            # print(f"{new_input_node_id} -> {dst}")
                        # print("new_io_sub_nodes", new_io_sub_nodes)
                        new_io_sub = GF.subgraph(new_io_sub_nodes)
                        # print("new_io_sub", new_io_sub)
                        # print("new_io_sub.nodes", new_io_sub.nodes)
                        # print("new_io_sub.edges", new_io_sub.edges)
                        # input("123")
                        input_op_name = input_op_names[input_idx]
                        output_op_name = output_op_names[output_idx]
                        new_constraint = f"{input_op_name} == {output_op_name}"
                        # print("new_constraint", new_constraint)
                        new_sub_data["Constraints"] = [new_constraint]
                        new_operand_names = [x for x in operand_names if x != input_op_name]
                        new_operand_dirs = [
                            operand_dirs[iii] if x != output_op_name else "INOUT"
                            for iii, x in enumerate(operand_names)
                            if x != input_op_name
                        ]
                        new_operand_nodes = [
                            # operand_nodes[iii] if x != input_op_name else new_input_node_id
                            operand_nodes[iii]
                            for iii, x in enumerate(operand_names)
                            if x != input_op_name
                        ]
                        new_operand_types = [
                            operand_types[iii] for iii, x in enumerate(operand_names) if x != input_op_name
                        ]
                        new_operand_reg_classes = [
                            operand_reg_classes[iii] for iii, x in enumerate(operand_names) if x != input_op_name
                        ]
                        new_operand_enc_bits = [
                            operand_enc_bits[iii] for iii, x in enumerate(operand_names) if x != input_op_name
                        ]
                        new_operand_enc_bits_sum = sum(new_operand_enc_bits)
                        parent = i
                        new_sub_data["Parent"] = parent
                        new_sub_data["Variations"] |= Variation.REUSE_IO
                        new_sub_data["OperandNames"] = new_operand_names
                        new_sub_data["OperandNodes"] = new_operand_nodes
                        new_input_nodes = [
                            inp if input_idx_ != input_idx else new_input_node_id
                            for input_idx_, inp in enumerate(inputs)
                        ]
                        new_sub_data["InputNodes"] = new_input_nodes
                        new_sub_data["OperandDirs"] = new_operand_dirs
                        new_sub_data["OperandTypes"] = new_operand_types
                        new_sub_data["OperandRegClasses"] = new_operand_reg_classes
                        new_sub_data["OperandEncBits"] = new_operand_enc_bits
                        new_sub_data["OperandEncBitsSum"] = new_operand_enc_bits_sum
                        for enc_size in settings.allowed_enc_sizes:
                            enc_bits_left, enc_weight, enc_footprint = calc_encoding_footprint(
                                new_operand_enc_bits_sum, enc_size
                            )
                            new_sub_data[f"EncodingBitsLeft ({enc_size} bits)"] = enc_bits_left
                            new_sub_data[f"EncodingWeight ({enc_size} bits)"] = enc_weight
                            new_sub_data[f"EncodingFootprint ({enc_size} bits)"] = enc_footprint
                        # TODO: re-calculate encoding footprint
                        new_sub_id = len(io_subs)
                        # print("new_sub_id", new_sub_id)
                        new_sub_data["result"] = new_sub_id
                        # print("new_sub_data_", new_sub_data)
                        subs_df.loc[new_sub_id] = new_sub_data
                        add_hash_attr(new_sub)
                        add_hash_attr(new_io_sub)
                        add_hash_attr(new_io_sub, attr_name="hash_attr_ignore_const", ignore_const=True)
                        subs.append(new_sub)
                        io_subs.append(new_io_sub)
                        # new.append(None)
                        # input("||")
            # print("new", new)
