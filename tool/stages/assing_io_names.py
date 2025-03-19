import logging

from tqdm import tqdm

from ..encoding import calc_encoding_footprint

logger = logging.getLogger("assign_io_names")


def assign_io_names(settings, subs, io_subs, subs_df):
    logger.info("Determining IO names...")
    filtered_subs_df = subs_df[(subs_df["Status"] & settings.write.gen_flt) > 0].copy()
    io_subs_iter = [(i, io_sub) for i, io_sub in enumerate(io_subs) if i in filtered_subs_df.index]
    # for i, sub in enumerate(tqdm(subs, disable=not settings.progress)):
    for i, io_sub in tqdm(io_subs_iter, disable=not settings.progress):
        # sub = subs[i]
        sub_data = subs_df.iloc[i]
        inputs = sub_data["InputNodes"]
        # print("inputs", inputs)
        # io_sub_topo = list(reversed(list(nx.topological_sort(io_sub))))
        # print("io_sub_topo", io_sub_topo)
        # inputs_sorted = sorted(inputs, key=lambda x: io_sub_topo.index(x))
        # print("inputs_sorted", inputs_sorted)
        # input_node_mapping = {n: f"src{i}" for i, n in enumerate(inputs_sorted)}
        # input("555")
        # inputs = sub_data["InputNames"]
        outputs = sub_data["OutputNodes"]
        operand_names = []
        operand_nodes = []
        operand_types = []
        operand_reg_classes = []
        operand_dirs = []
        operand_enc_bits = []
        output_names = []
        for output_idx, j in enumerate(outputs):
            output_name = f"outp{output_idx}"
            # output_node = io_sub.nodes[j]
            # output_properties = output_node["properties"]
            # op_type = output_properties["op_type"]
            output_names.append(output_name)
            # TODO: do not hardcode
            op_type_ = "REG"
            op_dir = "OUT"
            enc_bits = 5
            reg_class = "GPR"  # TODO: get from nodes!
            op_name = "rd" if output_idx == 0 else f"rd{output_idx+1}"
            operand_names.append(op_name)
            operand_nodes.append(j)
            operand_types.append(op_type_)
            operand_reg_classes.append(reg_class)
            operand_dirs.append(op_dir)
            operand_enc_bits.append(enc_bits)
        input_names = []
        for input_idx, j in enumerate(inputs):
            input_name = f"inp{input_idx}"
            # input_node = io_sub.nodes[j]
            # input_properties = input_node["properties"]
            # op_type = input_properties["op_type"]
            input_names.append(input_name)
            # TODO: do not hardcode
            op_type_ = "REG"
            op_dir = "IN"
            enc_bits = 5
            reg_class = "GPR"
            op_name = f"rs{input_idx+1}"
            operand_names.append(op_name)
            operand_nodes.append(j)
            operand_types.append(op_type_)
            operand_reg_classes.append(reg_class)
            operand_dirs.append(op_dir)
            operand_enc_bits.append(enc_bits)
        subs_df.at[i, "OutputNames"] = output_names
        subs_df.at[i, "InputNames"] = input_names
        subs_df.at[i, "OperandNames"] = operand_names
        subs_df.at[i, "OperandTypes"] = operand_types
        subs_df.at[i, "OperandRegClasses"] = operand_reg_classes
        subs_df.at[i, "OperandNodes"] = operand_nodes
        subs_df.at[i, "OperandDirs"] = operand_dirs
        subs_df.at[i, "OperandEncBits"] = operand_enc_bits
        enc_bits_sum = sum(operand_enc_bits)
        subs_df.loc[i, "OperandEncBitsSum"] = enc_bits_sum
        # TODO: move to new stage
        for enc_size in settings.allowed_enc_sizes:
            enc_bits_left, enc_weight, enc_footprint = calc_encoding_footprint(enc_bits_sum, enc_size)
            subs_df.loc[i, f"EncodingBitsLeft ({enc_size} bits)"] = enc_bits_left
            subs_df.loc[i, f"EncodingWeight ({enc_size} bits)"] = enc_weight
            subs_df.loc[i, f"EncodingFootprint ({enc_size} bits)"] = enc_footprint
