import logging
from typing import Optional, List

from tqdm import tqdm

from ..hash import add_hash_attr
from ..enums import Variation, ImmStrategy
from ..encoding import calc_encoding_footprint

logger = logging.getLogger("generate_variations")

def generate_variations(settings, subs, GF, io_subs, subs_df):
    # TODO: REG2IMM, CONST2IMM, CONST2REG
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
            if num_inputs <= 1 or num_outputs < 1:
                # 1 source & 1 dest as as cheap as it can get, no need for sharing
                continue
            # new = []
            # TODO: avoid generating 2 variations for MUL(rd_val, rs2_val) & MUL(rs1_val, rd_val)
            # -> if commutable op is only consumer of 2 inputs, only consider lhs (op_idx=0)
            # from collections import defaultdict

            # input_consumers = defaultdict(set)
            # for input_idx, input_node in enumerate(inputs):
            #     out_edges = io_sub.out_edges(input_node)
            #     for _, dst in out_edges:
            #         input_consumers[input_idx].add(dst)
            # print("input_consumers", input_consumers)
            # input_consumer_single = {
            #     input_idx: list(consumers)[0] for input_idx, consumers in input_consumers.items() if len(consumers) == 1
            # }
            # print("input_consumer_single", input_consumer_single)
            # consumer_is_comutable = {
            #     consumer: io_sub.nodes[consumer]["properties"]["isCommutable"]
            #     for consumer in set(input_consumer_single.values())
            # }
            # print("consumer_is_comutable", consumer_is_comutable)
            # consumer_input_nodes = {
            #     consumer: set([src for src, _ in io_sub.in_edges(consumer)])
            #     for consumer in set(input_consumer_single.values())
            # }
            # print("consumer_input_nodes", consumer_input_nodes)
            # consumer_input_nodes_ok = {
            #     consumer: set([input_node in inputs for input_node in input_nodes])
            #     for consumer, input_nodes in consumer_input_nodes.items()
            # }
            # print("consumer_input_nodes_ok", consumer_input_nodes_ok)
            # input("!!!")

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
                        new_io_sub = GF.subgraph(new_io_sub_nodes).copy()
                        for src, dst, dat in io_sub.in_edges(k, data=True):
                            assert src in io_sub.nodes
                            # print("src", src)
                            # print("dst", dst)
                            # print("dat", dat)
                            new_io_sub.add_edge(src, dst, **dat)
                            # print(f"{new_input_node_id} -> {dst}")
                        # print("new_io_sub_nodes", new_io_sub_nodes)
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
                        for enc_size in settings.filters.allowed_enc_sizes:
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
                        subs_df.loc[new_sub_id] = new_sub_data
                        add_hash_attr(new_sub)
                        add_hash_attr(new_io_sub)
                        add_hash_attr(new_io_sub, attr_name="hash_attr_ignore_const", ignore_const=True)
                        subs.append(new_sub)
                        io_subs.append(new_io_sub)
                        # if new_sub_id == 12:
                        #     print("sub_data", sub_data)
                        #     print("new_sub_data_", new_sub_data)
                        #     print("io_sub", io_sub, io_sub.nodes, io_sub.edges)
                        #     print("new_io_sub", new_io_sub, new_io_sub.nodes, new_io_sub.edges)
                        #     print("alias", k)
                        #     input("!")
                        # new.append(None)
                        # input("||")
            # print("new", new)
    if settings.enable_variation_auto_imm:
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
            # print("inputs", inputs)
            constants = sub_data["ConstantNodes"]
            # print("constants", constants)
            constant_min_bits = sub_data["ConstantMinBits"]
            # print("constant_min_bits", constant_min_bits)
            constant_signs = sub_data["ConstantSigns"]
            # print("constant_signs", constant_signs)
            # TODO: elen
            enc_bits_left = int(sub_data["EncodingBitsLeft (32 bits)"])
            # print("enc_bits_left", enc_bits_left)
            if enc_bits_left == 0:
                continue
            outputs = sub_data["OutputNodes"]
            # print("outputs", outputs)
            # nodes = sub_data["Nodes"]
            # print("nodes", nodes)
            # instrs = sub_data["Instrs"]
            # print("instrs", instrs)
            # print("!", [(inp, [instrs[nodes.index(x[1])] for x in io_sub.out_edges(inp)]) for inp in inputs])
            # print(
            #     "!",
            #     [
            #         (
            #             inp,
            #             [
            #                 (x[2]["properties"]["op_idx"], io_sub.nodes[x[1]]["label"])
            #                 for x in io_sub.out_edges(inp, data=True)
            #             ],
            #         )
            #         for inp in inputs
            #     ],
            # )
            temp = [
                (
                    inp,
                    [
                        (x[2]["properties"]["op_idx"], io_sub.nodes[x[1]]["label"])
                        for x in io_sub.out_edges(inp, data=True)
                    ],
                )
                for inp in constants
            ]
            # print("temp", temp, len(temp))
            # TODO: drop suffix check?
            temp2 = [x for x in temp if len(x[1]) == 1 and x[1][0][0] == 1 and x[1][0][1][-1] == "I"]
            # print("temp2", temp2, len(temp2))
            temp3 = [
                (x[0], x[1][0][1], constant_min_bits[constants.index(x[0])], constant_signs[constants.index(x[0])])
                for x in temp2
            ]
            # print("temp3", temp3)
            # input(">>>")
            auto_imm_strategies = [
                ImmStrategy.DEFAULT_BITS,
                ImmStrategy.MIN_REQUIRED_BITS,
                # ImmStrategy.MIN_REQUIRED_BITS_P1,  # +1
                # ImmStrategy.MIN_REQUIRED_BITS_P2,  # +2 ...
                ImmStrategy.NEXT_SUPPORTED_BITS,
                ImmStrategy.MAX_SUPPORTED_BITS,
                ImmStrategy.MAX_AVAILABLE_BITS,
            ]

            def apply_imm_strategy(
                strategy: ImmStrategy, bits: int, instr: str = None, signed: bool = False, available_bits: int = None
            ):
                supported_bits = [5, 7, 10, 12]
                # TODO: optionally uimm -> simm?
                if strategy == ImmStrategy.MIN_REQUIRED_BITS:
                    ret = bits
                elif strategy == ImmStrategy.DEFAULT_BITS:
                    assert instr is not None  # TODO: check name
                    ret = 12
                elif strategy == ImmStrategy.NEXT_SUPPORTED_BITS:
                    suitable_bits = [x for x in supported_bits if x >= bits]
                    # print("suitable_bits", suitable_bits)
                    if len(suitable_bits) == 0:
                        return None
                    ret = min(suitable_bits)
                elif strategy == ImmStrategy.MAX_SUPPORTED_BITS:
                    max_ = max(supported_bits)
                    if max_ < bits:
                        return None
                    ret = max_
                elif strategy == ImmStrategy.MAX_AVAILABLE_BITS:
                    assert available_bits is not None, "Undefined available_bits"
                    if available_bits == 0.0:
                        return None
                    ret = available_bits
                else:
                    assert False, f"Unknown ImmStrategy: {strategy}"
                assert ret > 0, "Expected non-zero imm bits"
                if available_bits is not None:
                    # assert ret <= available_bits, "Imm bits exteed available encoding bits"
                    if ret > available_bits:
                        return None
                return ret

            # TODO: handle signed
            for node, instr, required_imm_bits, signed in temp3:
                # print("node", node)
                # print("required_imm_bits", required_imm_bits)
                # print("signed", signed)
                strategy_bits = {
                    strategy: apply_imm_strategy(
                        strategy, required_imm_bits, instr=instr, signed=signed, available_bits=enc_bits_left
                    )
                    for strategy in auto_imm_strategies
                }
                # TODO: filter None
                # print("strategy_bits", strategy_bits)
                unique_bits = list(set(strategy_bits.values()))
                # print("unique_bits", unique_bits)
                unique_bits = [x for x in unique_bits if x is not None]
                # print("unique_bits_", unique_bits)
                if len(unique_bits) == 0:
                    continue
                sorted_bits = list(sorted(unique_bits))
                # print("sorted_bits", sorted_bits)
                const_pos = constants.index(node)
                # print("const_pos", const_pos)

                def get_next_imm_name_by_idx(imm_count: int = None, base: str = "imm", signed: Optional[bool] = None):
                    idx = imm_count
                    prefix = "" if signed is None else ("s" if signed else "u")
                    ret = prefix + base
                    if idx != 0:
                        ret += str(idx)
                    return ret

                def get_next_imm_name_by_size(
                    bits: int = None, base: str = "imm", signed: Optional[bool] = None, used: Optional[List[str]] = None
                ):
                    prefix = "" if signed is None else ("s" if signed else "u")
                    ret = prefix + base + str(int(bits))
                    assert used is not None
                    assert ret not in used, "Imm name not available"
                    # TODO: fallback to alternative names
                    return ret

                for imm_bits in sorted_bits:
                    # print("imm_bits", imm_bits)
                    BY_IDX = False
                    operand_types = sub_data["OperandTypes"]
                    operand_names = sub_data["OperandNames"]
                    if BY_IDX:
                        imm_count = operand_types.count("IMM")
                        # print("imm_count", imm_count)
                        imm_name = get_next_imm_name_by_idx(imm_count, signed=signed)
                    else:  # BY_SIZE
                        imm_names = [operand_names[i] for i, x in enumerate(operand_types) if x == "IMM"]
                        # print("imm_names", imm_names)
                        imm_name = get_next_imm_name_by_size(imm_bits, signed=signed, used=imm_names)
                    # print("imm_name", imm_name)
                    new_sub = sub.copy()
                    # print("new_sub", new_sub)
                    import copy

                    # new_sub_data = sub_data.copy()
                    # new_sub_data = copy.deepcopy(sub_data)
                    import pandas as pd

                    new_sub_data = pd.Series(copy.deepcopy(sub_data.to_dict()))
                    new_io_sub_nodes = [x for x in io_sub.nodes if x != node]
                    print("new_io_sub_nodes", new_io_sub_nodes)
                    constant_node_data = GF.nodes[node]
                    # new_constant_node_data = constant_node_data.copy()
                    new_constant_node_data = copy.deepcopy(constant_node_data)
                    new_constant_node_id = max(GF.nodes) + 1
                    new_constant_node_data["label"] = imm_name
                    new_constant_node_data["op_type"] = "imm"
                    new_constant_node_data["properties"]["op_type"] = "imm"
                    new_constant_node_data["properties"]["name"] = imm_name
                    GF.add_node(new_constant_node_id, **new_constant_node_data)
                    new_io_sub_nodes.append(new_constant_node_id)
                    for src, dst, dat in io_sub.out_edges(node, data=True):
                        assert dst in io_sub.nodes
                        print("src", src)
                        print("dst", dst)
                        print("dat", dat)
                        GF.add_edge(new_constant_node_id, dst, **dat)
                    new_io_sub = GF.subgraph(new_io_sub_nodes).copy()
                    outputs = sub_data["OutputNodes"]
                    # fix missing out edges
                    print("outputs", outputs)
                    for outp in outputs:
                        for src, dst, dat in io_sub.in_edges(outp, data=True):
                            assert src in io_sub.nodes
                            new_io_sub.add_edge(src, outp, **dat)
                    # for src, dst, dat in io_sub.out_edges(node, data=True):
                    #     assert dst in io_sub.nodes
                    #     print("src", src)
                    #     print("dst", dst)
                    #     print("dat", dat)
                    #     new_io_sub.add_edge(new_constant_node_id, dst, **dat)
                    # print("\n" * 10)
                    # print("new_constant_node_data", new_constant_node_data)
                    # print("new_constant_node_id", new_constant_node_id)
                    # print("111", GF.nodes[new_constant_node_id])
                    # print("222", new_io_sub.nodes[new_constant_node_id])
                    # print("new_sub_data", new_sub_data)
                    # print("new_io_sub.nodes", new_io_sub.nodes)
                    # print("\n" * 10)
                    # input(">>>")
                    # TODO: provide helper class to convert operands into sub_data
                    """
                    #Operands                                                   NaN
                    OperandNames                                     [rd, rd2, rs1]
                    OperandNodes                                      [176, 177, 4]
                    OperandDirs                                      [OUT, OUT, IN]
                    OperandTypes                                    [REG, REG, REG]
                    OperandRegClasses                               [GPR, GPR, GPR]
                    OperandEncBits                                        [5, 5, 5]
                    OperandEncBitsSum                                          15.0
                    """
                    parent = i
                    # TODO: update NODES in sub_data and new_io_sub
                    new_sub_data["Parent"] = parent
                    new_sub_data["#Operands"] += 1
                    new_sub_data["Variations"] |= Variation.AUTO_IMM
                    new_sub_data["OperandNames"].append(imm_name)
                    new_sub_data["OperandNodes"].append(new_constant_node_id)
                    new_sub_data["OperandDirs"].append("IN")
                    new_sub_data["OperandTypes"].append("IMM")
                    new_sub_data["OperandRegClasses"].append(None)
                    new_sub_data["OperandEncBits"].append(imm_bits)
                    new_sub_data["OperandEncBitsSum"] += imm_bits
                    # new_sub_data["#InputNodes"] += 1
                    # new_sub_data["InputNodes"].append(new_constant_node_id)
                    new_sub_data["#ConstantNodes"] -= 1
                    new_sub_data["ConstantNodes"] = [
                        x for i, x in enumerate(new_sub_data["ConstantNodes"]) if i != const_pos
                    ]
                    new_sub_data["ConstantValues"] = [
                        x for i, x in enumerate(new_sub_data["ConstantValues"]) if i != const_pos
                    ]
                    new_sub_data["ConstantMinBits"] = [
                        x for i, x in enumerate(new_sub_data["ConstantMinBits"]) if i != const_pos
                    ]
                    for enc_size in settings.filters.allowed_enc_sizes:
                        enc_bits_left, enc_weight, enc_footprint = calc_encoding_footprint(
                            new_sub_data["OperandEncBitsSum"], enc_size
                        )
                        new_sub_data[f"EncodingBitsLeft ({enc_size} bits)"] = enc_bits_left
                        new_sub_data[f"EncodingWeight ({enc_size} bits)"] = enc_weight
                        new_sub_data[f"EncodingFootprint ({enc_size} bits)"] = enc_footprint
                    new_sub_id = len(io_subs)
                    new_sub_data["result"] = new_sub_id
                    subs_df.loc[new_sub_id] = new_sub_data
                    add_hash_attr(new_sub)
                    add_hash_attr(new_io_sub)
                    add_hash_attr(new_io_sub, attr_name="hash_attr_ignore_const", ignore_const=True)
                    subs.append(new_sub)
                    io_subs.append(new_io_sub)
