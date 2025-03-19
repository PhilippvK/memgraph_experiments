import logging
from math import log2, ceil

from tqdm import tqdm

from ..llvm_utils import parse_llvm_const_str

logger = logging.getLogger("analyze_const")


def analyze_const(settings, subs, io_subs, subs_df):
    logger.info("Analyzing constants...")
    filtered_subs_df = subs_df[(subs_df["Status"] & settings.write.gen_flt) > 0].copy()
    io_subs_iter = [(i, io_sub) for i, io_sub in enumerate(io_subs) if i in filtered_subs_df.index]
    # for i, sub in enumerate(tqdm(subs, disable=not settings.progress)):
    for i, io_sub in tqdm(io_subs_iter, disable=not settings.progress):
        # sub = subs[i]
        sub_data = subs_df.iloc[i]
        constants = sub_data["ConstantNodes"]
        # print("constants")
        # constant_names = []
        constant_values = []
        constant_signs = []
        constant_min_bits = []
        for constant_idx, j in enumerate(constants):
            # print("j", j)
            constant_node = io_sub.nodes[j]
            # print("constant_node", constant_node)
            constant_properties = constant_node["properties"]
            # print("constant_properties", constant_properties)
            op_type = constant_properties["op_type"]
            assert op_type == "constant"
            # name = f"const{constant_idx}"
            val_str = constant_properties["inst"]
            val, llvm_type, sign = parse_llvm_const_str(val_str)
            # print("sign", sign)

            min_bits = 1 if val == 0 else (ceil(log2(abs(val))) + 1)
            # print("min_bits", min_bits)
            # TODO: handle float!!!
            # constant_names.append(name)
            constant_values.append(val)
            constant_signs.append(sign)
            constant_min_bits.append(min_bits)
            # input("##")
            # TODO: do not hardcode
        # subs_df.at[i, "ConstantNames"] = constant_names
        subs_df.at[i, "ConstantValues"] = constant_values
        subs_df.at[i, "ConstantSigns"] = constant_signs
        subs_df.at[i, "ConstantMinBits"] = constant_min_bits
