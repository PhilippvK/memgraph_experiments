import logging

from tqdm import tqdm

from ..enums import ExportFilter
from ..branches import BRANCHES, BRANCH_OFFSET

logger = logging.getLogger("filter_subs_operands")


def filter_subs_operands(settings, subs, subs_df):
    filtered_operands = set()
    logger.info("Filtering subgraphs (Operands)...")
    filtered_subs_df = subs_df[(subs_df["Status"] & settings.write.gen_flt.value) > 0].copy()
    subs_iter = [(i, sub) for i, sub in enumerate(subs) if i in filtered_subs_df.index]
    # for i, sub in enumerate(tqdm(subs, disable=not settings.progress)):
    # print("len(subs_iter)", len(subs_iter))
    for i, sub in tqdm(subs_iter, disable=not settings.progress):
        sub_data = subs_df.iloc[i]
        # valids = []
        operand_dirs = sub_data["OperandDirs"]
        operand_types = sub_data["OperandTypes"]
        # ignore imm operands
        reg_operand_dirs = [x for i, x in enumerate(operand_dirs) if operand_types[i] == "REG"]
        # print("operand_dirs")
        num_in_operands = reg_operand_dirs.count("IN")
        # print("num_in_operands", num_in_operands)
        num_out_operands = reg_operand_dirs.count("OUT")
        # print("num_out_operands", num_out_operands)
        num_inout_operands = reg_operand_dirs.count("INOUT")
        # print("num_inout_operands", num_inout_operands)
        valid = True
        if num_in_operands > settings.filters.max_in_operands:
            valid = False
        if (num_out_operands + num_inout_operands) > settings.filters.max_out_operands:
            valid = False
        if num_inout_operands > settings.filters.max_inout_operands:
            valid = False
        if "IMM" in operand_types:
            pass
        if sub_data["#Branches"]:
            branch_nodes = [n for n in sub.nodes if sub.nodes[n]["properties"].get("isBranch")]
            valid = valid and (
                len(branch_nodes) == 1
                and sub.nodes[branch_nodes[0]]["properties"].get("name") in BRANCHES
                and num_out_operands == num_inout_operands == 0
                and num_in_operands <= 2
                and list(sub_data["OperandNames"]).count(BRANCH_OFFSET) == 1
                and sub_data["#Mems"] == 0
            )
        if sub_data["#Mems"]:
            valid = valid and (
                sub_data["#Mems"] == 1 and not sub_data["#Branches"]
                and num_inout_operands == 0
                and num_out_operands == (0 if sub_data["#Stores"] else 1)
                and num_in_operands <= 3
            )
        if settings.filters.cfu:
            valid = valid and (
                operand_types == ["REG"] * len(operand_types)
                and sub_data["OperandRegClasses"] == ["GPR"] * len(operand_types)
                and sub_data["OperandEncBits"] == [5] * len(operand_types)
                and num_in_operands <= 2 and num_out_operands == 1 and num_inout_operands == 0
                and all(sub_data[field] == 0 for field in
                        ("#Loads", "#Stores", "#Mems", "#Branches", "#Terminators"))
            )
        if not valid:
            filtered_operands.add(i)
        # input(">>>")
    # print("filtered_operands", filtered_operands)
    # input(">>>2")
    subs_df.loc[list(filtered_operands), "Status"] = ExportFilter.FILTERED_OPERANDS.value
