import logging

from tqdm import tqdm

from ..enums import ExportFilter

logger = logging.getLogger("filter_subs_operands")


def filter_subs_operands(settings, subs, subs_df):
    filtered_operands = set()
    logger.info("Filtering subgraphs (Operands)...")
    filtered_subs_df = subs_df[(subs_df["Status"] & settings.write.gen_flt) > 0].copy()
    subs_iter = [(i, sub) for i, sub in enumerate(subs) if i in filtered_subs_df.index]
    # for i, sub in enumerate(tqdm(subs, disable=not settings.progress)):
    # print("len(subs_iter)", len(subs_iter))
    for i, sub in tqdm(subs_iter, disable=not settings.progress):
        sub_data = subs_df.iloc[i]
        # valids = []
        operand_dirs = sub_data["OperandDirs"]
        # print("operand_dirs")
        num_in_operands = operand_dirs.count("IN")
        # print("num_in_operands", num_in_operands)
        num_out_operands = operand_dirs.count("OUT")
        # print("num_out_operands", num_out_operands)
        num_inout_operands = operand_dirs.count("INOUT")
        # print("num_inout_operands", num_inout_operands)
        valid = True
        if num_in_operands > settings.filters.max_in_operands:
            valid = False
        if (num_out_operands + num_inout_operands) > settings.filters.max_out_operands:
            valid = False
        if num_inout_operands > settings.filters.max_inout_operands:
            valid = False
        # print("valid", valid)
        if not valid:
            filtered_operands.add(i)
        # input(">>>")
    # print("filtered_operands", filtered_operands)
    # input(">>>2")
    subs_df.loc[list(filtered_operands), "Status"] = ExportFilter.FILTERED_OPERANDS
