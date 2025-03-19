import logging

from tqdm import tqdm

from ..enum import ExportFilter
from ..pred import check_predicates

logger = logging.getLogger("filter_subs")


def filter_subs(settings, subs, subs_df, io_isos):
    filtered_io = set()
    filtered_complex = set()
    filtered_simple = set()
    filtered_predicates = set()
    filtered_mem = set()
    filtered_branch = set()
    invalid = set()
    logger.info("Filtering subgraphs...")
    subs_iter = [(i, sub) for i, sub in enumerate(subs) if i not in io_isos]
    # for i, sub in enumerate(tqdm(subs, disable=not settings.progress)):
    for i, sub in tqdm(subs_iter, disable=not settings.progress):
        # if i in io_isos:
        #     continue
        # print("===========================")
        # print("i, sub", i, sub)
        num_nodes = len(sub.nodes)
        sub_data = subs_df.iloc[i]
        # inputs = sub_data["InputNodes"]
        num_inputs = int(sub_data["#InputNodes"])
        # outputs = sub_data["OutputNodes"]
        num_outputs = int(sub_data["#OutputNodes"])
        if num_inputs == 0:  # or num_outputs == 0:
            # TODO heck if branches and stores have outputs?
            invalid.add(i)
        elif (
            settings.filters.min_inputs <= num_inputs <= settings.filters.max_inputs
            and settings.filters.min_outputs <= num_outputs <= settings.filters.max_outputs
        ):
            pred = subs_df.loc[i, "Predicates"]
            if num_nodes > settings.filters.max_nodes:
                filtered_complex.add(i)
            elif num_nodes < settings.filters.min_nodes:
                filtered_simple.add(i)
            elif not check_predicates(pred, settings.filters.instr_predicates):
                # TODO: add predicates details to df in prerequisite step
                filtered_predicates.add(i)
            else:
                num_loads = subs_df.loc[i, "#Loads"]
                num_stores = subs_df.loc[i, "#Stores"]
                num_mems = subs_df.loc[i, "#Mems"]
                if (
                    num_loads > settings.filters.max_loads
                    or num_stores > settings.filters.max_stores
                    or num_mems > settings.filters.max_mems
                ):
                    filtered_mem.add(i)
                else:
                    num_branches = subs_df.loc[i, "#Branches"]
                    if num_branches > settings.filters.max_branches:
                        filtered_branch.add(i)
        else:
            filtered_io.add(i)
            # print("sub_data", sub_data)
            # input("FILTERED_IO")
    subs_df.loc[list(filtered_io), "Status"] = ExportFilter.FILTERED_IO
    subs_df.loc[list(filtered_complex), "Status"] = ExportFilter.FILTERED_COMPLEX
    subs_df.loc[list(filtered_simple), "Status"] = ExportFilter.FILTERED_SIMPLE
    subs_df.loc[list(filtered_predicates), "Status"] = ExportFilter.FILTERED_PRED
    subs_df.loc[list(filtered_mem), "Status"] = ExportFilter.FILTERED_MEM
    subs_df.loc[list(filtered_branch), "Status"] = ExportFilter.FILTERED_BRANCH
    subs_df.loc[list(invalid), "Status"] = ExportFilter.INVALID
