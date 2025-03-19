import logging

import networkx as nx
import numpy as np
from tqdm import tqdm

logger = logging.getLogger("schedule_subs")


def schedule_subs(settings, io_subs, subs_df, io_isos):
    logger.info("Scheduling Subs...")
    # Very coarse measure to find longest path in subgraph (between inputs and outputs)
    subs_df["ScheduleLength"] = np.nan
    io_subs_iter = [(i, io_sub) for i, io_sub in enumerate(io_subs) if i not in io_isos]
    # for i, io_sub in enumerate(tqdm(io_subs, disable=not settings.progress)):
    for i, io_sub in tqdm(io_subs_iter, disable=not settings.progress):
        # if i in io_isos:
        #     continue
        # print("i", i)
        # print("io_sub", io_sub)
        # print("io_sub.nodes", io_sub.nodes)
        # sub_data = subs_df.iloc[i]
        # print("sub_data", sub_data)
        inputs = subs_df.loc[i, "InputNodes"]
        # inputs = subs_df.loc[i, "InputNames"]
        # print("inputs", inputs)
        outputs = subs_df.loc[i, "OutputNodes"]
        terminators = subs_df.loc[i, "TerminatorNodes"]
        # print("outputs", outputs)

        def estimate_schedule_length(io_sub, ins, ends):
            # TODO: allow resource constraints (regfile ports, alus, ...)
            # print("io_sub", io_sub)
            # print("ins", ins)
            # print("ends", ends)
            lengths = []
            for inp in ins:
                lengths_ = [
                    nx.shortest_path_length(io_sub, source=inp, target=outp)
                    for outp in ends
                    if nx.has_path(io_sub, inp, outp)
                ]
                # print("lengths_", lengths_)
                lengths += lengths_
            # print("lengths", lengths)
            # TODO: handle None?
            # print("lengths", lengths)
            assert len(lengths) > 0  # TODO: investigate
            return max(lengths)

        ends = list(set(outputs) | set(terminators))
        if len(inputs) == 0 or len(ends) == 0:  # TODO: handle properly
            length = 1
        else:
            length = estimate_schedule_length(io_sub, inputs, ends)
        # print("length", length)
        #  TODO
        subs_df.loc[i, "ScheduleLength"] = length
