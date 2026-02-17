import logging

from tqdm import tqdm

from ..enums import InstrPredicate
from ..pred import detect_predicates as detect_predicates_

logger = logging.getLogger("detect_predicates")


def detect_predicates(settings, subs, subs_df, io_isos):
    logger.info("Detecting Predicates...")
    subs_df["Predicates"] = InstrPredicate.NONE.value
    subs_iter = [(i, sub) for i, sub in enumerate(subs) if i not in io_isos]
    # for i, sub in enumerate(tqdm(subs, disable=not settings.progress)):
    for i, sub in tqdm(subs_iter, disable=not settings.progress):
        # if i in io_isos:
        #     continue
        pred, num_loads, loads, num_stores, stores, num_terminators, terminators, num_branches, branches = (
            detect_predicates_(sub)
        )
        num_mems = num_loads + num_stores
        subs_df.loc[i, "Predicates"] = pred.value
        subs_df.loc[i, "#Loads"] = num_loads
        subs_df.loc[i, "#Stores"] = num_stores
        subs_df.loc[i, "#Mems"] = num_mems
        subs_df.loc[i, "#Terminators"] = num_terminators
        subs_df.loc[i, "#Branches"] = num_branches
        # TODO: maybe move to predicates detection?
        subs_df.at[i, "LoadNodes"] = list(loads)
        subs_df.at[i, "StoreNodes"] = list(stores)
        subs_df.at[i, "TerminatorNodes"] = list(terminators)
        subs_df.at[i, "BranchNodes"] = list(branches)
