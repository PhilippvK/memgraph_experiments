import logging

# import h5py
# import numpy as np


from ..enums import ExportFilter, InstrPredicate

logger = logging.getLogger("finalize_df")


def finalize_df(subs_df):
    logger.info("Finalizing DataFrame...")

    subs_df["Status (str)"] = subs_df["Status"].apply(lambda x: str(ExportFilter(x)))
    subs_df["Predicates (str)"] = subs_df["Predicates"].apply(lambda x: str(InstrPredicate(x)))
