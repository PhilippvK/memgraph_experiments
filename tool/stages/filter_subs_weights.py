import logging

from tqdm import tqdm

from ..enum import ExportFilter

logger = logging.getLogger("filter_subs_weights")


def filter_subs_weights(settings, subs, subs_df):
    filtered_weights = set()
    logger.info("Filtering subgraphs (Weights)...")
    filtered_subs_df = subs_df[(subs_df["Status"] & settings.write.gen_flt) > 0].copy()
    subs_iter = [(i, sub) for i, sub in enumerate(subs) if i in filtered_subs_df.index]
    # for i, sub in enumerate(tqdm(subs, disable=not settings.progress)):
    for i, sub in tqdm(subs_iter, disable=not settings.progress):
        sub_data = subs_df.iloc[i]
        iso_weight = sub_data["IsoWeight"]
        if iso_weight < settings.filters.min_iso_weight:
            filtered_weights.add(i)
    # print("filtered_weights", filtered_weights)
    subs_df.loc[list(filtered_weights), "Status"] = ExportFilter.FILTERED_WEIGHTS
