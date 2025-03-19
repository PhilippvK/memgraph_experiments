import logging

from tqdm import tqdm

from ..hash import calc_full_hash, calc_global_hash

logger = logging.getLogger("create_fullhash")


def create_fullhash(settings, subs, io_subs, subs_df, global_df):
    logger.info("Creating FullHashes...")
    global_hash = calc_global_hash(global_df)
    # print("global_hash", global_hash)
    # input(">")
    # for i, io_sub in enumerate(tqdm(io_subs, disable=not settings.progress)):
    filtered_subs_df = subs_df[(subs_df["Status"] & settings.write.gen_flt) > 0].copy()
    io_subs_iter = [(i, io_sub) for i, io_sub in enumerate(io_subs) if i in filtered_subs_df.index]
    for i, io_sub in tqdm(io_subs_iter, disable=not settings.progress):
        # sub = subs[i]
        sub_data = subs_df.iloc[i]

        full_hash = calc_full_hash(sub_data)
        # print("full_hash", full_hash)
        subs_df.loc[i, "FullHash"] = full_hash
        subs_df.loc[i, "GlobalHash"] = global_hash
