import logging

# import h5py
# import numpy as np


from ..enums import ExportFormat

logger = logging.getLogger("write_dfs")


def write_dfs(settings, subs_df, global_df, index_artifacts):
    filtered_subs_df = subs_df[(subs_df["Status"] & settings.write.df_flt) > 0].copy()
    logger.info("Exporting Global DataFrame...")
    if settings.write.df_fmt & ExportFormat.CSV:
        global_df.to_csv(settings.out_dir / "global.csv")
    if settings.write.df_fmt & ExportFormat.PKL:
        global_df.to_pickle(settings.out_dir / "global.pkl")
    logger.info("Exporting Subs DataFrame...")
    if settings.write.df_fmt & ExportFormat.CSV:
        filtered_subs_df.to_csv(settings.out_dir / "subs.csv")
    if settings.write.df_fmt & ExportFormat.PKL:
        filtered_subs_df.to_pickle(settings.out_dir / "subs.pkl")
        index_artifacts[None]["subs"] = settings.out_dir / "subs.pkl"
