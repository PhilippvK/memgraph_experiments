import logging

from ..enums import ExportFormat
from ..index import write_index_file

logger = logging.getLogger("write_index")


def write_index(settings, GF, G, subs, io_subs, trees, subs_df, global_df, index_artifacts):
    filtered_subs_df = subs_df[(subs_df["Status"] & settings.write.index_flt.value) > 0].copy()
    global_df["candidate_count"] = len(filtered_subs_df)
    logger.info("Writing Index File...")
    if settings.write.index_fmt & ExportFormat.YAML:
        write_index_file(
            settings,
            settings.out_dir / "index.yml",
            GF,
            G,
            subs,
            io_subs,
            trees,
            filtered_subs_df,
            global_df,
            index_artifacts,
        )
