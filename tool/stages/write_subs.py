import logging
import pickle

from ..enums import ExportFormat
from ..export_utils import export_graphs
from ..graph_utils import graph_to_file

logger = logging.getLogger("write_subs")


def worker(i, sub, settings):
    ret = {}
    if settings.write.sub.fmt & ExportFormat.DOT:
        graph_to_file(sub, settings.out_dir / f"sub{i}.dot")
    if settings.write.sub.fmt & ExportFormat.PDF:
        graph_to_file(sub, settings.out_dir / f"sub{i}.pdf")
    if settings.write.sub.fmt & ExportFormat.PNG:
        graph_to_file(sub, settings.out_dir / f"sub{i}.png")
    if settings.write.sub.fmt & ExportFormat.PKL:
        with open(settings.out_dir / f"sub{i}.pkl", "wb") as f:
            pickle.dump(sub.copy(), f)
        ret["sub"] = settings.out_dir / f"sub{i}.pkl"
    return i, ret


def write_subs(settings, subs, subs_df, index_artifacts):
    logger.info("Exporting subgraphs...")
    filtered_subs_df = subs_df[(subs_df["Status"] & settings.write.sub.flt.value) > 0].copy()
    subs_iter = [(i, sub) for i, sub in enumerate(subs) if i in filtered_subs_df.index]
    export_graphs(worker, subs_iter, settings, settings.write.sub.fmt, index_artifacts)
