import logging
import pickle

from ..graph_utils import graph_to_file
from ..enum import ExportFormat

logger = logging.getLogger("dump_func_graph")


def dump_func_graph(settings, GF, index_artifacts):
    logger.info("Exporting GF graph...")
    if settings.write.func_fmt & ExportFormat.DOT:
        graph_to_file(GF, settings.out_dir / "func.dot")
    if settings.write.func_fmt & ExportFormat.PDF:
        graph_to_file(GF, settings.out_dir / "func.pdf")
    if settings.write.func_fmt & ExportFormat.PNG:
        graph_to_file(GF, settings.out_dir / "func.png")
    if settings.write.func_fmt & ExportFormat.PKL:
        with open(settings.out_dir / "func.pkl", "wb") as f:
            pickle.dump(GF.copy(), f)
        index_artifacts[None]["func"] = settings.out_dir / "func.pkl"
