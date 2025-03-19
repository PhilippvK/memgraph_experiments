import logging

from ..graph_utils import memgraph_to_nx

logger = logging.getLogger("convert_func")


def convert_func(func_results):
    logger.info("Converting func graph to NX...")
    GF = memgraph_to_nx(func_results)
    # print("GF", GF)
    return GF
