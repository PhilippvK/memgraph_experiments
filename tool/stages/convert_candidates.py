import logging

from ..graph_utils import memgraph_to_nx

logger = logging.getLogger("convert_candidates")


def convert_candidates(results):
    logger.info("Converting candidates to NX...")
    G = memgraph_to_nx(results)
    return G
