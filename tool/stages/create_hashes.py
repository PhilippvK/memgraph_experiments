import logging

import networkx as nx
from tqdm import tqdm

from ..hash import add_hash_attr


logger = logging.getLogger("create_hashes")


def create_hashes(settings, subs, io_subs, subs_df):
    logger.info("Creating SubHashes...")
    # Does not work for MultiDiGraphs?
    for i, io_sub in enumerate(tqdm(io_subs, disable=not settings.progress)):
        # print("i", i)
        # print("io_sub", io_sub)
        sub = subs[i]

        add_hash_attr(sub)
        add_hash_attr(io_sub)
        add_hash_attr(io_sub, attr_name="hash_attr_ignore_const", ignore_const=True)
        #     edge_attr = "hash_attr"
        node_attr = "hash_attr"
        # TODO: check if num iters and digest is fine?
        sub_hash = nx.weisfeiler_lehman_graph_hash(sub, node_attr=node_attr, iterations=3, digest_size=16)
        io_sub_hash = nx.weisfeiler_lehman_graph_hash(io_sub, node_attr=node_attr, iterations=3, digest_size=16)
        # print("sub_hash", sub_hash)
        # print("io_sub_hash", io_sub_hash)
        subs_df.loc[i, "SubHash"] = sub_hash
        subs_df.loc[i, "IOSubHash"] = io_sub_hash
