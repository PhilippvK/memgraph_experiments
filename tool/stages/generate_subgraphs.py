import logging

from tqdm import tqdm
import pandas as pd


logger = logging.getLogger("generate_subgraphs")


def generate_subgraphs(settings, results, G):
    logger.info("Generating subgraphs...")
    subs = []
    for i, result in enumerate(tqdm(results, disable=not settings.progress)):
        # print("result", result, dir(result))
        # print("result.data", result.data())
        # print("result.value", result.value())
        nodes_ = set()
        # path = result.value()
        for p, path in enumerate(result):
            # print("path", path)
            # print("path", path, dir(path))
            nodes__ = path.nodes
            # print("nodes__", nodes__[0].element_id)
            # 'count', 'data', 'get', 'index', 'items', 'keys', 'value', 'values'
            nodes_ |= {int(n.element_id) for n in nodes__}
        # print("nodes_", nodes_)
        G_ = G.subgraph(nodes_)
        # G_ = nx.subgraph_view(G, filter_node=lambda x: x in nodes_)
        # print("G_", G_)
        count = subs.count(G_)
        if count > 0:
            pass
        subs.append(G_)
    if settings.write.query_metrics:
        graph = results.graph()
        num_rows = len(subs)
        num_nodes = len(graph.nodes)
        num_edges = len(graph.relationships)
        logger.info("Exporting query metrics...")
        query_metrics_data = {"num_rows": num_rows, "num_nodes": num_nodes, "num_edges": num_edges}
        query_metrics_df = pd.DataFrame([query_metrics_data])
        query_metrics_df.to_csv(settings.out_dir / "query_metrics.csv", index=False)
    return subs
