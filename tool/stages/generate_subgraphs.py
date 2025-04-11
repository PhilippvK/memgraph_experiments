import logging

from tqdm import tqdm
import pandas as pd


logger = logging.getLogger("generate_subgraphs")


def generate_subgraphs(settings, all_results, G):
    logger.info("Generating subgraphs...")
    subs = []
    num_rows = 0
    num_nodes = 0
    num_edges = 0
    if not isinstance(all_results, list):
        all_results = [all_results]
    for results in all_results:
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
            graph = results.graph()
            num_rows_ = len(subs)
            num_nodes_ = len(graph.nodes)
            num_edges_ = len(graph.relationships)
            num_rows += num_rows_
            num_nodes += num_nodes_
            num_edges += num_edges_

    if settings.write.query_metrics:
        logger.info("Exporting query metrics...")
        query_metrics_data = {"num_rows": num_rows, "num_nodes": num_nodes, "num_edges": num_edges}
        query_metrics_df = pd.DataFrame([query_metrics_data])
        query_metrics_df.to_csv(settings.out_dir / "query_metrics.csv", index=False)
    return subs
