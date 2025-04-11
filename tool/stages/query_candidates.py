import logging

from ..settings import Settings
from ..memgraph import run_query
from ..queries import generate_candidates_query

logger = logging.getLogger("query_candidates")


def query_candidates(settings: Settings, driver):
    maxmiso_idxs = settings.query.maxmisos
    if maxmiso_idxs is None:
        maxmiso_idxs = [None]
    all_results = []
    queries = []
    for maxmiso_idx in maxmiso_idxs:
        query = generate_candidates_query(
            settings.query.session,
            settings.query.func,
            settings.query.bb,
            settings.query.min_path_len,
            settings.query.max_path_len,
            settings.query.max_path_width,
            settings.query.ignore_names,
            settings.query.ignore_op_types,
            min_nodes=settings.filters.min_nodes,
            max_nodes=settings.filters.max_nodes,
            stage=settings.query.stage,
            limit=settings.query.limit_results,
            maxmiso_idx=maxmiso_idx,
        )
        queries.append(query)
        results = run_query(driver, query)
        all_results.append(results)
    if settings.write.queries:
        logger.info("Exporting queries...")
        with open(settings.out_dir / "query_candidates.cypher", "w") as f:
            queries_str = "\n".join(queries)
            f.write(queries_str)
    return all_results
