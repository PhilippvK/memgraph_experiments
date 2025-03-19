import logging

from ..settings import Settings
from ..memgraph import run_query
from ..queries import generate_candidates_query

logger = logging.getLogger("query_candidates")


def query_candidates(settings: Settings, driver):
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
    )
    if settings.write.queries:
        logger.info("Exporting queries...")
        with open(settings.out_dir / "query_candidates.cypher", "w") as f:
            f.write(query)
    results = run_query(driver, query)
    return results
