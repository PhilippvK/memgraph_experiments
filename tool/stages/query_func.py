import logging

from ..settings import Settings
from ..memgraph import run_query
from ..queries import generate_func_query

logger = logging.getLogger("query_func")


def query_func(settings: Settings, driver):
    query_func = generate_func_query(settings.session, settings.func, stage=settings.stage)
    if settings.write.queries:
        logger.info("Exporting queries...")
        with open(settings.out_dir / "query_func.cypher", "w") as f:
            f.write(query_func)
    func_results = run_query(driver, query_func)
    return func_results
