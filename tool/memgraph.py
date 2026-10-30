import logging
import math

from neo4j import GraphDatabase, Query

logger = logging.getLogger("memgraph")


def connect_memgraph(host, port, user="", password=""):
    driver = GraphDatabase.driver(f"bolt://{host}:{port}", auth=(user, password))
    return driver


def run_query(driver, query, timeout=None):
    # TODO: use logging module
    logger.debug("QUERY> %s", query)
    if timeout is not None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("query timeout must be a positive finite number of seconds")
        query = Query(query, timeout=timeout)
    return driver.session().run(query)
