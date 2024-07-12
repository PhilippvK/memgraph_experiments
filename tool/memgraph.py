import logging
from neo4j import GraphDatabase

logger = logging.getLogger("memgraph")


def connect_memgraph(host, port, user="", password=""):
    driver = GraphDatabase.driver(f"bolt://{host}:{port}", auth=(user, password))
    return driver


def run_query(driver, query):
    # TODO: use logging module
    logger.debug("QUERY> %s", query)
    return driver.session().run(query)
