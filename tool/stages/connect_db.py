import logging

from ..settings import Settings
from ..memgraph import connect_memgraph

logger = logging.getLogger("connect_db")


def connect_db(settings: Settings):
    logger.info("Connecting to DB...")
    driver = connect_memgraph(settings.memgraph.host, settings.memgraph.port, user="", password="")
    return driver
