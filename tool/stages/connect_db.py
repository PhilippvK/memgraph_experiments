import os
import logging

from ..settings import Settings
from ..memgraph import connect_memgraph

logger = logging.getLogger("connect_db")


def connect_db(settings: Settings):
    logger.info("Connecting to DB...")
    default_memgraph_host = os.environ.get("MEMGRAPH_HOST")
    default_memgraph_port = os.environ.get("MEMGRAPH_PORT")
    if default_memgraph_port:
        default_memgraph_port = int(default_memgraph_port)
    driver = connect_memgraph(
        default_memgraph_host or settings.memgraph.host,
        default_memgraph_port or settings.memgraph.port,
        user="",
        password="",
    )
    return driver
