import logging
import re

from neo4j import Query

from ..settings import Settings
from ..memgraph import run_query
from ..queries import generate_func_query

logger = logging.getLogger("query_func")


def query_func(settings: Settings, driver):
    params = dict(label=settings.query.session, func=settings.query.func, stage=int(settings.query.stage),
                  module=settings.query.module_name, compilation=settings.query.compilation_id, bb_id=None)
    if settings.query.bb is not None:
        if not re.fullmatch(r'%bb\.(0|[1-9][0-9]*)', str(settings.query.bb)):
            raise ValueError("Expected a fixed BB ID spelled %bb.<integer>")
        params["bb_id"] = int(settings.query.bb[4:])
    with driver.session() as session:
        blocks = [dict(r) for r in session.run(Query("""MATCH (b:BB)
            WHERE b.session = $label AND b.func_name = $func AND b.stage = $stage
              AND ($module IS NULL OR b.module_name = $module)
              AND ($compilation IS NULL OR b.compilation_id = $compilation)
              AND ($bb_id IS NULL OR b.bb_id = $bb_id)
            RETURN b.module_name AS module, b.compilation_id AS compilation,
                   b.bb_id AS bb_id, b.name AS mir_name, b.bb_clone_id AS clone, b.bb_identity_version AS version""",
            timeout=settings.query.timeout_seconds), **params)]
    scopes = {(b["module"], b["compilation"]) for b in blocks}
    if len(scopes) != 1 or any(not m or not c for m, c in scopes):
        raise ValueError("Missing/ambiguous CDFG module or import; select a verified compilation")
    if any(b["version"] != 1 or b["clone"] != 0 or not isinstance(b["bb_id"], int) or b["bb_id"] < 0
           for b in blocks) or len({b["bb_id"] for b in blocks}) != len(blocks):
        raise ValueError("Invalid or duplicated fixed BB identities; regenerate the CDFG")
    settings.query.module_name, settings.query.compilation_id = next(iter(scopes))
    settings.query.mir_bb_name = blocks[0]["mir_name"] if settings.query.bb is not None else None
    query = generate_func_query(settings.query.session, settings.query.func, stage=settings.query.stage,
                                module_name=settings.query.module_name,
                                compilation_id=settings.query.compilation_id)
    if settings.write.queries:
        logger.info("Exporting queries...")
        with open(settings.out_dir / "query_func.cypher", "w") as f:
            f.write(query)
    return run_query(driver, query, timeout=settings.query.timeout_seconds)
