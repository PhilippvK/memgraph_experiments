# memgraph_experiments

Candidate queries support configurable edge lengths (`--min-path-length`,
`--max-path-length`) and path width (`--max-path-width`). Width is the number of
path slots; slots may repeat, so narrower candidates remain eligible. The Python
API supports sharing either or both endpoints; disconnected multi-path requests
are rejected. `simple_mode=True` requests one path.

Nodes are filtered during DFS expansion using the configured names, operand
kinds, register classes, graph scope, and optional MaxMISO index. Cycles are
excluded. Maximum node counts prune partial paths and path combinations.

`--limit-results` defaults to 100000 and limits traversal-order results **before**
sorting the selected sample by node count. Results are no longer random samples
or guaranteed globally largest candidates. `--query-timeout` defaults to 60
seconds and applies separately to each candidate query, including each MaxMISO
partition. The equivalent YAML keys under `query` are `limit_results` and
`timeout_seconds`. The Python query generator accepts `limit=None` for exhaustive
enumeration. Dense graphs and large widths can still generate many combinations;
a finite result limit does not replace the execution timeout.

Run regression tests from the parent workspace:

```sh
PYTHONPATH=memgraph_experiments venv/bin/python -m unittest discover -s memgraph_experiments/tests -v
```

Set `MEMGRAPH_TEST_URI=bolt://127.0.0.1:17687` to also run database integration
coverage. It creates its graph inside a transaction and rolls it back.
