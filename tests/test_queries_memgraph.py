"""Run with MEMGRAPH_TEST_URI set; fixture writes are rolled back."""
import itertools
import os
import unittest
import uuid

from tool.queries import generate_candidates_query
from tool.types import RegisterClass


@unittest.skipUnless(os.environ.get('MEMGRAPH_TEST_URI'), 'MEMGRAPH_TEST_URI is not set')
class MemgraphQueriesTest(unittest.TestCase):
    def test_candidates_match_exhaustive_reference(self):
        from neo4j import GraphDatabase
        session = 'query-test-' + uuid.uuid4().hex + "'\\"
        edges = [(0, 2), (1, 2), (2, 3), (0, 3), (0, 4), (4, 3), (0, 5), (5, 3), (0, 6), (6, 3), (3, 0)]
        paths = []

        def visit(path):
            if len(path) > 1:
                paths.append(path)
            if len(path) < 4:
                for a, b in edges:
                    if a == path[-1] and b < 4 and b not in path:
                        visit(path + [b])
        for n in range(4):
            visit([n])
        with GraphDatabase.driver(os.environ['MEMGRAPH_TEST_URI'], auth=('', '')) as driver:
            with driver.session() as s, s.begin_transaction(timeout=15) as tx:
                for n in range(7):
                    tx.run('CREATE (:INSTR {session:$session, func_name:$func, stage:8, bb_id:0, '
                           'module_name:"m", compilation_id:"c", basic_block:"entry", name:$name, out_reg_class:$reg, op_type:"instruction", test_id:$id})',
                           session=session, func='f' if n != 5 else 'other', name='BAD' if n == 4 else 'ADD',
                           reg='fpr32' if n == 6 else 'gpr', id=n).consume()
                for a, b in edges:
                    tx.run('MATCH (a:INSTR {session:$session, test_id:$a}), (b:INSTR {session:$session, test_id:$b}) '
                           'CREATE (a)-[:DFG]->(b)', session=session, a=a, b=b).consume()
                for width in (1, 2, 3):
                    for si, so in ((True, False), (False, True), (True, True)):
                        expected = set()
                        for combo in itertools.product(paths, repeat=width):
                            if si and len({p[0] for p in combo}) > 1:
                                continue
                            if so and len({p[-1] for p in combo}) > 1:
                                continue
                            nodes = frozenset(itertools.chain.from_iterable(combo))
                            if 2 <= len(nodes) <= 3:
                                expected.add(nodes)
                        q = generate_candidates_query(session, 'f', None, 1, 3, width, ['BAD'], [],
                                                      min_nodes=2, max_nodes=3, shared_input=si, shared_output=so,
                                                      allowed_reg_classes=RegisterClass.GPR)
                        actual = set()
                        for record in tx.run(q):
                            actual.add(frozenset(n['test_id'] for p in record.values() for n in p.nodes))
                        self.assertEqual(expected, actual, (width, si, so))
                tx.rollback()  # Neo4j commits successful transaction contexts by default.


    def test_limit_after_minimum_node_filter(self):
        from neo4j import GraphDatabase
        label = 'limit-test-' + uuid.uuid4().hex
        with GraphDatabase.driver(os.environ['MEMGRAPH_TEST_URI'], auth=('', '')) as driver:
            with driver.session() as session, session.begin_transaction(timeout=15) as tx:
                tx.run('CREATE (a:INSTR {session:$label, func_name:"f", stage:8, bb_id:0, module_name:"m", compilation_id:"c"}), '
                       '(b:INSTR {session:$label, func_name:"f", stage:8, bb_id:0, module_name:"m", compilation_id:"c"}), '
                       '(c:INSTR {session:$label, func_name:"f", stage:8, bb_id:0, module_name:"m", compilation_id:"c"}), '
                       '(a)-[:DFG]->(b), (b)-[:DFG]->(c)', label=label).consume()
                for limit in (0, 1, 3):
                    query = generate_candidates_query(label, 'f', None, 1, 2, 1, [], [],
                                                      min_nodes=3, max_nodes=3, limit=limit)
                    records = list(tx.run(query))
                    self.assertEqual(len(records), 0 if limit == 0 else 1)
                    for record in records:
                        self.assertEqual(len(record['p0'].nodes), 3)
                tx.rollback()

    def test_positive_predicate_caps_match_exhaustive_paths(self):
        from neo4j import GraphDatabase
        label = 'caps-' + uuid.uuid4().hex
        edges = [(0, 1), (1, 2), (0, 2), (2, 3), (1, 3)]
        flags = [dict(mayLoad=True), dict(mayStore=True),
                 dict(mayLoad=True, mayStore=True), dict(isBranch=True)]
        paths = []
        def visit(path):
            if len(path) > 1:
                paths.append(path)
            if len(path) < 4:
                for a, b in edges:
                    if a == path[-1] and b not in path:
                        visit(path + [b])
        for n in range(4):
            visit([n])
        with GraphDatabase.driver(os.environ['MEMGRAPH_TEST_URI'], auth=('', '')) as driver:
            with driver.session() as session, session.begin_transaction(timeout=15) as tx:
                try:
                    for n, properties in enumerate(flags):
                        tx.run('CREATE (n:INSTR {session:$label, func_name:"f", stage:8, bb_id:0, '
                               'module_name:"m", compilation_id:"c", test_id:$id}) SET n += $props',
                               label=label, id=n, props=properties).consume()
                    for a, b in edges:
                        tx.run('MATCH (a:INSTR {session:$label,test_id:$a}), '
                               '(b:INSTR {session:$label,test_id:$b}) CREATE (a)-[:DFG]->(b)',
                               label=label, a=a, b=b).consume()
                    for width in (1, 2):
                        for caps in ((1, 1, 2, 1), (0, 1, 1, 0), (2, 2, 3, 1)):
                            for enabled in (False, True):
                                expected = set()
                                for combo in itertools.product(paths, repeat=width):
                                    if len({p[-1] for p in combo}) != 1:
                                        continue
                                    nodes = frozenset(itertools.chain.from_iterable(combo))
                                    loads = sum(bool(flags[n].get('mayLoad')) for n in nodes)
                                    stores = sum(bool(flags[n].get('mayStore')) for n in nodes)
                                    branches = sum(bool(flags[n].get('isBranch')) for n in nodes)
                                    if not enabled or all(v <= c for v, c in zip((loads, stores, loads+stores, branches), caps)):
                                        expected.add(nodes)
                                query = generate_candidates_query(label, 'f', '%bb.0', 1, 3, width, [], [],
                                    module_name='m', compilation_id='c', query_level_filtering=enabled,
                                    max_loads=caps[0], max_stores=caps[1], max_mems=caps[2], max_branches=caps[3])
                                actual = {frozenset(n['test_id'] for p in record.values() for n in p.nodes)
                                          for record in tx.run(query)}
                                self.assertEqual(expected, actual, (width, caps, enabled))
                finally:
                    tx.rollback()
