import unittest
from tool.queries import generate_candidates_query, generate_func_query
from tool.types import RegisterClass


class CandidateQueriesTest(unittest.TestCase):
    def query(self, **changes):
        args = dict(session='s', func='f', bb=None, min_path_length=1,
                    max_path_length=3, max_path_width=2, ignore_names=[], ignore_op_types=[])
        args.update(changes)
        return generate_candidates_query(**args)

    def test_width_and_sharing(self):
        for width in (1, 2, 4):
            for shared_input, shared_output in ((True, False), (False, True), (True, True)):
                q = self.query(max_path_width=width, shared_input=shared_input, shared_output=shared_output)
                self.assertEqual(q.count('MATCH p'), width)
                self.assertIn('RETURN ' + ', '.join(f'p{i}' for i in range(width)), q)
        with self.assertRaises(ValueError):
            self.query(shared_input=False, shared_output=False)
        self.assertEqual(self.query(simple_mode=True).count('MATCH p'), 1)

    def test_filters_during_expansion(self):
        q = self.query(ignore_names=['CUSTOM'], ignore_op_types=['input'], maxmiso_idx=7,
                       allowed_reg_classes=RegisterClass.GPR | RegisterClass.UNKNOWN)
        self.assertIn("NOT node.name IN ['CUSTOM']", q)
        self.assertIn("NOT node.op_type IN ['input']", q)
        self.assertIn('node.maxmiso_idx = 7', q)
        self.assertIn('node.out_reg_class IS NULL', q)
        self.assertIn('single(visited IN nodes(partial) WHERE visited = node)', q)

    def test_bounds_and_limit(self):
        q = self.query(max_nodes=3, min_nodes=2, limit=7, max_path_width=3)
        self.assertIn('*1..2 ', q)
        self.assertEqual(q.count('<= 3'), 6)
        self.assertIn('>= 2', q)
        self.assertLess(q.index('LIMIT 7'), q.index('ORDER BY'))
        self.assertNotIn('rand()', q)
        self.assertNotIn('LIMIT', self.query())
        self.assertNotIn('None', self.query())
        self.assertNotIn('p1', self.query(max_path_width=1, max_nodes=3))
        for args in ({'limit': 0}, {'max_nodes': 0}, {'min_path_length': 2, 'max_nodes': 2}):
            self.assertIn('LIMIT 0', self.query(**args))

    def test_positive_predicate_caps_use_memgraph_compatible_reduce(self):
        q = self.query(max_loads=1, max_stores=2, max_mems=3, max_branches=1)
        self.assertIn('reduce(total = 0, node IN all_nodes', q)
        self.assertIn('CASE WHEN coalesce(node.mayLoad, false) THEN 1 ELSE 0 END', q)
        self.assertNotIn('size([node IN', q)
        self.assertNotIn('reduce(', self.query(max_loads=1, query_level_filtering=False))

    def test_invalid_bounds(self):
        for args in ({'max_path_width': 0}, {'max_path_width': True}, {'min_path_length': -1},
                     {'min_path_length': 4}, {'limit': -1}, {'min_nodes': 4, 'max_nodes': 2}):
            with self.assertRaises(ValueError):
                self.query(**args)

    def test_bb_and_escaping(self):
        self.assertNotIn('.bb_id', self.query())
        self.assertIn('node.bb_id = 7', self.query(bb='%bb.7'))
        self.assertIn("node.basic_block = 'entry'", self.query(bb='entry', use_bb_id=False))
        with self.assertRaises(ValueError):
            self.query(bb='entry')
        for q in (self.query(func="f'\\\n"), generate_func_query('s', "f'\\\n")):
            self.assertIn("f\\'\\\\\\n", q)


class QueryTimeoutTest(unittest.TestCase):
    def test_timeout_is_passed_to_driver(self):
        from unittest.mock import MagicMock
        from tool.memgraph import run_query
        driver = MagicMock()
        run_query(driver, 'RETURN 1', timeout=2.5)
        query = driver.session.return_value.run.call_args.args[0]
        self.assertEqual(query.text, 'RETURN 1')
        self.assertEqual(query.timeout, 2.5)
        for timeout in (0, -1, float('inf'), float('nan'), True):
            with self.assertRaises(ValueError):
                run_query(driver, 'RETURN 1', timeout=timeout)


if __name__ == '__main__':
    unittest.main()
