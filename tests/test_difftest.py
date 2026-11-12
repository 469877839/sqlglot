import sys
import tempfile
import unittest


sys.path.insert(0, "tools")

from difftest.compare import normalize_row, rows_equal
from difftest.gen import generate_case
from difftest.report import case_from_payload, read_counterexample, write_counterexample
from difftest.rules import RULE_SETS
from difftest.runner import MISMATCH, OK, SKIPPED, Case, has_top_level_order_by, run_case
from difftest.shrink import shrink_case


class TestComparison(unittest.TestCase):
    def test_multiset_ignores_order_but_not_duplicates(self):
        self.assertTrue(rows_equal([(1,), (2,), (1,)], [(2,), (1,), (1,)], ordered=False))
        self.assertFalse(rows_equal([(1,), (1,)], [(1,)], ordered=False))

    def test_ordered_is_sequence_sensitive(self):
        self.assertTrue(rows_equal([(1,), (2,)], [(1,), (2,)], ordered=True))
        self.assertFalse(rows_equal([(1,), (2,)], [(2,), (1,)], ordered=True))

    def test_numeric_equivalence(self):
        self.assertEqual(normalize_row([1.0, None, "x"]), normalize_row([1, None, "x"]))

    def test_ordered_flag_follows_top_level_order_by(self):
        self.assertTrue(has_top_level_order_by("SELECT a FROM t1 ORDER BY 1"))
        self.assertFalse(has_top_level_order_by("SELECT a FROM t1"))


class TestClassification(unittest.TestCase):
    def test_mismatch_detected(self):
        case = Case(
            seed=0,
            schema={"t1": {"a": "INT"}},
            ddl=["CREATE TABLE t1 (a INT)"],
            inserts=["INSERT INTO t1 VALUES (1), (1)"],
            query="SELECT DISTINCT a FROM t1",
            profile="small",
        )
        outcome = run_case(case, RULE_SETS["bug-drop-distinct"])
        self.assertEqual(outcome.status, MISMATCH)
        self.assertEqual(outcome.original_rows, [(1,)])
        self.assertEqual(outcome.optimized_rows, [(1,)][:0] + [(1,), (1,)])

    def test_execution_error_is_skipped_not_mismatch(self):
        case = Case(
            seed=0,
            schema={"t1": {"a": "INT"}},
            ddl=["CREATE TABLE t1 (a INT)"],
            inserts=["INSERT INTO t1 VALUES (1)"],
            query="SELECT missing_column FROM t1",
            profile="small",
        )
        self.assertEqual(run_case(case).status, SKIPPED)

    def test_syntax_error_is_skipped_not_mismatch(self):
        case = Case(
            seed=0,
            schema={"t1": {"a": "INT"}},
            ddl=["CREATE TABLE t1 (a INT)"],
            inserts=["INSERT INTO t1 VALUES (1)"],
            query="SELECT a FROM t1 WHERE",
            profile="small",
        )
        self.assertEqual(run_case(case).status, SKIPPED)


class TestShrinker(unittest.TestCase):
    def test_shrinks_query_and_data(self):
        case = generate_case(26, "small")
        reduced, spent = shrink_case(
            case,
            is_failure=lambda c: run_case(c, RULE_SETS["bug-drop-distinct"]).status == MISMATCH,
            budget=100,
        )
        self.assertLess(len(reduced.query), len(case.query))
        self.assertLess(sum(map(len, reduced.inserts)), sum(map(len, case.inserts)))
        self.assertEqual(run_case(reduced, RULE_SETS["bug-drop-distinct"]).status, MISMATCH)

    def test_shrinker_respects_budget(self):
        case = generate_case(26, "small")
        reduced, spent = shrink_case(
            case,
            is_failure=lambda c: run_case(c, RULE_SETS["bug-drop-distinct"]).status == MISMATCH,
            budget=3,
        )
        self.assertLessEqual(spent, 3)


class TestReplay(unittest.TestCase):
    def test_counterexample_roundtrip(self):
        case = generate_case(26, "small")
        outcome = run_case(case, RULE_SETS["bug-drop-distinct"])
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
            path = handle.name
        write_counterexample(path, case, outcome, RULE_SETS["bug-drop-distinct"], 9, 100)
        payload = read_counterexample(path)
        self.assertEqual(payload["rules"][-1], "drop_distinct_BUG")
        replayed = run_case(case_from_payload(payload), RULE_SETS["bug-drop-distinct"])
        self.assertEqual(replayed.status, MISMATCH)


class TestSuite(unittest.TestCase):
    def test_default_rules_have_no_false_positives(self):
        for profile in ("small", "nullable"):
            for seed in range(20):
                self.assertEqual(run_case(generate_case(seed, profile)).status, OK)
