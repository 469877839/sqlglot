import copy
import unittest

from sqlglot import parse_one
from sqlglot.errors import ImpactAnalysisError
from sqlglot.impact import (
    ImpactCategory,
    UncertaintyReason,
    analyze_impact,
)
from sqlglot.schema import MappingSchema


def _categories(report):
    return [change.category for change in report.changes]


class TestImpactAnalysis(unittest.TestCase):
    def _analyze(self, sql, before, after, **kwargs):
        return analyze_impact(sql, before, after, dialect="spark", **kwargs)

    def test_unchanged_contract(self):
        schema = {"t": {"a": "INT", "b": "TEXT"}}
        report = self._analyze("SELECT a, b FROM t", schema, schema)

        self.assertEqual(report.changes, [])
        self.assertFalse(report.has_breaking_changes)
        self.assertEqual([column.name for column in report.before_contract], ["a", "b"])
        self.assertEqual([column.type_sql for column in report.before_contract], ["INT", "TEXT"])

    def test_type_change(self):
        report = self._analyze(
            "SELECT a FROM t",
            {"t": {"a": "INT"}},
            {"t": {"a": "BIGINT"}},
        )

        self.assertEqual(_categories(report), [ImpactCategory.TYPE_CHANGED])
        change = report.changes[0]
        self.assertEqual(change.before.type_sql, "INT")
        self.assertEqual(change.after.type_sql, "BIGINT")
        self.assertEqual(change.before.sources, (("t", "a"),))
        self.assertTrue(report.has_breaking_changes)

    def test_star_expansion_drives_added_removed_and_moved(self):
        report = self._analyze(
            "SELECT * FROM t",
            {"t": {"a": "INT", "b": "TEXT", "c": "DATE"}},
            {"t": {"b": "TEXT", "c": "TIMESTAMP", "d": "BOOLEAN"}},
        )

        self.assertEqual(
            _categories(report),
            [
                ImpactCategory.REMOVED,
                ImpactCategory.MOVED,
                ImpactCategory.MOVED,
                ImpactCategory.TYPE_CHANGED,
                ImpactCategory.ADDED,
            ],
        )
        self.assertEqual(report.changes[0].before.name, "a")
        self.assertEqual(report.changes[-1].after.name, "d")
        self.assertEqual(report.changes[-1].after.position, 2)
        moved = report.changes[1]
        self.assertEqual((moved.before.name, moved.before.position), ("b", 1))
        self.assertEqual((moved.after.name, moved.after.position), ("b", 0))

    def test_moved_detected_behind_star(self):
        # dict ordering cannot reorder an explicit projection; MOVED is driven by
        # real star expansion shifting surviving columns around
        report = self._analyze(
            "SELECT *, 1 AS one FROM t",
            {"t": {"a": "INT", "b": "TEXT"}},
            {"t": {"b": "TEXT", "a": "INT"}},
        )

        moved = report.changes_of(ImpactCategory.MOVED)
        self.assertEqual(
            [
                (change.after.name, change.before.position, change.after.position)
                for change in moved
            ],
            [("b", 1, 0), ("a", 0, 1)],
        )

    def test_duplicate_column_names_are_matched_independently(self):
        report = self._analyze(
            "SELECT a, a, b FROM t",
            {"t": {"a": "INT", "b": "TEXT"}},
            {"t": {"a": "BIGINT", "b": "TEXT"}},
        )

        self.assertEqual(
            _categories(report),
            [ImpactCategory.TYPE_CHANGED, ImpactCategory.TYPE_CHANGED],
        )
        self.assertEqual([change.after.position for change in report.changes], [0, 1])

    def test_alias_expression_matches_across_type_change(self):
        report = self._analyze(
            "SELECT a + 1 AS k FROM t",
            {"t": {"a": "INT"}},
            {"t": {"a": "DOUBLE"}},
        )

        self.assertEqual(_categories(report), [ImpactCategory.TYPE_CHANGED])
        self.assertEqual(report.changes[0].after.name, "k")
        self.assertEqual(report.changes[0].before.sources, (("t", "a"),))

    def test_join_using_expands_shared_key(self):
        before = {"u": {"id": "INT", "name": "TEXT"}, "o": {"id": "INT", "amt": "DECIMAL(8, 2)"}}
        after = {
            "u": {"id": "DOUBLE", "name": "TEXT"},
            "o": {"id": "DOUBLE", "amt": "DECIMAL(8, 2)"},
        }

        report = self._analyze("SELECT * FROM u JOIN o USING (id)", before, after)

        self.assertEqual(_categories(report), [ImpactCategory.TYPE_CHANGED])
        self.assertEqual(report.changes[0].after.name, "id")
        # COALESCE promotes the shared INT key to BIGINT under Spark
        self.assertEqual(report.changes[0].before.type_sql, "BIGINT")
        self.assertEqual(report.changes[0].after.type_sql, "DOUBLE")

    def test_join_using_key_removed_is_indeterminate(self):
        before = {"u": {"id": "INT", "name": "TEXT"}, "o": {"id": "INT", "amt": "DECIMAL(8, 2)"}}
        after = {"u": {"id": "INT", "name": "TEXT"}, "o": {"amt": "DECIMAL(8, 2)"}}

        report = self._analyze("SELECT * FROM u JOIN o USING (id)", before, after)

        self.assertTrue(
            all(category is ImpactCategory.INDETERMINATE for category in _categories(report))
        )
        self.assertTrue(report.has_indeterminate_changes)
        self.assertEqual(
            report.after_contract[0].reason, UncertaintyReason.QUALIFICATION_FAILED
        )
        self.assertTrue(
            all(
                "star" in change.reason or "unresolved" in change.reason
                for change in report.changes
            )
        )

    def test_nested_star_resolves_through_derived_table(self):
        report = self._analyze(
            "SELECT * FROM (SELECT * FROM t) x",
            {"t": {"a": "INT", "b": "TEXT"}},
            {"t": {"a": "BIGINT", "b": "TEXT", "c": "DATE"}},
        )

        self.assertEqual(_categories(report), [ImpactCategory.TYPE_CHANGED, ImpactCategory.ADDED])
        self.assertEqual(report.changes[0].after.name, "a")
        self.assertEqual(report.changes[1].after.name, "c")

    def test_star_exclude_removes_column_from_contract(self):
        report = analyze_impact(
            "SELECT * EXCLUDE (b) FROM t",
            {"t": {"a": "INT", "b": "TEXT", "c": "DATE"}},
            {"t": {"a": "BIGINT", "c": "DATE", "d": "BOOLEAN"}},
            dialect="duckdb",
        )

        self.assertEqual(
            _categories(report),
            [ImpactCategory.TYPE_CHANGED, ImpactCategory.ADDED],
        )
        self.assertEqual([change.after.name for change in report.changes], ["a", "d"])

    def test_upstream_rename_surfaces_as_remove_and_add(self):
        # there is no stable identity to pair a renamed upstream column with, so
        # the honest report is a removal plus an addition
        report = self._analyze(
            "SELECT * FROM t",
            {"t": {"a": "INT"}},
            {"t": {"b": "INT"}},
        )

        self.assertEqual(_categories(report), [ImpactCategory.REMOVED, ImpactCategory.ADDED])
        self.assertEqual(report.changes[0].before.name, "a")
        self.assertEqual(report.changes[1].after.name, "b")

    def test_cte_output_resolved_through_cte(self):
        before = {"t": {"a": "INT", "b": "TEXT"}}
        after = {"t": {"a": "VARCHAR(8)", "b": "TEXT"}}

        report = self._analyze("WITH x AS (SELECT a, b FROM t) SELECT a FROM x", before, after)

        self.assertEqual(_categories(report), [ImpactCategory.TYPE_CHANGED])
        self.assertEqual(report.changes[0].before.type_sql, "INT")
        self.assertEqual(report.changes[0].after.type_sql, "VARCHAR(8)")

    def test_set_operation_uses_leftmost_select_contract(self):
        before = {"t": {"a": "INT", "b": "TEXT"}, "s": {"a": "BIGINT", "b": "VARCHAR(10)"}}
        after = {"t": {"a": "BIGINT", "b": "TEXT"}, "s": {"a": "BIGINT", "b": "VARCHAR(10)"}}

        report = self._analyze("SELECT a, b FROM t UNION ALL SELECT a, b FROM s", before, after)

        self.assertEqual(_categories(report), [ImpactCategory.TYPE_CHANGED])
        self.assertEqual(report.changes[0].after.name, "a")

    def test_missing_table_is_indeterminate_not_compatible(self):
        report = self._analyze("SELECT x FROM ghost", {"t": {"a": "INT"}}, {"t": {"a": "INT"}})

        self.assertEqual(_categories(report), [ImpactCategory.INDETERMINATE])
        self.assertTrue(report.has_breaking_changes)
        self.assertEqual(report.changes[0].after.reason, UncertaintyReason.UNKNOWN_TYPE)

    def test_unknown_type_on_one_side_is_indeterminate(self):
        report = self._analyze(
            "SELECT a, x FROM g",
            {"t": {"a": "INT"}},
            {"t": {"a": "INT"}, "g": {"x": "VARCHAR(5)"}},
        )

        self.assertEqual(
            _categories(report),
            [ImpactCategory.INDETERMINATE, ImpactCategory.INDETERMINATE],
        )
        for change in report.changes:
            self.assertIsNotNone(change.reason)

    def test_unexpanded_star_is_indeterminate(self):
        report = self._analyze(
            "SELECT t.* FROM t",
            {"t": {"a": "INT"}},
            {},
        )

        self.assertEqual(_categories(report), [ImpactCategory.INDETERMINATE])
        self.assertEqual(report.changes[0].after.reason, UncertaintyReason.UNEXPANDED_STAR)
        self.assertEqual(report.changes[0].after.sources, (("t", "*"),))

        unqualified = self._analyze("SELECT * FROM t", {"t": {"a": "INT"}}, {})
        self.assertEqual(unqualified.changes[0].after.reason, UncertaintyReason.UNEXPANDED_STAR)
        self.assertEqual(unqualified.changes[0].after.sources, ())

    def test_unknown_type_stays_indeterminate_even_if_identical(self):
        # the same unresolved column on both sides must not be reported as
        # unchanged, because the change cannot be excluded
        report = self._analyze("SELECT x FROM t", {"t": {"a": "INT"}}, {"t": {"a": "INT"}})

        self.assertEqual(_categories(report), [ImpactCategory.INDETERMINATE])
        self.assertFalse(
            any(change.category is ImpactCategory.ADDED for change in report.changes)
        )
        self.assertFalse(
            any(change.category is ImpactCategory.REMOVED for change in report.changes)
        )

    def test_changes_locate_the_originating_expression(self):
        report = self._analyze(
            "SELECT a + 1 AS k, b FROM t",
            {"t": {"a": "INT", "b": "TEXT"}},
            {"t": {"a": "DOUBLE", "b": "TEXT"}},
        )

        change = report.changes[0]
        self.assertIn("a + 1", change.after.expression.sql())

    def test_inputs_are_not_mutated(self):
        ast = parse_one("SELECT a, b FROM t")
        ast_sql = ast.sql()
        before = {"t": {"a": "INT", "b": "TEXT"}}
        after = {"t": {"a": "BIGINT", "b": "TEXT"}}
        before_copy = copy.deepcopy(before)
        after_copy = copy.deepcopy(after)
        schema = MappingSchema(before)

        report = analyze_impact(ast, schema, after, dialect="spark")

        self.assertTrue(report.changes)
        self.assertEqual(ast.sql(), ast_sql)
        self.assertEqual(before, before_copy)
        self.assertEqual(after, after_copy)
        self.assertEqual(schema.mapping, before_copy)

    def test_dropped_projected_column_is_indeterminate(self):
        # a missing non-USING column does not abort qualification: surviving
        # columns stay comparable, the missing one is UNKNOWN_TYPE
        report = self._analyze(
            "SELECT a, b FROM t",
            {"t": {"a": "INT", "b": "TEXT"}},
            {"t": {"a": "INT"}},
        )

        self.assertEqual(_categories(report), [ImpactCategory.INDETERMINATE])
        self.assertEqual(report.after_contract[1].reason, UncertaintyReason.UNKNOWN_TYPE)
        self.assertEqual(report.after_contract[0].reason, None)

    def test_non_query_input_is_rejected(self):
        with self.assertRaises(ImpactAnalysisError):
            analyze_impact("CREATE TABLE x (a INT)", {}, {})

    def test_to_dict_serialization(self):
        report = self._analyze(
            "SELECT a FROM t",
            {"t": {"a": "INT"}},
            {"t": {"a": "BIGINT"}},
        )

        payload = report.to_dict()
        self.assertEqual(payload["changes"][0]["category"], "TYPE_CHANGED")
        self.assertEqual(payload["changes"][0]["before"]["type"], "INT")
        self.assertEqual(payload["changes"][0]["after"]["type"], "BIGINT")

    def test_report_str(self):
        report = self._analyze(
            "SELECT a FROM t",
            {"t": {"a": "INT"}},
            {"t": {"a": "BIGINT"}},
        )

        rendered = str(report)
        self.assertIn("TYPE_CHANGED", rendered)
        self.assertIn("INT -> BIGINT", rendered)

    def test_dialect_affects_type_parsing(self):
        report = analyze_impact(
            "SELECT a FROM t",
            {"t": {"a": "NUMBER"}},
            {"t": {"a": "VARCHAR"}},
            dialect="snowflake",
        )

        self.assertEqual(_categories(report), [ImpactCategory.TYPE_CHANGED])
        self.assertEqual(report.changes[0].before.type_sql, "DECIMAL(38, 0)")
