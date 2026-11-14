import copy
import unittest

from sqlglot import MappingSchema, parse_one
from sqlglot.schema_change import ChangeKind, analyze_schema_change


class TestSchemaChange(unittest.TestCase):
    def analyze(self, sql, before, after, dialect=None):
        return analyze_schema_change(sql, before, after, dialect=dialect)

    def test_no_changes(self):
        report = self.analyze(
            "SELECT a, b FROM t",
            {"t": {"a": "INT", "b": "TEXT"}},
            {"t": {"a": "INT", "b": "TEXT"}},
        )
        self.assertEqual(report.changes, [])
        self.assertFalse(report.has_changes)
        self.assertEqual(report.summary(), "No contract changes detected.")

    def test_column_added_by_star_expansion(self):
        report = self.analyze(
            "SELECT * FROM t",
            {"t": {"a": "INT"}},
            {"t": {"a": "INT", "b": "TEXT"}},
        )
        self.assertEqual(len(report.changes), 1)
        change = report.changes[0]
        self.assertEqual(change.kind, ChangeKind.ADDED)
        self.assertEqual(change.name_after, "b")
        self.assertEqual(change.position_after, 1)
        self.assertEqual(change.type_after, "TEXT")
        self.assertFalse(change.metadata_limited)
        self.assertIn("b", change.expression_after)

    def test_column_removed_by_star_expansion(self):
        report = self.analyze(
            "SELECT * FROM t",
            {"t": {"a": "INT", "b": "TEXT"}},
            {"t": {"a": "INT"}},
        )
        self.assertEqual(len(report.changes), 1)
        change = report.changes[0]
        self.assertEqual(change.kind, ChangeKind.REMOVED)
        self.assertEqual(change.name_before, "b")
        self.assertEqual(change.position_before, 1)
        self.assertFalse(change.metadata_limited)

    def test_type_changed(self):
        report = self.analyze(
            "SELECT a FROM t", {"t": {"a": "INT"}}, {"t": {"a": "BIGINT"}}
        )
        self.assertEqual(len(report.changes), 1)
        change = report.changes[0]
        self.assertEqual(change.kind, ChangeKind.TYPE_CHANGED)
        self.assertEqual(change.type_before, "INT")
        self.assertEqual(change.type_after, "BIGINT")
        self.assertEqual(change.name_before, "a")

    def test_position_changed(self):
        report = self.analyze(
            "SELECT * FROM t",
            {"t": {"a": "INT", "b": "TEXT"}},
            {"t": {"b": "TEXT", "a": "INT"}},
        )
        self.assertEqual(len(report.changes), 2)
        self.assertTrue(all(c.kind is ChangeKind.MOVED for c in report.changes))
        moved = {c.name_after: (c.position_before, c.position_after) for c in report.changes}
        self.assertEqual(moved, {"a": (0, 1), "b": (1, 0)})

    def test_renamed(self):
        report = self.analyze("SELECT * FROM t", {"t": {"a": "INT"}}, {"t": {"b": "INT"}})
        self.assertEqual(len(report.changes), 1)
        change = report.changes[0]
        self.assertEqual(change.kind, ChangeKind.RENAMED)
        self.assertEqual(change.name_before, "a")
        self.assertEqual(change.name_after, "b")
        self.assertEqual(change.position_before, change.position_after)

    def test_unknown_type_is_not_compatible(self):
        # The source table is missing from both schemas, so nothing can be verified
        report = self.analyze("SELECT * FROM t", {}, {})
        self.assertTrue(report.has_changes)
        self.assertTrue(all(c.kind is ChangeKind.INDETERMINATE for c in report.changes))
        self.assertTrue(all(c.metadata_limited for c in report.changes))

    def test_unknown_column_type_is_not_compatible(self):
        # m is missing from both schemas, so the type of m.x cannot be determined
        report = self.analyze("SELECT m.x FROM m", {}, {})
        self.assertEqual(len(report.changes), 1)
        self.assertEqual(report.changes[0].kind, ChangeKind.INDETERMINATE)

    def test_metadata_limited_expansion(self):
        # The star cannot be expanded under the before schema, so the columns that
        # appear under the after schema cannot be confirmed as real additions
        report = self.analyze(
            "SELECT * FROM t", {}, {"t": {"a": "INT", "b": "TEXT"}}
        )
        self.assertTrue(report.has_changes)
        self.assertTrue(all(c.kind is ChangeKind.INDETERMINATE for c in report.changes))
        self.assertTrue(all(c.metadata_limited for c in report.changes))
        names = {c.name_after for c in report.changes}
        self.assertIn("a", names)
        self.assertIn("b", names)

    def test_join_using(self):
        before = {"a": {"id": "INT", "x": "TEXT"}, "b": {"id": "INT", "y": "DATE"}}
        after = {"a": {"id": "INT", "x": "TEXT"}, "b": {"id": "BIGINT", "y": "DATE"}}
        report = self.analyze("SELECT * FROM a JOIN b USING (id)", before, after)
        self.assertEqual(len(report.changes), 1)
        change = report.changes[0]
        self.assertEqual(change.kind, ChangeKind.TYPE_CHANGED)
        self.assertEqual(change.name_after, "id")
        self.assertEqual(change.type_before, "INT")
        self.assertEqual(change.type_after, "BIGINT")

    def test_cte(self):
        report = self.analyze(
            "WITH c AS (SELECT a FROM t) SELECT * FROM c",
            {"t": {"a": "INT"}},
            {"t": {"a": "DOUBLE"}},
        )
        self.assertEqual(len(report.changes), 1)
        change = report.changes[0]
        self.assertEqual(change.kind, ChangeKind.TYPE_CHANGED)
        self.assertEqual(change.type_before, "INT")
        self.assertEqual(change.type_after, "DOUBLE")

    def test_set_operation(self):
        before = {"t1": {"a": "INT"}, "t2": {"a": "INT"}}
        after = {"t1": {"a": "INT"}, "t2": {"a": "DOUBLE"}}
        report = self.analyze("SELECT a FROM t1 UNION ALL SELECT a FROM t2", before, after)
        self.assertEqual(len(report.changes), 1)
        change = report.changes[0]
        self.assertEqual(change.kind, ChangeKind.TYPE_CHANGED)
        self.assertEqual(change.name_after, "a")
        self.assertEqual(change.type_before, "INT")
        self.assertEqual(change.type_after, "DOUBLE")

    def test_duplicate_column_names(self):
        before = {"a": {"id": "INT"}, "b": {"id": "INT"}}
        after = {"a": {"id": "INT"}, "b": {"id": "BIGINT"}}
        report = self.analyze("SELECT * FROM a JOIN b ON a.id = b.id", before, after)
        self.assertEqual(len(report.changes), 1)
        change = report.changes[0]
        self.assertEqual(change.kind, ChangeKind.TYPE_CHANGED)
        self.assertEqual(change.name_after, "id")
        self.assertEqual(change.position_before, 1)
        self.assertEqual(change.position_after, 1)

    def test_aliased_expression(self):
        report = self.analyze(
            "SELECT amount * 2 AS total FROM t",
            {"t": {"amount": "INT"}},
            {"t": {"amount": "DOUBLE"}},
        )
        self.assertEqual(len(report.changes), 1)
        change = report.changes[0]
        self.assertEqual(change.kind, ChangeKind.TYPE_CHANGED)
        self.assertEqual(change.name_after, "total")
        self.assertIn("total", change.expression_after)

    def test_inputs_are_not_mutated(self):
        sql = "SELECT * FROM a JOIN b USING (id)"
        expression = parse_one(sql)
        expression_sql = expression.sql()

        before = {"a": {"id": "INT", "x": "TEXT"}, "b": {"id": "INT", "y": "DATE"}}
        after = {"a": {"id": "INT"}, "b": {"id": "BIGINT", "y": "DATE", "z": "INT"}}
        before_snapshot = copy.deepcopy(before)
        after_snapshot = copy.deepcopy(after)

        mapping_before = MappingSchema(before)
        mapping_after = MappingSchema(after)
        mapping_before_snapshot = copy.deepcopy(mapping_before.mapping)
        mapping_after_snapshot = copy.deepcopy(mapping_after.mapping)

        report = analyze_schema_change(expression, mapping_before, mapping_after)
        self.assertTrue(report.has_changes)

        self.assertEqual(expression.sql(), expression_sql)
        self.assertEqual(before, before_snapshot)
        self.assertEqual(after, after_snapshot)
        self.assertEqual(mapping_before.mapping, mapping_before_snapshot)
        self.assertEqual(mapping_after.mapping, mapping_after_snapshot)

    def test_report_locates_output_expression(self):
        report = self.analyze(
            "SELECT a, b + 1 AS c FROM t",
            {"t": {"a": "INT", "b": "INT"}},
            {"t": {"a": "INT", "b": "DOUBLE"}},
        )
        self.assertEqual(len(report.changes), 1)
        change = report.changes[0]
        self.assertEqual(change.name_after, "c")
        self.assertIn("b", change.expression_before)
        self.assertIn("c", change.expression_after)

    def test_summary(self):
        report = self.analyze(
            "SELECT * FROM t",
            {"t": {"a": "INT", "b": "TEXT"}},
            {"t": {"a": "BIGINT", "c": "TEXT"}},
        )
        self.assertEqual(report.summary(), "2 change(s): 1 renamed, 1 type_changed")

    def test_non_query_rejected(self):
        with self.assertRaises(ValueError):
            analyze_schema_change("CREATE TABLE t (a INT)", {}, {})

    def test_dialect(self):
        report = analyze_schema_change(
            "SELECT a FROM t",
            {"t": {"a": "INT"}},
            {"t": {"a": "BIGINT"}},
            dialect="snowflake",
        )
        self.assertEqual(report.dialect, "Snowflake")
        self.assertEqual(report.changes[0].kind, ChangeKind.TYPE_CHANGED)
