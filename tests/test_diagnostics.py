import unittest

from sqlglot import (
    Category,
    ConversionDiagnostic,
    ConversionPolicy,
    LossKind,
    LossyConversionError,
    exp,
    parse_one,
    transpile,
    transpile_with_diagnostics,
)
from sqlglot.dialects.dialect import Dialect
from sqlglot.diagnostics import SOURCE_ID_META_KEY
from sqlglot.helper import logger as helper_logger

# Columns of the case table: (read, sql, generated_sql, expected_diagnostic) where
# expected_diagnostic is (category, loss, node_sql, node_type) or None if the rewrite
# is exact.
CaseRow = tuple[str, str, str, tuple | None]

# Real generation paths of two existing dialects: BigQuery and Spark. Together they
# cover all four diagnostic categories, with exact rewrites interleaved so the table
# also proves those are never mislabeled as lossy.
BIGQUERY_CASES: list[CaseRow] = [
    # BigQuery natively orders NULLS LAST for ascending keys: exact rewrite.
    (
        "postgres",
        "SELECT * FROM t ORDER BY a NULLS LAST",
        "SELECT * FROM t ORDER BY a NULLS LAST",
        None,
    ),
    # RANK() supports NULLS FIRST inside its window: exact rewrite.
    (
        "postgres",
        "SELECT RANK() OVER (ORDER BY y NULLS FIRST) FROM t",
        "SELECT RANK() OVER (ORDER BY y) FROM t",
        None,
    ),
    # An aggregate window cannot express DESC NULLS FIRST in BigQuery: clause dropped.
    (
        "postgres",
        "SELECT SUM(x) OVER (ORDER BY y DESC NULLS FIRST) FROM t",
        "SELECT SUM(x) OVER (ORDER BY y DESC) FROM t",
        (
            Category.NULL_ORDERING,
            LossKind.NULL_ORDERING,
            "y DESC NULLS FIRST",
            "Ordered",
        ),
    ),
    # BigQuery has no INTERSECT ALL: the ALL clause is emitted verbatim.
    (
        "postgres",
        "SELECT 1 INTERSECT ALL SELECT 1",
        "SELECT 1 INTERSECT ALL SELECT 1",
        (
            Category.SET_OPERATION,
            LossKind.SET_OPERATION,
            "SELECT 1 INTERSECT ALL SELECT 1",
            "Intersect",
        ),
    ),
    # A plain INTERSECT is exact in BigQuery: no diagnostic.
    (
        "postgres",
        "SELECT 1 INTERSECT SELECT 1",
        "SELECT 1 INTERSECT DISTINCT SELECT 1",
        None,
    ),
    # BigQuery TIMESTAMP stores an instant: the tz-aware mapping is exact.
    (
        "postgres",
        "SELECT CAST(x AS TIMESTAMPTZ)",
        "SELECT CAST(x AS TIMESTAMP)",
        None,
    ),
]

SPARK_CASES: list[CaseRow] = [
    # Spark TIMESTAMP is TIMESTAMP_LTZ: the tz-aware mapping is exact.
    (
        "postgres",
        "SELECT CAST(x AS TIMESTAMPTZ)",
        "SELECT CAST(x AS TIMESTAMP)",
        None,
    ),
    # Spark keeps NULLS LAST natively: exact rewrite.
    (
        "postgres",
        "SELECT * FROM t ORDER BY a NULLS LAST",
        "SELECT * FROM t ORDER BY a NULLS LAST",
        None,
    ),
    # Spark accepts INTERSECT ALL: exact rewrite.
    (
        "postgres",
        "SELECT 1 INTERSECT ALL SELECT 1",
        "SELECT 1 INTERSECT ALL SELECT 1",
        None,
    ),
    # Spark temporal types carry no precision parameter: (9) is dropped.
    (
        "postgres",
        "SELECT CAST(x AS TIMESTAMP(9))",
        "SELECT CAST(x AS TIMESTAMP)",
        (Category.TEMPORAL, LossKind.PRECISION, "TIMESTAMP(9)", "DataType"),
    ),
    # Even a supported-by-others precision such as (3) cannot be represented.
    (
        "postgres",
        "SELECT CAST(x AS TIMESTAMP(3))",
        "SELECT CAST(x AS TIMESTAMP)",
        (Category.TEMPORAL, LossKind.PRECISION, "TIMESTAMP(3)", "DataType"),
    ),
    # A non-literal array position cannot be shifted from 1-based to 0-based.
    (
        "duckdb",
        "SELECT ARRAY_INSERT(arr, i, 9) FROM t",
        "SELECT ARRAY_INSERT(arr, i, 9) FROM t",
        (
            Category.ARRAY_INDEXING,
            LossKind.INDEXING,
            "ARRAY_INSERT(arr, i, 9)",
            "ArrayInsert",
        ),
    ),
    # A literal array position is shifted exactly (1-based 1 -> 0-based 2).
    (
        "duckdb",
        "SELECT ARRAY_INSERT(arr, 1, 9) FROM t",
        "SELECT ARRAY_INSERT(arr, 2, 9) FROM t",
        None,
    ),
    # Bracket subscripts are normalized at parse time, so generating is exact.
    (
        "duckdb",
        "SELECT arr[1] FROM t",
        "SELECT arr[0] FROM t",
        None,
    ),
]

CASE_TABLES: dict[str, list[CaseRow]] = {
    "bigquery": BIGQUERY_CASES,
    "spark": SPARK_CASES,
}


def target_dialect_title(write: str) -> str:
    return Dialect.get_or_raise(write).__class__.__name__


class TestDiagnostics(unittest.TestCase):
    maxDiff = None

    def generate(self, sql, read, write, diagnostics=ConversionPolicy.COLLECT):
        generator = Dialect.get_or_raise(write).generator(diagnostics=diagnostics)
        expression = parse_one(sql, read=read)
        return generator, expression, generator.generate(expression, copy=False)

    def test_case_table(self):
        """The two-dialect case table is backed by executable generation paths."""
        for write, cases in CASE_TABLES.items():
            title = target_dialect_title(write)
            for read, sql, expected_sql, expected in cases:
                with self.subTest(write=write, sql=sql):
                    generator, _, generated = self.generate(sql, read, write)
                    self.assertEqual(generated, expected_sql)

                    if expected is None:
                        self.assertEqual(generator.diagnostics, [])
                        continue

                    category, loss, node_sql, node_type = expected
                    self.assertEqual(len(generator.diagnostics), 1)
                    diagnostic = generator.diagnostics[0]

                    self.assertIsInstance(diagnostic, ConversionDiagnostic)
                    self.assertEqual(diagnostic.category, category)
                    self.assertEqual(diagnostic.loss, loss)
                    self.assertEqual(diagnostic.node_sql, node_sql)
                    self.assertEqual(type(diagnostic.node).__name__, node_type)
                    self.assertEqual(diagnostic.target_dialect, title)
                    self.assertTrue(diagnostic.action)
                    self.assertTrue(diagnostic.message)
                    self.assertIsNotNone(diagnostic.source_id)

    def test_policies(self):
        sql = "SELECT 1 INTERSECT ALL SELECT 1"

        collected = transpile_with_diagnostics(sql, read="postgres", write="bigquery")
        self.assertEqual(len(collected), 1)
        self.assertEqual(collected[0].sql, sql)
        self.assertEqual(len(collected[0].diagnostics), 1)

        allowed = transpile_with_diagnostics(
            sql, read="postgres", write="bigquery", diagnostics=ConversionPolicy.ALLOW
        )
        self.assertEqual(allowed[0].sql, sql)
        self.assertEqual(allowed[0].diagnostics, [])

        with self.assertRaises(LossyConversionError) as ctx:
            transpile(sql, read="postgres", write="bigquery", diagnostics=ConversionPolicy.REJECT)
        self.assertEqual(ctx.exception.diagnostics[0].category, Category.SET_OPERATION)
        self.assertEqual(len(ctx.exception.diagnostics), 1)

    def test_default_transpile_is_compatible(self):
        sql = "SELECT 1 INTERSECT ALL SELECT 1"
        self.assertEqual(transpile(sql, read="postgres", write="bigquery"), [sql])

        # Without a policy the legacy `unsupported` warnings are still emitted.
        with self.assertLogs(helper_logger, level="WARNING") as logs:
            transpile(sql, read="postgres", write="bigquery")
        self.assertIn("INTERSECT ALL is not supported", logs.output[0])

        # `ALLOW` accepts the loss silently: no warning is surfaced.
        with self.assertNoLogs(helper_logger, level="WARNING"):
            transpile(sql, read="postgres", write="bigquery", diagnostics=ConversionPolicy.ALLOW)

    def test_multiple_statements_do_not_cross_contaminate(self):
        sql = (
            "SELECT 1 INTERSECT ALL SELECT 1;"
            "SELECT CAST(x AS TIMESTAMPTZ);"
            "SELECT CAST(x AS TIMESTAMP(3));"
            "SELECT CAST(x AS TIMESTAMP(9))"
        )

        results = transpile_with_diagnostics(sql, read="postgres", write="spark")
        self.assertEqual(len(results), 4)
        self.assertEqual([len(result.diagnostics) for result in results], [0, 0, 1, 1])
        self.assertEqual(
            [result.diagnostics[0].loss if result.diagnostics else None for result in results],
            [None, None, LossKind.PRECISION, LossKind.PRECISION],
        )
        for result in results:
            for diagnostic in result.diagnostics:
                self.assertEqual(diagnostic.target_dialect, "Spark")

        # The same statement list against BigQuery keeps its own diagnostics separate.
        results = transpile_with_diagnostics(sql, read="postgres", write="bigquery")
        self.assertEqual(len(results), 4)
        self.assertEqual([len(result.diagnostics) for result in results], [1, 0, 0, 0])
        self.assertEqual(results[0].diagnostics[0].category, Category.SET_OPERATION)

    def test_ast_copy_and_repeated_compilation(self):
        sql = "SELECT 1 INTERSECT ALL SELECT 1"
        generator = Dialect.get_or_raise("bigquery").generator(
            diagnostics=ConversionPolicy.COLLECT
        )
        source = parse_one(sql, read="postgres")
        intersect = source.find(exp.Intersect)

        # `copy=True` leaves the caller's tree untouched and still traces to it.
        generator.generate(source, copy=True)
        self.assertEqual(len(generator.diagnostics), 1)
        self.assertIs(generator.diagnostics[0].node, intersect)
        self.assertIsNone(intersect.meta_get(SOURCE_ID_META_KEY))

        # `copy=False` does not tag the caller's tree either.
        generator.generate(source, copy=False)
        self.assertIsNone(intersect.meta_get(SOURCE_ID_META_KEY))
        self.assertIs(generator.diagnostics[0].node, intersect)

        # Repeated compilation replaces diagnostics instead of accumulating them.
        generator.generate(source)
        generator.generate(source)
        self.assertEqual(len(generator.diagnostics), 1)

        # A copy of the AST compiles independently and keeps its own diagnostics.
        clone = source.copy()
        clone_generator = Dialect.get_or_raise("bigquery").generator(
            diagnostics=ConversionPolicy.COLLECT
        )
        clone_generator.generate(clone)
        self.assertEqual(len(clone_generator.diagnostics), 1)
        self.assertEqual(len(generator.diagnostics), 1)

        # A lossless run in another generator leaves that generator clean.
        lossless = Dialect.get_or_raise("spark").generator(diagnostics=ConversionPolicy.COLLECT)
        lossless.generate(parse_one("SELECT CAST(x AS TIMESTAMPTZ)", read="postgres"))
        self.assertEqual(lossless.diagnostics, [])

        # A generator without diagnostics keeps the legacy field empty.
        legacy = Dialect.get_or_raise("bigquery").generator()
        self.assertEqual(legacy.generate(source), sql)
        self.assertEqual(legacy.diagnostics, [])

    def test_provenance_across_multi_step_transforms(self):
        """Two losses on one source node share one provenance id that points at it."""
        generator = Dialect.get_or_raise("hive").generator(
            diagnostics=ConversionPolicy.COLLECT
        )
        source = parse_one("SELECT CAST(x AS TIMESTAMPTZ(9))", read="postgres")
        data_type = source.find(exp.DataType)
        generator.generate(source)

        self.assertEqual(
            {diagnostic.loss for diagnostic in generator.diagnostics},
            {LossKind.PRECISION, LossKind.TIMEZONE},
        )
        self.assertEqual(len({d.source_id for d in generator.diagnostics}), 1)

        for diagnostic in generator.diagnostics:
            self.assertIs(diagnostic.node, data_type)
            self.assertEqual(diagnostic.node_sql, "TIMESTAMPTZ(9)")

        # The source node is nested (non-root) and only reachable through a type mapping.
        self.assertIsNot(data_type, source)
        self.assertIsNone(data_type.meta_get(SOURCE_ID_META_KEY))

    def test_mysql_null_ordering_paths(self):
        # MySQL cannot emit NULLS FIRST/LAST: an exact CASE-based simulation must not be
        # flagged, while clauses that can only be dropped must produce a diagnostic.
        exact = transpile_with_diagnostics(
            "SELECT * FROM t ORDER BY a NULLS LAST", read="postgres", write="mysql"
        )
        self.assertEqual(
            exact[0].sql,
            "SELECT * FROM t ORDER BY CASE WHEN a IS NULL THEN 1 ELSE 0 END, a",
        )
        self.assertEqual(exact[0].diagnostics, [])

        positional = transpile_with_diagnostics(
            "SELECT a, b FROM t ORDER BY 1 DESC NULLS FIRST", read="postgres", write="mysql"
        )
        self.assertEqual(positional[0].sql, "SELECT a, b FROM t ORDER BY 1 DESC")
        self.assertEqual(len(positional[0].diagnostics), 1)
        self.assertEqual(positional[0].diagnostics[0].category, Category.NULL_ORDERING)
        self.assertEqual(positional[0].diagnostics[0].node_sql, "1 DESC NULLS FIRST")

        window = transpile_with_diagnostics(
            "SELECT ROW_NUMBER() OVER (ORDER BY x DESC NULLS FIRST RANGE BETWEEN "
            "UNBOUNDED PRECEDING AND CURRENT ROW) FROM t",
            read="postgres",
            write="mysql",
        )
        self.assertEqual(len(window[0].diagnostics), 1)
        self.assertEqual(window[0].diagnostics[0].loss, LossKind.NULL_ORDERING)
        self.assertIn("ROW_NUMBER", window[0].diagnostics[0].action)

        random_ordering = transpile_with_diagnostics(
            "SELECT a FROM t ORDER BY RAND() DESC NULLS FIRST",
            read="postgres",
            write="mysql",
        )
        self.assertEqual(len(random_ordering[0].diagnostics), 1)
        self.assertEqual(random_ordering[0].diagnostics[0].category, Category.NULL_ORDERING)

    def test_hive_timezone_and_precision_losses(self):
        results = transpile_with_diagnostics(
            "SELECT CAST(x AS TIMESTAMPTZ(9))", read="postgres", write="hive"
        )
        self.assertEqual(results[0].sql, "SELECT CAST(x AS TIMESTAMP)")
        kinds = {diagnostic.loss for diagnostic in results[0].diagnostics}
        self.assertEqual(kinds, {LossKind.PRECISION, LossKind.TIMEZONE})
        self.assertEqual(
            {diagnostic.target_dialect for diagnostic in results[0].diagnostics}, {"Hive"}
        )

        # A tz-agnostic TIMESTAMP(3) only reports the precision loss on Hive.
        results = transpile_with_diagnostics(
            "SELECT CAST(x AS TIMESTAMP(3))", read="postgres", write="hive"
        )
        self.assertEqual([d.loss for d in results[0].diagnostics], [LossKind.PRECISION])

        # Presto keeps tz semantics explicitly: no timezone diagnostic.
        results = transpile_with_diagnostics(
            "SELECT CAST(x AS TIMESTAMPTZ)", read="postgres", write="presto"
        )
        self.assertEqual(results[0].sql, "SELECT CAST(x AS TIMESTAMP WITH TIME ZONE)")
        self.assertEqual(results[0].diagnostics, [])

    def test_delegating_generator_propagates_diagnostics(self):
        # AthenaGenerator delegates statement generation to Hive or Trino sub-generators;
        # the hive-routed statement must still surface diagnostics on the outer generator.
        results = transpile_with_diagnostics(
            "CREATE EXTERNAL TABLE t (a TIMESTAMP(9)) STORED AS PARQUET",
            read="hive",
            write="athena",
        )
        self.assertEqual(len(results), 1)
        self.assertIn("a TIMESTAMP", results[0].sql)
        self.assertEqual(len(results[0].diagnostics), 1)
        self.assertEqual(results[0].diagnostics[0].loss, LossKind.PRECISION)

    def test_diagnostic_serialization(self):
        results = transpile_with_diagnostics(
            "SELECT CAST(x AS TIMESTAMP(9))", read="postgres", write="spark"
        )
        payload = results[0].diagnostics[0].to_dict()
        self.assertEqual(
            payload,
            {
                "category": "TEMPORAL",
                "node_sql": "TIMESTAMP(9)",
                "node_type": "DataType",
                "target_dialect": "Spark",
                "action": "dropped temporal precision of TIMESTAMP",
                "loss": "PRECISION",
                "message": "TIMESTAMP precision is not supported by Spark; dropping it",
                "source_id": "n4",
            },
        )


if __name__ == "__main__":
    unittest.main()
