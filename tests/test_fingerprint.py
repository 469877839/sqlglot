import unittest

import sqlglot
from sqlglot.fingerprint import fingerprint, same_family


class TestFingerprint(unittest.TestCase):
    def assert_merged(self, left, right, dialect=None):
        """Both statements belong to the same query family."""
        left_fp = fingerprint(left, dialect=dialect)
        right_fp = fingerprint(right, dialect=dialect)

        self.assertEqual(left_fp.digest, right_fp.digest)
        self.assertEqual(left_fp.normalized, right_fp.normalized)
        return left_fp.normalized

    def assert_split(self, left, right, dialect=None):
        """The statements must stay in different families."""
        left_fp = fingerprint(left, dialect=dialect)
        right_fp = fingerprint(right, dialect=dialect)

        self.assertNotEqual(left_fp.digest, right_fp.digest)
        self.assertNotEqual(left_fp.normalized, right_fp.normalized)

    def test_constants_are_typed_placeholders(self):
        normalized = self.assert_merged(
            "SELECT * FROM t WHERE id = 1 AND name = 'a' AND flag = TRUE",
            "SELECT * FROM t WHERE id = 42 AND name = 'b' AND flag = FALSE",
        )

        self.assertEqual(
            normalized,
            'SELECT * FROM "t" AS "_0" WHERE "id" = :num AND "name" = :str AND "flag" = :bool',
        )

    def test_normalized_form_is_reparseable(self):
        normalized = fingerprint("SELECT * FROM t WHERE x = 1 AND y = 'a'").normalized
        self.assertEqual(
            sqlglot.parse_one(normalized).sql(identify=True),
            normalized,
            "normalized form should be valid SQL that readers can inspect",
        )

    def test_whitespace_comments_and_case_are_ignored(self):
        self.assert_merged(
            "SELECT a, B FROM T WHERE a = 1",
            "select  a , b\nfrom t  -- trailing comment\nwhere A = 2",
            dialect="duckdb",
        )

    def test_local_aliases_merge(self):
        normalized = self.assert_merged(
            "SELECT a.x, b.y FROM t1 a JOIN t2 b ON a.id = b.id",
            "SELECT left_.x, right_.y FROM t1 left_ JOIN t2 right_ ON left_.id = right_.id",
        )

        self.assertEqual(
            normalized,
            'SELECT "_0"."x", "_1"."y" FROM "t1" AS "_0" JOIN "t2" AS "_1" '
            'ON "_0"."id" = "_1"."id"',
        )

    def test_self_join_aliases_merge(self):
        normalized = self.assert_merged(
            "SELECT a.x, b.y FROM t a JOIN t b ON a.id = b.pid",
            "SELECT l.x, r.y FROM t l JOIN t r ON l.id = r.pid",
        )

        self.assertEqual(
            normalized,
            'SELECT "_0"."x", "_1"."y" FROM "t" AS "_0" JOIN "t" AS "_1" '
            'ON "_0"."id" = "_1"."pid"',
        )

    def test_correlated_subquery_aliases_merge(self):
        normalized = self.assert_merged(
            "SELECT a.x FROM t1 a WHERE a.id IN (SELECT c.x FROM t2 c WHERE c.z = a.id)",
            "SELECT o.x FROM t1 o WHERE o.id IN (SELECT i.x FROM t2 i WHERE i.z = o.id)",
        )

        self.assertEqual(
            normalized,
            'SELECT "_0"."x" FROM "t1" AS "_0" WHERE "_0"."id" IN '
            '(SELECT "_1"."x" FROM "t2" AS "_1" WHERE "_1"."z" = "_0"."id")',
        )

    def test_cte_names_and_references_merge(self):
        normalized = self.assert_merged(
            "WITH recent AS (SELECT id FROM src WHERE ts > 10) "
            "SELECT id FROM recent WHERE id IN (SELECT id FROM recent r WHERE r.id > 0)",
            "WITH r2 AS (SELECT id FROM src WHERE ts > 99) "
            "SELECT id FROM r2 WHERE id IN (SELECT id FROM r2 x WHERE x.id > 0)",
        )

        self.assertEqual(
            normalized,
            'WITH "_cte0" AS (SELECT "id" FROM "src" AS "_1" WHERE "ts" > :num) '
            'SELECT "id" FROM "_cte0" AS "_0" WHERE "id" IN '
            '(SELECT "id" FROM "_cte0" AS "_2" WHERE "_2"."id" > :num)',
        )

    def test_nested_cte_shadowing_merges(self):
        self.assert_merged(
            "WITH t AS (SELECT n FROM u) "
            "SELECT * FROM (WITH t AS (SELECT n FROM v) SELECT n FROM t) d",
            "WITH outer_ AS (SELECT n FROM u) "
            "SELECT * FROM (WITH inner_ AS (SELECT n FROM v) SELECT n FROM inner_) x",
        )

    def test_parameters_merge_regardless_of_style(self):
        for left, right, dialect in (
            ("SELECT * FROM t WHERE x = ?", "SELECT * FROM t WHERE x = $1", "postgres"),
            ("SELECT * FROM t WHERE x = :p", "SELECT * FROM t WHERE x = ?", None),
            ("SELECT * FROM t WHERE x = @p", "SELECT * FROM t WHERE x = ?", "tsql"),
        ):
            normalized = self.assert_merged(left, right, dialect=dialect)
            self.assertIn("param", normalized)

    def test_set_operation_branches_merge(self):
        normalized = self.assert_merged(
            "SELECT a FROM t1 x UNION ALL SELECT b FROM t2 y",
            "SELECT a FROM t1 p UNION ALL SELECT b FROM t2 q",
        )

        self.assertEqual(
            normalized,
            'SELECT "a" FROM "t1" AS "_0" UNION ALL SELECT "b" FROM "t2" AS "_1"',
        )

    def test_derived_tables_merge(self):
        normalized = self.assert_merged(
            "SELECT d.a FROM (SELECT a FROM t WHERE b = 1) d JOIN (SELECT c FROM u) e ON d.a = e.c",
            "SELECT sub.a FROM (SELECT a FROM t WHERE b = 2) sub "
            "JOIN (SELECT c FROM u) other ON sub.a = other.c",
        )

        self.assertEqual(
            normalized,
            'SELECT "_0"."a" FROM (SELECT "a" FROM "t" AS "_2" WHERE "b" = :num) AS "_0" '
            'JOIN (SELECT "c" FROM "u" AS "_3") AS "_1" ON "_0"."a" = "_1"."c"',
        )

    def test_dml_constants_merge(self):
        self.assert_merged(
            "INSERT INTO t (a, b) VALUES (1, 'x')",
            "INSERT INTO t (a, b) VALUES (99, 'y')",
        )
        self.assert_merged(
            "UPDATE t SET x = 1 WHERE id = 2",
            "UPDATE t SET x = 8 WHERE id = 9",
        )
        self.assert_merged(
            "DELETE FROM t WHERE id IN (SELECT id FROM u WHERE v > 1)",
            "DELETE FROM t WHERE id IN (SELECT id FROM u WHERE v > 5)",
        )

    def test_dml_relation_aliases_merge(self):
        self.assert_merged(
            "UPDATE t SET x = 1 FROM u WHERE u.id = t.id",
            "UPDATE t SET x = 1 FROM u AS v WHERE v.id = t.id",
        )
        self.assert_merged(
            "DELETE FROM t USING u AS a WHERE a.id = t.id",
            "DELETE FROM t USING u AS b WHERE b.id = t.id",
        )
        self.assert_merged(
            "MERGE INTO t USING u AS a ON t.id = a.id "
            "WHEN MATCHED THEN UPDATE SET x = a.x",
            "MERGE INTO t USING u AS b ON t.id = b.id "
            "WHEN MATCHED THEN UPDATE SET x = b.x",
        )

    def test_recursive_cte_aliases_merge(self):
        normalized = self.assert_merged(
            "WITH RECURSIVE r AS ("
            "SELECT 1 AS n UNION ALL SELECT n + 1 FROM r WHERE n < 10"
            ") SELECT * FROM r",
            "WITH RECURSIVE nums AS ("
            "SELECT 1 AS n UNION ALL SELECT n + 1 FROM nums WHERE n < 99"
            ") SELECT * FROM nums",
        )
        self.assertNotIn("r", normalized)
        self.assertNotIn("nums", normalized)

    def test_duplicate_cte_name_in_different_scopes(self):
        # Both CTEs happen to be called t; their distinct positions must get
        # distinct canonical names even though the spelling is identical.
        normalized = fingerprint(
            "WITH t AS (SELECT n FROM u) "
            "SELECT * FROM (WITH t AS (SELECT n FROM v) SELECT n FROM t) d"
        ).normalized
        self.assertIn('"_cte0"', normalized)
        self.assertIn('"_cte1"', normalized)

    def test_quoted_vs_unquoted_identifiers(self):
        # In case-insensitive dialects the quotes do not change the binding.
        self.assert_merged('SELECT "a" FROM t', "SELECT a FROM t", dialect="duckdb")
        self.assert_split('SELECT "a" FROM t', "SELECT a FROM t", dialect="snowflake")

    def test_quoted_identifier_case_is_preserved(self):
        self.assert_split(
            'SELECT "Col" FROM "T"',
            'SELECT "col" FROM "T"',
            dialect="snowflake",
        )

    def test_table_identity_is_preserved(self):
        self.assert_split(
            "SELECT x FROM t1 WHERE x = 1",
            "SELECT x FROM t2 WHERE x = 1",
        )
        self.assert_split(
            "SELECT x FROM cat.db.tbl WHERE x = 1",
            "SELECT x FROM other.db.tbl WHERE x = 1",
        )

    def test_qualified_table_columns_merge(self):
        # The db qualifier is identity (kept on the table), but its column
        # qualifiers follow the canonical source alias regardless of spelling.
        self.assert_merged(
            "SELECT real.t.n FROM real.t WHERE real.t.n > 1",
            "SELECT z.n FROM real.t AS z WHERE z.n > 2",
        )

    def test_dml_target_aliases_merge(self):
        self.assert_merged(
            "UPDATE t AS x SET a = 1 WHERE x.id = 2",
            "UPDATE t AS y SET a = 1 WHERE y.id = 2",
        )
        self.assert_merged(
            "DELETE FROM t AS z WHERE z.id = 1",
            "DELETE FROM t AS w WHERE w.id = 1",
        )
        self.assert_merged(
            "INSERT INTO t AS z (a) VALUES (1)",
            "INSERT INTO t AS w (a) VALUES (2)",
        )

    def test_column_identity_is_preserved(self):
        self.assert_split(
            "SELECT a FROM t WHERE a = 1",
            "SELECT b FROM t WHERE b = 1",
        )

    def test_projection_alias_is_preserved(self):
        self.assert_split(
            "SELECT a AS foo, b AS bar FROM t",
            "SELECT a AS foo, b AS baz FROM t",
        )

    def test_projection_order_is_preserved(self):
        self.assert_split("SELECT x, y FROM t", "SELECT y, x FROM t")

    def test_comparison_operand_order_is_preserved(self):
        self.assert_split("SELECT * FROM t WHERE a = 1", "SELECT * FROM t WHERE 1 = a")

    def test_join_order_is_preserved(self):
        self.assert_split(
            "SELECT * FROM t1 a JOIN t2 b ON a.id = b.id",
            "SELECT * FROM t1 a JOIN t2 b ON b.id = a.id",
        )
        self.assert_split(
            "SELECT * FROM t1 a JOIN t2 b ON a.id = b.id",
            "SELECT * FROM t2 b JOIN t1 a ON b.id = a.id",
        )

    def test_join_type_is_preserved(self):
        self.assert_split(
            "SELECT * FROM t1 a JOIN t2 b ON a.id = b.id",
            "SELECT * FROM t1 a LEFT JOIN t2 b ON a.id = b.id",
        )

    def test_set_operation_kind_is_preserved(self):
        self.assert_split(
            "SELECT a FROM t1 UNION SELECT a FROM t2",
            "SELECT a FROM t1 UNION ALL SELECT a FROM t2",
        )

    def test_literal_types_are_distinct(self):
        self.assert_split("SELECT * FROM t WHERE x = 1", "SELECT * FROM t WHERE x = '1'")
        self.assert_split("SELECT * FROM t WHERE x = 1", "SELECT * FROM t WHERE x = ?")

    def test_negation_is_preserved(self):
        self.assert_split(
            "SELECT * FROM t WHERE x IN (SELECT y FROM u)",
            "SELECT * FROM t WHERE x NOT IN (SELECT y FROM u)",
        )

    def test_binding_equivalent_alias_shadowing_a_cte_merges(self):
        # `FROM t` and `FROM t AS t` both reference the outer CTE, so they bind
        # identically and belong together.
        self.assert_merged(
            "WITH t AS (SELECT n FROM u) SELECT n FROM t",
            "WITH t AS (SELECT n FROM u) SELECT n FROM t AS t",
        )

    def test_cte_reference_vs_real_table_diverges(self):
        # Text-identical `FROM t`, but one binds to the CTE and one to a real
        # table named `t`; they must not merge.
        self.assert_split(
            "WITH t AS (SELECT n FROM u) SELECT * FROM s WHERE EXISTS "
            "(SELECT 1 FROM t WHERE t.n = s.n)",
            "WITH t AS (SELECT n FROM u) SELECT * FROM s WHERE EXISTS "
            "(SELECT 1 FROM real.t WHERE real.t.n = s.n)",
        )

    def test_shadowed_cte_vs_outer_cte_diverges(self):
        # The inner `t` is a different CTE (defined in the subquery's WITH),
        # even though the outer one has the same name.
        self.assert_split(
            "WITH t AS (SELECT n FROM u) SELECT (SELECT n FROM t) FROM s",
            "WITH t AS (SELECT n FROM u) "
            "SELECT (SELECT n FROM (WITH t AS (SELECT n FROM v) SELECT n FROM t) i) FROM s",
        )

    def test_same_table_different_self_join_role_diverges(self):
        # Same two textually identical sources, but the projection reads
        # through a different one of them.
        self.assert_split(
            "SELECT a.x FROM t a JOIN t b ON a.id = b.pid",
            "SELECT b.x FROM t a JOIN t b ON a.id = b.pid",
        )

    def test_correlation_target_diverges(self):
        self.assert_split(
            "SELECT * FROM t1 a WHERE a.id IN (SELECT b.id FROM t2 b WHERE b.z = a.id)",
            "SELECT * FROM t1 a WHERE a.id IN (SELECT b.id FROM t2 b WHERE b.z = b.id)",
        )

    def test_different_cte_body_diverges(self):
        self.assert_split(
            "WITH t AS (SELECT n FROM u) SELECT n FROM t",
            "WITH t AS (SELECT n FROM v) SELECT n FROM t",
        )

    def test_different_source_shape_diverges(self):
        # `FROM t` (a table) vs `FROM (SELECT ...) AS t` (a derived table).
        self.assert_split(
            "WITH t AS (SELECT n FROM u) SELECT n FROM t",
            "WITH t AS (SELECT n FROM u) SELECT * FROM (SELECT n FROM t) AS t",
        )

    def test_fingerprint_is_stable_and_case_sensitive_per_dialect(self):
        first = fingerprint("SELECT Col FROM T", dialect="snowflake")
        second = fingerprint("SELECT Col FROM T", dialect="snowflake")
        self.assertEqual(first, second)
        self.assertEqual(first.dialect, "Snowflake")
        self.assertEqual(len(first.digest), 64)

    def test_same_family_helper(self):
        self.assertTrue(same_family("SELECT a FROM t WHERE x = 1", "SELECT a FROM t WHERE x = 2"))
        self.assertFalse(same_family("SELECT a FROM t", "SELECT b FROM t"))

    def test_invalid_sql_raises(self):
        with self.assertRaises(sqlglot.errors.ParseError):
            fingerprint("SELECT FROM WHERE")

    def test_copy_target_is_not_aliased(self):
        normalized = fingerprint(
            "COPY INTO mytable (col1, col2) FROM 's3://bucket/files'"
        ).normalized
        self.assertEqual(normalized, 'COPY INTO "mytable" ("col1", "col2") FROM :str')


if __name__ == "__main__":
    unittest.main()
