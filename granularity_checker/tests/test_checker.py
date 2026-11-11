"""Verdict tests for the granularity checker.

Run:  python -m pytest granularity_checker/tests -q
"""

from __future__ import annotations

import sqlite3

from granularity_checker import DeclaredSchema, analyze_text

SCHEMA = DeclaredSchema.from_dict(
    {
        "orders": {
            "columns": ["order_id", "customer_id", "amount"],
            "unique_keys": [["order_id"]],
            "foreign_keys": [
                {"columns": ["customer_id"], "ref_table": "customers", "ref_columns": ["customer_id"]}
            ],
        },
        "order_items": {
            "columns": ["item_id", "order_id", "qty"],
            "unique_keys": [["item_id"]],
            "foreign_keys": [
                {"columns": ["order_id"], "ref_table": "orders", "ref_columns": ["order_id"]}
            ],
        },
        "customers": {"columns": ["customer_id"], "unique_keys": [["customer_id"]]},
        "click_log": {"columns": ["order_id", "ts"], "unique_keys": []},
    }
)


def verdicts(sql, schema=SCHEMA):
    report = analyze_text(sql, schema)
    assert report.parse_error is None
    return report


def only(sql, schema=SCHEMA):
    report = verdicts(sql, schema)
    assert len(report.verdicts) == 1
    return report.verdicts[0]


# ---------------------------------------------------------------- pair A
def test_join_on_child_unique_key_is_safe():
    v = only("SELECT sum(o.amount) FROM orders o JOIN order_items i ON i.item_id = o.order_id")
    assert v.verdict == "SAFE"


def test_join_on_child_fk_is_risk():
    v = only("SELECT sum(o.amount) FROM orders o JOIN order_items i ON i.order_id = o.order_id")
    assert v.verdict == "RISK"
    assert "order_id" in v.reason


# ---------------------------------------------------------------- pair B
def test_preaggregated_child_is_safe():
    v = only(
        "SELECT sum(o.amount) FROM orders o JOIN "
        "(SELECT order_id, sum(qty) q FROM order_items GROUP BY order_id) i "
        "ON i.order_id = o.order_id"
    )
    assert v.verdict == "SAFE"


# ---------------------------------------------------------------- pair C
def test_missing_unique_key_declaration_flips_to_unknown():
    bare = DeclaredSchema.from_dict(
        {
            "orders": {"columns": ["order_id", "amount"], "unique_keys": []},
            "order_items": {"columns": ["item_id", "order_id"], "unique_keys": []},
        }
    )
    sql = "SELECT sum(i.qty) FROM order_items i JOIN orders o ON o.order_id = i.order_id"
    assert only(sql, SCHEMA).verdict == "SAFE"
    assert only(sql, bare).verdict == "UNKNOWN"


# ---------------------------------------------------------------- pair D
def test_union_dedup_is_safe_but_union_all_unknown():
    safe_sql = (
        "SELECT * FROM orders o JOIN "
        "(SELECT order_id FROM click_log UNION SELECT order_id FROM click_log) u "
        "ON u.order_id = o.order_id"
    )
    all_sql = safe_sql.replace("UNION SELECT", "UNION ALL SELECT")
    assert only(safe_sql).verdict == "SAFE"
    assert only(all_sql).verdict == "UNKNOWN"


# ---------------------------------------------------------------- pair E
def test_distinct_derived_is_safe():
    plain = "SELECT * FROM orders o JOIN (SELECT order_id, ts FROM click_log) d ON d.order_id = o.order_id"
    distinct = "SELECT * FROM orders o JOIN (SELECT DISTINCT order_id FROM click_log) d ON d.order_id = o.order_id"
    assert only(plain).verdict == "UNKNOWN"
    assert only(distinct).verdict == "SAFE"


# ------------------------------------------------------------- other rules
def test_join_to_parent_unique_key_is_safe():
    v = only("SELECT sum(i.qty) FROM order_items i JOIN orders o ON o.order_id = i.order_id")
    assert v.verdict == "SAFE"


def test_cte_group_by_grain_flows_to_outer_join():
    v = only(
        "WITH g AS (SELECT order_id, sum(qty) q FROM order_items GROUP BY order_id) "
        "SELECT * FROM orders o JOIN g ON g.order_id = o.order_id"
    )
    assert v.verdict == "SAFE"


def test_cross_join_is_risk():
    v = only("SELECT * FROM orders o CROSS JOIN customers c")
    assert v.verdict == "RISK"


def test_unknown_table_is_unknown_not_risk():
    v = only("SELECT * FROM orders o JOIN mystery m ON m.order_id = o.order_id")
    assert v.verdict == "UNKNOWN"


def test_chain_reports_each_join():
    report = verdicts(
        "SELECT * FROM orders o "
        "JOIN order_items i ON i.order_id = o.order_id "
        "JOIN customers c ON c.customer_id = o.customer_id"
    )
    assert [v.verdict for v in report.verdicts] == ["RISK", "SAFE"]
    assert report.worst_verdict == "RISK"


def test_correlated_subquery_degrades():
    report = verdicts(
        "SELECT * FROM orders o WHERE EXISTS "
        "(SELECT 1 FROM order_items i WHERE i.order_id = o.order_id)"
    )
    assert any("correlated" in d for d in report.degradations)


def test_uncorrelated_subquery_does_not_degrade():
    report = verdicts(
        "SELECT * FROM orders o WHERE o.order_id IN (SELECT order_id FROM order_items)"
    )
    assert not report.degradations


def test_recursive_cte_degrades():
    report = verdicts(
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 10) "
        "SELECT * FROM orders o JOIN r ON r.n = o.order_id"
    )
    assert any("recursive" in d for d in report.degradations)
    assert report.verdicts[0].verdict == "UNKNOWN"


def test_safe_is_not_a_business_correctness_claim():
    report = verdicts("SELECT sum(i.qty) FROM order_items i JOIN orders o ON o.order_id = i.order_id")
    assert "not" in report.format_text().lower()


# ------------------------------------------------------------- sqlite proof
def test_sqlite_actually_double_counts():
    db = sqlite3.connect(":memory:")
    db.executescript(
        """
        CREATE TABLE orders (order_id INTEGER PRIMARY KEY, amount REAL);
        CREATE TABLE order_items (item_id INTEGER PRIMARY KEY, order_id INTEGER, qty INTEGER);
        INSERT INTO orders VALUES (1, 100.0), (2, 200.0);
        INSERT INTO order_items VALUES (1, 1, 1), (2, 1, 2), (3, 2, 1);
        """
    )
    truth = db.execute("SELECT sum(amount) FROM orders").fetchone()[0]
    risky_sql = "SELECT sum(o.amount) FROM orders o JOIN order_items i ON i.order_id = o.order_id"
    risky = db.execute(risky_sql).fetchone()[0]
    assert truth == 300.0
    assert risky == 400.0  # order 1 counted twice
    assert only(risky_sql).verdict == "RISK"

    fixed_sql = (
        "SELECT sum(o.amount) FROM orders o JOIN "
        "(SELECT order_id, sum(qty) q FROM order_items GROUP BY order_id) i "
        "ON i.order_id = o.order_id"
    )
    assert db.execute(fixed_sql).fetchone()[0] == 300.0
    assert only(fixed_sql).verdict == "SAFE"
