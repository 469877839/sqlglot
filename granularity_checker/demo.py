"""Runnable demo: paired cases plus an in-memory SQLite fan-out example.

Run:  python -m granularity_checker.demo
"""

from __future__ import annotations

import sqlite3

from .checker import analyze_text
from .schema_model import DeclaredSchema

BASE_SCHEMA = DeclaredSchema.from_dict(
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
        "customers": {
            "columns": ["customer_id", "name"],
            "unique_keys": [["customer_id"]],
        },
        "click_log": {"columns": ["order_id", "ts"], "unique_keys": []},
    }
)


def _banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _show(sql: str, schema: DeclaredSchema, note: str) -> None:
    report = analyze_text(sql, schema)
    print(f"-- {note}")
    print(f"   SQL: {sql}")
    for v in report.verdicts:
        print(f"   -> [{v.verdict}] join {v.left_relation} -> {v.right_relation}")
        print(f"      {v.reason}")
    for d in report.degradations:
        print(f"   -> [DEGRADED] {d}")
    if not report.verdicts and not report.degradations:
        print("   -> no joins found")


def paired_cases() -> None:
    _banner("PAIR A: one join-condition column decides the verdict")
    _show(
        "SELECT sum(o.amount) FROM orders o JOIN order_items i ON i.item_id = o.order_id",
        BASE_SCHEMA,
        "A1: join hits the child unique key item_id (contrived equality, but provably non-expanding)",
    )
    _show(
        "SELECT sum(o.amount) FROM orders o JOIN order_items i ON i.order_id = o.order_id",
        BASE_SCHEMA,
        "A2: same query, join on the child's non-unique FK column -> fan-out risk",
    )

    _banner("PAIR B: pre-aggregation inside a derived table")
    _show(
        "SELECT sum(o.amount) FROM orders o JOIN order_items i ON i.order_id = o.order_id",
        BASE_SCHEMA,
        "B1: raw child table joined before aggregation",
    )
    _show(
        "SELECT sum(o.amount) FROM orders o JOIN "
        "(SELECT order_id, sum(qty) AS q FROM order_items GROUP BY order_id) i "
        "ON i.order_id = o.order_id",
        BASE_SCHEMA,
        "B2: child pre-aggregated by the join key -> GROUP BY proves uniqueness",
    )

    _banner("PAIR C: the unique-key declaration itself")
    _show(
        "SELECT sum(i.qty) FROM order_items i JOIN orders o ON o.order_id = i.order_id",
        BASE_SCHEMA,
        "C1: orders.order_id declared unique -> join provably non-expanding",
    )
    no_keys = DeclaredSchema.from_dict(
        {
            "orders": {"columns": ["order_id", "customer_id", "amount"], "unique_keys": []},
            "order_items": {"columns": ["item_id", "order_id", "qty"], "unique_keys": []},
        }
    )
    _show(
        "SELECT sum(i.qty) FROM order_items i JOIN orders o ON o.order_id = i.order_id",
        no_keys,
        "C2: identical SQL, no unique keys declared -> evidence insufficient, UNKNOWN",
    )

    _banner("PAIR D: UNION vs UNION ALL inside a derived table")
    _show(
        "SELECT * FROM orders o JOIN "
        "(SELECT order_id FROM click_log UNION SELECT order_id FROM click_log) u "
        "ON u.order_id = o.order_id",
        BASE_SCHEMA,
        "D1: UNION deduplicates -> full projection is provably unique",
    )
    _show(
        "SELECT * FROM orders o JOIN "
        "(SELECT order_id FROM click_log UNION ALL SELECT order_id FROM click_log) u "
        "ON u.order_id = o.order_id",
        BASE_SCHEMA,
        "D2: UNION ALL keeps duplicates -> grain unknown",
    )

    _banner("PAIR E: DISTINCT in a derived table")
    _show(
        "SELECT * FROM orders o JOIN (SELECT order_id, ts FROM click_log) d ON d.order_id = o.order_id",
        BASE_SCHEMA,
        "E1: plain projection of a keyless log -> UNKNOWN",
    )
    _show(
        "SELECT * FROM orders o JOIN (SELECT DISTINCT order_id FROM click_log) d ON d.order_id = o.order_id",
        BASE_SCHEMA,
        "E2: DISTINCT makes the projection unique -> SAFE",
    )

    _banner("EXPLICIT DEGRADATION: correlated subquery and recursive CTE")
    _show(
        "SELECT * FROM orders o WHERE EXISTS "
        "(SELECT 1 FROM order_items i WHERE i.order_id = o.order_id)",
        BASE_SCHEMA,
        "correlated EXISTS: checker degrades explicitly instead of guessing",
    )
    _show(
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 10) "
        "SELECT * FROM orders o JOIN r ON r.n = o.order_id",
        BASE_SCHEMA,
        "recursive CTE: grain unknown, join through it is UNKNOWN",
    )


def sqlite_fanout_demo() -> None:
    _banner("IN-MEMORY SQLITE: the RISK case really double-counts")
    db = sqlite3.connect(":memory:")
    db.executescript(
        """
        CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL);
        CREATE TABLE order_items (item_id INTEGER PRIMARY KEY, order_id INTEGER, qty INTEGER);
        INSERT INTO orders VALUES (1, 10, 100.0), (2, 20, 200.0);
        INSERT INTO order_items VALUES (1, 1, 1), (2, 1, 2), (3, 2, 1);
        """
    )
    risky = "SELECT sum(o.amount) FROM orders o JOIN order_items i ON i.order_id = o.order_id"
    safe = (
        "SELECT sum(o.amount) FROM orders o JOIN "
        "(SELECT order_id, sum(qty) AS q FROM order_items GROUP BY order_id) i "
        "ON i.order_id = o.order_id"
    )
    truth = "SELECT sum(amount) FROM orders"

    for label, sql in [("ground truth", truth), ("risky join", risky), ("pre-aggregated join", safe)]:
        (value,) = db.execute(sql).fetchone()
        report = analyze_text(sql, BASE_SCHEMA)
        verdict = report.worst_verdict or "NO JOINS"
        print(f"   {label:22s} sum(amount) = {value:7.1f}   checker verdict: {verdict}")
    print()
    print("   300.0 is correct; 400.0 shows order 1 counted once per item row.")


def main() -> None:
    paired_cases()
    sqlite_fanout_demo()
    print()
    print("Note: a SAFE verdict only proves the join cannot multiply rows on the")
    print("declared keys. It never proves a business metric is computed correctly.")


if __name__ == "__main__":
    main()
