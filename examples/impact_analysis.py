"""
Offline impact analysis walkthrough.

Run with:

    python examples/impact_analysis.py

The first half uses complete, small schemas: every difference comes from real
``SELECT *`` / ``JOIN USING`` / set-operation expansion, so each change has a
definite category. The second half removes pieces of metadata to show how the
analysis degrades to INDETERMINATE instead of guessing that unknown columns are
compatible.
"""

from sqlglot.impact import ImpactCategory, analyze_impact


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def show(sql: str, before: dict, after: dict, dialect: str = "spark") -> None:
    report = analyze_impact(sql, before, after, dialect=dialect)
    print(f"\nSQL: {sql}")
    print(f"breaking categories: {[category.value for category in report.breaking_categories()]}")
    print(report)


section("1. Differences produced by real expansion (full metadata)")

# The upstream table drops a, widens c's type and adds d. With an explicit
# projection the removed column is a definite contract change.
orders_before = {"orders": {"id": "INT", "amount": "DECIMAL(10, 2)", "channel": "TEXT"}}
orders_after = {
    "orders": {"id": "INT", "amount": "DECIMAL(12, 2)", "channel": "TEXT", "status": "TEXT"}
}
show("SELECT id, amount, channel FROM orders", orders_before, orders_after)

# SELECT *: a real expansion surfaces added / removed / moved / type changes
# against actual resolved columns rather than the star token.
show("SELECT * FROM orders", orders_before, orders_after)

# JOIN USING normalizes the shared key into a single COALESCE output; dropping
# that key on one side is a hard failure for that snapshot.
users_before = {"users": {"id": "INT", "name": "VARCHAR(32)"}}
joins_before = {**users_before, **orders_before}
joins_after = {**users_before, **orders_after}
show("SELECT * FROM users JOIN orders USING (id)", joins_before, joins_after)

# CTE and set operations resolve through to the base table types.
show(
    "WITH top_orders AS (SELECT id, amount FROM orders) SELECT amount FROM top_orders",
    orders_before,
    orders_after,
)
show(
    "SELECT id, channel FROM orders UNION ALL SELECT id, channel FROM users",
    {
        "orders": {"id": "INT", "channel": "TEXT"},
        "users": {"id": "INT", "channel": "VARCHAR(8)"},
    },
    {
        "orders": {"id": "BIGINT", "channel": "TEXT"},
        "users": {"id": "INT", "channel": "VARCHAR(8)"},
    },
)

section("2. Differences limited by missing metadata (never silently compatible)")

# A table absent from both mappings: nothing can be proven, so the output is
# INDETERMINATE instead of "unchanged".
show("SELECT mystery_col FROM mystery_table", orders_before, orders_after)

# One snapshot expands the star fully while the other cannot resolve the table
# at all: even the columns visible on the resolved side are reported as
# INDETERMINATE, because the unexpanded star may or may not still contain them.
show("SELECT * FROM orders", orders_before, {})

# A qualified star over a partially-known join: t.* still expands, but a star
# whose source table is missing keeps the whole projection uncertain.
show(
    "SELECT orders.* FROM orders JOIN events e ON orders.id = e.order_id",
    {"orders": orders_before["orders"], "events": {"order_id": "INT"}},
    {"orders": orders_after["orders"]},
)

# A known column whose type the schema does not carry stays UNKNOWN on both
# sides; identical unknowns are still flagged, never treated as compatible.
show(
    "SELECT id, untyped_note FROM orders",
    {"orders": {"id": "INT"}},
    {"orders": {"id": "BIGINT"}},
)

section("3. Machine-readable output and source localization")

report = analyze_impact("SELECT * FROM orders", orders_before, orders_after, dialect="spark")
for change in report.changes:
    affected = change.after or change.before
    print(
        f"{change.category.value:14} "
        f"pos={affected.position} name={affected.name!r:10} "
        f"type={affected.type_sql or 'UNKNOWN':14} sources={affected.sources}"
    )
print("\nCategories used in this report:", {c.category.value for c in report.changes})
print("Categories available:", [category.value for category in ImpactCategory])
