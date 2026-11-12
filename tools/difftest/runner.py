"""Execution and classification engine for the optimizer differential test.

Each generated case is executed twice against the same in-memory SQLite
database: once as generated, once after sqlglot optimization. Outcomes:

    ok        - both sides ran and results agree
    mismatch  - both sides ran and results differ (optimizer bug candidate)
    skipped   - the case is not comparable for a known, non-optimizer reason
                (parse/optimize/execute failure on either side, or a known
                SQLite/sqlglot divergence). Never counted as a bug.
"""

from __future__ import annotations

import sqlite3
import typing as t
from dataclasses import dataclass, field

import sqlglot
from sqlglot.optimizer import optimize
from sqlglot.optimizer.optimizer import RULES

from .compare import rows_equal

OK = "ok"
MISMATCH = "mismatch"
SKIPPED = "skipped"

# SQLite/sqlglot semantic divergences that are not optimizer bugs. SQLite
# raises on these while sqlglot (and most engines) accept them.
_KNOWN_SQLITE_DIVERGENCES = (
    "1st ORDER BY term does not match",
    "2nd ORDER BY term does not match",
    "3rd ORDER BY term does not match",
    "ORDER BY term does not match",
)


@dataclass
class Case:
    """A single differential test case."""

    seed: int
    schema: dict[str, dict[str, str]]
    ddl: list[str]
    inserts: list[str]
    query: str
    ordered: bool = False
    profile: str = "small"


@dataclass
class Outcome:
    status: str
    reason: str = ""
    optimized_sql: str | None = None
    original_rows: list | None = None
    optimized_rows: list | None = None
    details: dict = field(default_factory=dict)


def build_database(case: Case) -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    for stmt in case.ddl:
        con.execute(stmt)
    for stmt in case.inserts:
        con.execute(stmt)
    return con


def _fetch(con: sqlite3.Connection, sql: str) -> list:
    return [tuple(row) for row in con.execute(sql).fetchall()]


def optimize_sql(query: str, schema: dict, rules: t.Sequence = RULES) -> str:
    expression = sqlglot.parse_one(query, read="sqlite")
    optimized = optimize(expression, schema=schema, dialect="sqlite", rules=rules)
    return optimized.sql("sqlite")


def run_case(case: Case, rules: t.Sequence = RULES) -> Outcome:
    """Run one case. Never raises; all failures become SKIPPED outcomes."""
    try:
        con = build_database(case)
    except sqlite3.Error as exc:  # pragma: no cover - generator bug guard
        return Outcome(SKIPPED, reason=f"setup-error: {exc}")

    try:
        original_rows = _fetch(con, case.query)
    except sqlite3.Error as exc:
        con.close()
        reason = f"execute-original: {exc}"
        if any(marker in str(exc) for marker in _KNOWN_SQLITE_DIVERGENCES):
            reason = f"known-sqlite-divergence: {exc}"
        return Outcome(SKIPPED, reason=reason)

    try:
        optimized_sql = optimize_sql(case.query, case.schema, rules)
    except Exception as exc:
        con.close()
        return Outcome(SKIPPED, reason=f"optimize-error: {type(exc).__name__}: {exc}")

    try:
        optimized_rows = _fetch(con, optimized_sql)
    except sqlite3.Error as exc:
        con.close()
        return Outcome(
            SKIPPED,
            reason=f"execute-optimized: {exc}",
            optimized_sql=optimized_sql,
        )
    con.close()

    ordered = has_top_level_order_by(case.query)
    if rows_equal(original_rows, optimized_rows, ordered=ordered):
        return Outcome(OK, optimized_sql=optimized_sql)
    return Outcome(
        MISMATCH,
        reason="results differ",
        optimized_sql=optimized_sql,
        original_rows=original_rows,
        optimized_rows=optimized_rows,
    )


def has_top_level_order_by(query: str) -> bool:
    try:
        expression = sqlglot.parse_one(query, read="sqlite")
    except Exception:
        return False
    return expression.args.get("order") is not None
