"""Helpers for naming relations and pretty-printing join paths."""

from __future__ import annotations

import sqlglot
from sqlglot import exp


def relation_label(node: exp.Expr, fallback: str = "?") -> str:
    """Best-effort display name for a FROM/JOIN source node."""
    if isinstance(node, exp.Table):
        return node.alias_or_name or node.name or fallback
    if isinstance(node, exp.Subquery):
        return node.alias_or_name or fallback
    if isinstance(node, exp.Select):
        return "__subquery__"
    return fallback


def column_label(col: exp.Column) -> str:
    if col.table:
        return f"{col.table}.{col.name}"
    return col.name


def render_equalities(condition: exp.Expr | None) -> list[str]:
    """Extract top-level AND-ed column equalities, rendered readably."""
    if condition is None:
        return []
    out: list[str] = []
    for eq in _conjuncts(condition):
        if isinstance(eq, exp.EQ) and isinstance(eq.left, exp.Column) and isinstance(eq.right, exp.Column):
            out.append(f"{column_label(eq.left)} = {column_label(eq.right)}")
    return out


def _conjuncts(node: exp.Expr) -> list[exp.Expr]:
    if isinstance(node, exp.And):
        return _conjuncts(node.left) + _conjuncts(node.right)
    if isinstance(node, exp.Paren):
        return _conjuncts(node.this)
    return [node]


def join_on_columns(condition: exp.Expr | None) -> dict[str, set[str]]:
    """Map relation alias -> qualified column names referenced by join equalities.

    Only top-level AND-ed ``col = col`` equalities are considered; anything
    more clever (OR, functions, inequalities) contributes no evidence.
    """
    used: dict[str, set[str]] = {}
    if condition is None:
        return used
    for eq in _conjuncts(condition):
        if not isinstance(eq, exp.EQ):
            continue
        for side in (eq.left, eq.right):
            if isinstance(side, exp.Column) and side.table:
                used.setdefault(side.table.lower(), set()).add(side.name.lower())
    return used


def source_text(node: exp.Expr) -> str:
    try:
        return node.sql(dialect="")
    except Exception:  # pragma: no cover - defensive only
        return repr(node)
