"""Optimizer rule sets and deliberately faulty injected rules.

The ``*_BUG`` transformations are intentionally wrong and exist only to prove
that the differential harness and the shrinker actually catch optimizer
misbehavior. Never ship them into a real pipeline.
"""

from __future__ import annotations

import typing as t

from sqlglot import exp
from sqlglot.optimizer.annotate_types import annotate_types
from sqlglot.optimizer.canonicalize import canonicalize
from sqlglot.optimizer.eliminate_ctes import eliminate_ctes
from sqlglot.optimizer.eliminate_joins import eliminate_joins
from sqlglot.optimizer.eliminate_subqueries import eliminate_subqueries
from sqlglot.optimizer.merge_subqueries import merge_subqueries
from sqlglot.optimizer.normalize import normalize
from sqlglot.optimizer.optimize_joins import optimize_joins
from sqlglot.optimizer.pushdown_predicates import pushdown_predicates
from sqlglot.optimizer.pushdown_projections import pushdown_projections
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.qualify_columns import quote_identifiers
from sqlglot.optimizer.simplify import simplify
from sqlglot.optimizer.unnest_subqueries import unnest_subqueries
from sqlglot.optimizer.optimizer import RULES

# --------------------------------------------------------------------------
# Injected faulty rules (test instrumentation only)
# --------------------------------------------------------------------------


def drop_outer_join_BUG(expression: exp.Expr) -> exp.Expr:
    """Turn every outer join into an inner join (drops NULL-padded rows)."""
    for join in expression.find_all(exp.Join):
        if join.args.get("side") is not None:
            join.set("side", None)
    return expression


def drop_distinct_BUG(expression: exp.Expr) -> exp.Expr:
    """Remove DISTINCT, re-introducing duplicate rows."""
    for select in expression.find_all(exp.Select):
        select.set("distinct", None)
    return expression


def invert_is_null_BUG(expression: exp.Expr) -> exp.Expr:
    """Flip x IS NULL into x IS NOT NULL and vice versa (NULL filter rewrite)."""
    unwrapped: set[int] = set()
    for node in list(expression.walk()):
        if (
            isinstance(node, exp.Not)
            and isinstance(node.this, exp.Is)
            and isinstance(node.this.expression, exp.Null)
        ):
            unwrapped.add(id(node.this))
            node.replace(node.this)
    for node in list(expression.find_all(exp.Is)):
        if isinstance(node.expression, exp.Null) and id(node) not in unwrapped:
            node.set("negate", not bool(node.args.get("negate")))
    return expression


BUG_RULES = {
    "drop_outer_join_BUG": drop_outer_join_BUG,
    "drop_distinct_BUG": drop_distinct_BUG,
    "invert_is_null_BUG": invert_is_null_BUG,
}

# --------------------------------------------------------------------------
# Named rule sets (all end with the default pipeline; bugs are inserted at
# the end so they act as a stand-in for a buggy rewrite stage)
# --------------------------------------------------------------------------


def _with_bug(name: str):
    return tuple(RULES) + (BUG_RULES[name],)


RULE_SETS: dict[str, t.Sequence] = {
    "default": RULES,
    "no-simplify": tuple(rule for rule in RULES if rule is not simplify),
    "qualify-only": (qualify,),
    "simplify-only": (qualify, simplify),
    "bug-drop-outer-join": _with_bug("drop_outer_join_BUG"),
    "bug-drop-distinct": _with_bug("drop_distinct_BUG"),
    "bug-invert-is-null": _with_bug("invert_is_null_BUG"),
}

# Names used when serializing counterexamples.
RULE_REGISTRY = {rule.__name__: rule for rule in RULES}
RULE_REGISTRY.update(BUG_RULES)
# Rule functions imported but not in the default pipeline; still resolvable
# for replay of counterexamples.
for _extra in (
    annotate_types,
    canonicalize,
    eliminate_ctes,
    eliminate_joins,
    eliminate_subqueries,
    merge_subqueries,
    normalize,
    optimize_joins,
    pushdown_predicates,
    pushdown_projections,
    qualify,
    quote_identifiers,
    simplify,
    unnest_subqueries,
):
    RULE_REGISTRY.setdefault(_extra.__name__, _extra)


def resolve_rules(names: t.Sequence[str]) -> tuple:
    missing = [name for name in names if name not in RULE_REGISTRY]
    if missing:
        raise KeyError(f"unknown optimizer rules: {', '.join(missing)}")
    return tuple(RULE_REGISTRY[name] for name in names)
