"""Reducer for failing differential cases.

Two cooperating passes, both guarded by a re-check that the candidate still
mismatches:

1. ddmin over the INSERT statements (data reduction, delta-debugging style).
2. Greedy removal of query clauses (LIMIT, HAVING, ORDER BY, WHERE, GROUP BY,
   DISTINCT) and select items, re-serialized from the AST.

The ``budget`` caps how many candidate evaluations (``is_failure`` calls) a
shrink run may spend, making reduction time reproducible.
"""

from __future__ import annotations

import typing as t
from dataclasses import replace

import sqlglot
from sqlglot import exp

from .runner import Case

_CLAUSE_KEYS = ("limit", "having", "order", "where", "group", "distinct")


class ShrinkBudgetExhausted(Exception):
    pass


class Shrinker:
    def __init__(self, is_failure: t.Callable[[Case], bool], budget: int = 200):
        self.is_failure = is_failure
        self.budget = budget
        self.spent = 0

    def check(self, case: Case) -> bool:
        if self.spent >= self.budget:
            raise ShrinkBudgetExhausted
        self.spent += 1
        return self.is_failure(case)

    # -- data reduction -----------------------------------------------------
    def shrink_inserts(self, case: Case) -> Case:
        # Row-level ddmin inside each INSERT, then statement-level ddmin.
        case = self.shrink_insert_rows(case)
        inserts = list(case.inserts)
        if len(inserts) < 2:
            return case
        granularity = 2
        while len(inserts) >= 2:
            subset_size = max(1, len(inserts) // granularity)
            reduced = False
            for start in range(0, len(inserts), subset_size):
                candidate = inserts[:start] + inserts[start + subset_size :]
                if not candidate:
                    continue
                trial = replace(case, inserts=candidate)
                try:
                    fails = self.check(trial)
                except ShrinkBudgetExhausted:
                    return replace(case, inserts=inserts)
                if fails:
                    inserts = candidate
                    granularity = max(2, granularity - 1)
                    reduced = True
                    break
            if not reduced:
                if granularity >= len(inserts):
                    break
                granularity = min(len(inserts), granularity * 2)
        return replace(case, inserts=inserts)

    def shrink_insert_rows(self, case: Case) -> Case:
        inserts = list(case.inserts)
        for index, statement in enumerate(inserts):
            try:
                parsed = sqlglot.parse_one(statement, read="sqlite")
                tuples = list(parsed.expression.expressions)
            except Exception:
                continue
            if len(tuples) < 2:
                continue
            kept = self._ddmin_list(
                tuples,
                lambda subset: self._try_rows(case, inserts, index, statement, subset),
            )
            if len(kept) != len(tuples):
                clone = sqlglot.parse_one(statement, read="sqlite")
                clone.expression.set("expressions", list(kept))
                inserts[index] = clone.sql("sqlite")
        return replace(case, inserts=inserts)

    def _try_rows(self, case, inserts, index, statement, subset) -> bool:
        clone = sqlglot.parse_one(statement, read="sqlite")
        clone.expression.set("expressions", list(subset))
        candidate = list(inserts)
        candidate[index] = clone.sql("sqlite")
        return self.check(replace(case, inserts=candidate))

    def _ddmin_list(self, items, predicate):
        """Classic ddmin over an ordered list of items."""
        items = list(items)
        granularity = 2
        while len(items) >= 2:
            chunk = max(1, len(items) // granularity)
            reduced = False
            for start in range(0, len(items), chunk):
                subset = items[:start] + items[start + chunk :]
                if subset:
                    try:
                        fails = predicate(subset)
                    except ShrinkBudgetExhausted:
                        return items
                    if fails:
                        items = subset
                        granularity = max(2, granularity - 1)
                        reduced = True
                        break
            if not reduced:
                if granularity >= len(items):
                    break
                granularity = min(len(items), granularity * 2)
        return items

    # -- query reduction ------------------------------------------------------
    def shrink_query(self, case: Case) -> Case:
        try:
            expression = sqlglot.parse_one(case.query, read="sqlite")
        except Exception:
            return case
        current = case
        improved = True
        while improved:
            improved = False
            for candidate_sql in self._removal_candidates(current.query):
                trial = replace(current, query=candidate_sql)
                try:
                    fails = self.check(trial)
                except ShrinkBudgetExhausted:
                    return current
                if fails:
                    current = trial
                    improved = True
                    break
        return current

    def _removal_candidates(self, query: str) -> t.Iterator[str]:
        try:
            expression = sqlglot.parse_one(query, read="sqlite")
        except Exception:
            return
        select = expression.find(exp.Select)
        for key in _CLAUSE_KEYS:
            if expression.args.get(key) is not None:
                if key == "group" and select is not None and not _all_aggregate_or_literal(select):
                    # Dropping GROUP BY with a bare column in the select list
                    # makes SQLite pick an arbitrary row -> nondeterministic.
                    continue
                clone = expression.copy()
                clone.set(key, None)
                yield clone.sql("sqlite")
        if select is not None and select.args.get("joins"):
            joins = select.args["joins"]
            for index in range(len(joins)):
                clone = expression.copy()
                clone_select = clone.find(exp.Select)
                kept = [j for i, j in enumerate(joins) if i != index]
                clone_select.set("joins", kept or None)
                yield clone.sql("sqlite")
        if select is not None and len(select.expressions) > 1:
            for index in range(len(select.expressions)):
                clone = expression.copy()
                clone_select = clone.find(exp.Select)
                kept = [e for i, e in enumerate(clone_select.expressions) if i != index]
                clone_select.set("expressions", kept)
                yield clone.sql("sqlite")

    def shrink(self, case: Case) -> Case:
        try:
            case = self.shrink_inserts(case)
            case = self.shrink_query(case)
            case = self.shrink_inserts(case)
        except ShrinkBudgetExhausted:
            pass
        return case


def shrink_case(
    case: Case, is_failure: t.Callable[[Case], bool], budget: int = 200
) -> tuple[Case, int]:
    shrinker = Shrinker(is_failure, budget=budget)
    reduced = shrinker.shrink(case)
    return reduced, shrinker.spent


def _all_aggregate_or_literal(select: exp.Select) -> bool:
    for item in select.expressions:
        if isinstance(item, (exp.AggFunc, exp.Literal)):
            continue
        if isinstance(item, exp.Alias) and isinstance(item.this, (exp.AggFunc, exp.Literal)):
            continue
        return False
    return True
