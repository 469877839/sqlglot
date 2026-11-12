"""Counterexample file format (JSON, deterministic, replayable).

A counterexample file is a single JSON object with the following fields:

    version        - format version
    seed           - generator seed that produced the original case
    profile        - data profile name
    rules          - optimizer rule names, in application order
    schema         - {table: {column: type}} used for qualification
    ddl            - CREATE TABLE statements
    inserts        - reduced INSERT statements
    query          - (possibly reduced) original SQLite SELECT
    optimized_sql  - optimizer output for ``query``
    ordered        - whether the query has a top-level ORDER BY
    original_rows  - result rows of ``query``
    optimized_rows - result rows of ``optimized_sql``
    shrink         - {"evaluations": int, "budget": int}

Replay with: ``python -m tools.difftest replay <file.json>``.
"""

from __future__ import annotations

import json
import typing as t

from .runner import Case, Outcome, has_top_level_order_by

FORMAT_VERSION = 1


def rule_names(rules: t.Sequence) -> list[str]:
    return [getattr(rule, "__name__", str(rule)) for rule in rules]


def write_counterexample(
    path: str,
    case: Case,
    outcome: Outcome,
    rules: t.Sequence,
    shrink_evaluations: int,
    shrink_budget: int,
) -> None:
    payload = {
        "version": FORMAT_VERSION,
        "seed": case.seed,
        "profile": case.profile,
        "rules": rule_names(rules),
        "schema": case.schema,
        "ddl": case.ddl,
        "inserts": case.inserts,
        "query": case.query,
        "optimized_sql": outcome.optimized_sql,
        "ordered": has_top_level_order_by(case.query),
        "original_rows": _rows(outcome.original_rows),
        "optimized_rows": _rows(outcome.optimized_rows),
        "shrink": {"evaluations": shrink_evaluations, "budget": shrink_budget},
    }
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def read_counterexample(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def case_from_payload(payload: dict) -> Case:
    return Case(
        seed=payload["seed"],
        schema=payload["schema"],
        ddl=list(payload["ddl"]),
        inserts=list(payload["inserts"]),
        query=payload["query"],
        ordered=payload["ordered"],
        profile=payload.get("profile", "small"),
    )


def _rows(rows: list | None) -> list:
    if rows is None:
        return []
    return [list(row) for row in rows]
