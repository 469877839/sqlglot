"""Seeded generator of small SQLite-executable SELECT statements.

The grammar deliberately stresses the areas where optimizer rewrites go
wrong: NULLs, duplicate rows, outer joins and aggregation combinations.
Everything is derived from a single ``random.Random(seed)`` so a seed fully
replays a case.
"""

from __future__ import annotations

import random
import typing as t

from . import schema as sch
from .runner import Case

LITERALS_INT = ["NULL", "0", "1", "2", "3"]
LITERALS_TEXT = ["NULL", "'x'", "'y'"]
AGGREGATES = (
    "COUNT(*)", "COUNT({c})", "COUNT(DISTINCT {c})",
    "SUM({c})", "MIN({c})", "MAX({c})", "AVG({c})",
)


class QueryBuilder:
    def __init__(self, rng: random.Random, schema: dict[str, dict[str, str]]):
        self.rng = rng
        self.schema = schema
        self.int_cols = sch.qualified_columns(schema, sch.INT)
        self.text_cols = sch.qualified_columns(schema, sch.TEXT)
        self.all_cols = self.int_cols + self.text_cols

    # -- scalar fragments -------------------------------------------------
    def col(self, tables: t.Sequence[str], type_: str = sch.INT) -> str:
        pool = [
            f"{t}.{c}" for t, c, ty in self.all_cols if t in tables and ty == type_
        ]
        return self.rng.choice(pool)

    def int_term(self, tables: t.Sequence[str]) -> str:
        return self.rng.choice([self.col(tables), self.rng.choice(LITERALS_INT)])

    def condition(self, tables: t.Sequence[str], depth: int = 0) -> str:
        rng = self.rng
        if depth > 1:
            return self.predicate(tables)
        roll = rng.random()
        if roll < 0.30:
            return self.predicate(tables)
        if roll < 0.45:
            left = self.condition(tables, depth + 1)
            right = self.condition(tables, depth + 1)
            return f"({left}) AND ({right})"
        if roll < 0.60:
            return f"({self.condition(tables, depth + 1)}) OR ({self.condition(tables, depth + 1)})"
        if roll < 0.72:
            return f"NOT ({self.condition(tables, depth + 1)})"
        return self.predicate(tables)

    def predicate(self, tables: t.Sequence[str]) -> str:
        rng = self.rng
        roll = rng.random()
        if roll < 0.16:
            return f"{self.col(tables)} IS NULL"
        if roll < 0.28:
            return f"{self.col(tables)} IS NOT NULL"
        if roll < 0.52:
            op = rng.choice(["=", "<>", "<", "<=", ">", ">="])
            return f"{self.int_term(tables)} {op} {self.int_term(tables)}"
        if roll < 0.62:
            values = ", ".join(rng.sample(LITERALS_INT, k=rng.randint(1, 3)))
            return f"{self.col(tables)} IN ({values})"
        if roll < 0.70:
            return f"{self.col(tables)} BETWEEN {rng.randint(0, 1)} AND {rng.randint(2, 3)}"
        if roll < 0.78 and any(t in tables for t, _, ty in self.all_cols if ty == sch.TEXT):
            return f"{self.col(tables, sch.TEXT)} = {rng.choice(LITERALS_TEXT)}"
        if roll < 0.86:
            return f"COALESCE({self.col(tables)}, {rng.randint(0, 3)}) = {rng.randint(0, 3)}"
        return f"({self.col(tables)} = {self.col(tables)}) IS NOT NULL"

    # -- FROM clause --------------------------------------------------------
    def from_clause(self) -> tuple[str, list[str]]:
        rng = self.rng
        tables = list(self.schema)
        roll = rng.random()
        if roll < 0.45:
            picked = rng.sample(tables, k=rng.choice([1, 1, 2]))
            return ", ".join(picked), picked
        left, right = rng.sample(tables, k=2)
        join_type = rng.choices(
            ["JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL JOIN", "CROSS JOIN"],
            weights=[30, 30, 12, 12, 16],
        )[0]
        if join_type == "CROSS JOIN":
            return f"{left} CROSS JOIN {right}", [left, right]
        on = self.join_condition(left, right)
        return f"{left} {join_type} {right} ON {on}", [left, right]

    def join_condition(self, left: str, right: str) -> str:
        rng = self.rng
        left_ints = [c for t, c, ty in self.int_cols if t == left]
        right_ints = [c for t, c, ty in self.int_cols if t == right]
        parts = [f"{left}.{rng.choice(left_ints)} = {right}.{rng.choice(right_ints)}"]
        if rng.random() < 0.35:
            extra = self.predicate([left, right])
            parts.append(extra)
        return " AND ".join(parts)

    # -- SELECT list ---------------------------------------------------------
    def select_item(self, tables: t.Sequence[str], grouped: bool, group_cols: list[str]) -> str:
        rng = self.rng
        roll = rng.random()
        if grouped:
            # Only aggregates or GROUP BY keys: SQLite allows bare columns in a
            # grouped query but picks an arbitrary row, which would introduce
            # nondeterminism unrelated to the optimizer.
            if roll < 0.65:
                template = rng.choice(AGGREGATES)
                return template.format(c=self.col(tables))
            return rng.choice(group_cols)
        if roll < 0.12:
            return rng.choice(LITERALS_INT)
        if roll < 0.2:
            return f"COALESCE({self.col(tables)}, {rng.randint(0, 3)})"
        return self.col(tables)

    # -- whole query ----------------------------------------------------------
    def build(self) -> tuple[str, bool]:
        rng = self.rng
        from_sql, tables = self.from_clause()

        grouped = rng.random() < 0.45
        group_cols: list[str] = []
        if grouped:
            candidates = [f"{t}.{c}" for t, c, _ in self.all_cols if t in tables]
            group_cols = rng.sample(candidates, k=rng.randint(1, min(2, len(candidates))))

        distinct = rng.random() < 0.2
        items = [self.select_item(tables, grouped, group_cols) for _ in range(rng.randint(1, 3))]
        if distinct:
            # DISTINCT over aggregate-free duplicates is the interesting case;
            # keep items deterministic for it.
            items = items[: rng.randint(1, len(items))]

        parts = [f"SELECT {'DISTINCT ' if distinct else ''}{', '.join(items)}", f"FROM {from_sql}"]

        if rng.random() < 0.55:
            parts.append(f"WHERE {self.condition(tables)}")
        if grouped:
            parts.append(f"GROUP BY {', '.join(group_cols)}")
            if rng.random() < 0.5:
                agg = rng.choice(AGGREGATES).format(c=self.col(tables))
                op = rng.choice([">", ">=", "=", "<>"])
                parts.append(f"HAVING {agg} {op} {rng.randint(0, 3)}")

        ordered = rng.random() < 0.3
        if ordered:
            keys = ", ".join(str(i) for i in range(1, rng.randint(1, len(items)) + 1))
            parts.append(f"ORDER BY {keys}")

        if rng.random() < 0.15:
            parts.append(f"LIMIT {rng.randint(1, 4)}")

        return " ".join(parts), ordered


def generate_case(seed: int, profile: str = "small") -> Case:
    rng = random.Random(seed)
    schema = sch.SCHEMAS[profile]
    builder = QueryBuilder(rng, schema)
    query, ordered = builder.build()
    return Case(
        seed=seed,
        schema=schema,
        ddl=sch.ddl_for(schema),
        inserts=sch.inserts_for(profile, schema),
        query=query,
        ordered=ordered,
        profile=profile,
    )
