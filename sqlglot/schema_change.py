"""
Offline impact analysis for upstream schema changes.

Given one SQL query, a dialect, and a "before"/"after" pair of schemas, this module
compares the query's result column contract (names, positions, and types) under both
schemas and reports how the contract would change. The analysis is purely static:
no SQL is executed and no rewrite or migration is attempted.

Example:
    >>> from sqlglot.schema_change import analyze_schema_change
    >>> report = analyze_schema_change(
    ...     "SELECT * FROM t",
    ...     before_schema={"t": {"a": "INT"}},
    ...     after_schema={"t": {"a": "INT", "b": "TEXT"}},
    ... )
    >>> report.summary()
    '1 change(s): 1 added'
    >>> report.changes[0].name_after
    'b'

Differences that stem from actual star expansion are reported as added/removed/
moved/renamed/type_changed, while anything that cannot be determined because of
missing metadata (e.g. an unexpanded `*` or an unknown type) is reported as
indeterminate instead of being treated as compatible.
"""

from __future__ import annotations

import typing as t

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

from sqlglot import exp, parse_one
from sqlglot.dialects.dialect import Dialect, DialectType
from sqlglot.errors import OptimizeError
from sqlglot.optimizer.annotate_types import TypeAnnotator
from sqlglot.optimizer.qualify import qualify
from sqlglot.schema import MappingSchema, Schema, ensure_schema

__all__ = [
    "ChangeKind",
    "ColumnChange",
    "ColumnContract",
    "SchemaChangeReport",
    "analyze_schema_change",
]


class ChangeKind(Enum):
    """The classification of a single difference between two result contracts."""

    ADDED = "added"
    REMOVED = "removed"
    MOVED = "moved"
    RENAMED = "renamed"
    TYPE_CHANGED = "type_changed"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class ColumnContract:
    """One column of a query's result contract under a given schema."""

    position: int
    name: str
    type: t.Optional[str]
    expression: str
    is_star: bool = False


@dataclass(frozen=True)
class ColumnChange:
    """A single detected difference between the before/after contracts."""

    kind: ChangeKind
    reason: str
    name_before: t.Optional[str] = None
    name_after: t.Optional[str] = None
    position_before: t.Optional[int] = None
    position_after: t.Optional[int] = None
    type_before: t.Optional[str] = None
    type_after: t.Optional[str] = None
    expression_before: t.Optional[str] = None
    expression_after: t.Optional[str] = None
    metadata_limited: bool = False


@dataclass
class SchemaChangeReport:
    """The result of comparing a query's output contract under two schemas."""

    dialect: str
    before_contract: list[ColumnContract]
    after_contract: list[ColumnContract]
    changes: list[ColumnChange]
    warnings: list[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.changes)

    @property
    def indeterminate(self) -> list[ColumnChange]:
        """Changes that could not be determined, e.g. due to missing metadata."""
        return [c for c in self.changes if c.kind is ChangeKind.INDETERMINATE]

    def summary(self) -> str:
        if not self.changes:
            return "No contract changes detected."

        counts: dict[ChangeKind, int] = defaultdict(int)
        for change in self.changes:
            counts[change.kind] += 1

        parts = ", ".join(
            f"{count} {kind.value}"
            for kind, count in sorted(counts.items(), key=lambda kv: kv[0].value)
        )
        return f"{len(self.changes)} change(s): {parts}"


def analyze_schema_change(
    sql: str | exp.Expr,
    before_schema: dict[str, object] | Schema | None = None,
    after_schema: dict[str, object] | Schema | None = None,
    dialect: DialectType = None,
) -> SchemaChangeReport:
    """
    Compare the result column contract of a query under two versions of a schema.

    Args:
        sql: The query to analyze, either as a SQL string or an expression. The
            expression is copied before analysis and is never mutated.
        before_schema: The schema before the upstream change.
        after_schema: The schema after the upstream change.
        dialect: The dialect used to parse, qualify, and render types.

    Returns:
        A `SchemaChangeReport` describing added/removed/moved/renamed/type-changed
        columns, as well as anything that could not be determined. Unknown types are
        never treated as compatible.
    """
    dialect_obj = Dialect.get_or_raise(dialect)
    query = _as_query(sql, dialect_obj)

    warnings: list[str] = []
    before = _extract_contract(query, before_schema, dialect_obj, warnings)
    after = _extract_contract(query, after_schema, dialect_obj, warnings)
    changes = _diff_contracts(before, after)

    return SchemaChangeReport(
        dialect=dialect_obj.__class__.__name__,
        before_contract=before,
        after_contract=after,
        changes=changes,
        warnings=warnings,
    )


def _as_query(sql: str | exp.Expr, dialect: Dialect) -> exp.Query:
    expression = parse_one(sql, dialect=dialect) if isinstance(sql, str) else sql.copy()

    while isinstance(expression, exp.Subquery):
        expression = expression.this

    if not isinstance(expression, exp.Query):
        raise ValueError(
            f"Expected a query expression (SELECT, UNION, ...), got {type(expression).__name__}"
        )

    return expression


def _isolated_schema(schema: dict[str, object] | Schema | None, dialect: Dialect) -> Schema:
    # Work on a private copy so the analysis never mutates the caller's schema
    if isinstance(schema, MappingSchema):
        return schema.copy()
    return ensure_schema(schema, dialect=dialect)


def _extract_contract(
    query: exp.Query,
    schema: dict[str, object] | Schema | None,
    dialect: Dialect,
    warnings: list[str],
) -> list[ColumnContract]:
    schema_obj = _isolated_schema(schema, dialect)
    annotator: t.Optional[TypeAnnotator] = None

    try:
        expression = qualify(
            query.copy(),
            dialect=dialect,
            schema=schema_obj,
            allow_partial_qualification=True,
            validate_qualify_columns=False,
        )
        annotator = TypeAnnotator(schema=schema_obj)
        annotator.annotate(expression)
    except OptimizeError as e:
        warnings.append(f"Could not fully qualify the query, types may be unknown: {e}")
        expression = query.copy()

    if isinstance(expression, exp.SetOperation):
        selects = expression.selects
        setop_columns = annotator._get_setop_column_types(expression) if annotator else []
        if setop_columns and len(setop_columns) == len(selects):
            return [
                ColumnContract(
                    position=i,
                    name=name or selects[i].sql(dialect=dialect),
                    type=_type_sql(column_type, dialect),
                    expression=selects[i].sql(dialect=dialect),
                    is_star=bool(selects[i].is_star),
                )
                for i, (name, column_type) in enumerate(setop_columns)
            ]
    else:
        selects = expression.selects

    return [
        ColumnContract(
            position=i,
            name=select.output_name or select.sql(dialect=dialect),
            type=_type_sql(select.type, dialect),
            expression=select.sql(dialect=dialect),
            is_star=bool(select.is_star),
        )
        for i, select in enumerate(selects)
    ]


def _type_sql(dtype: exp.DataType | exp.DType | None, dialect: Dialect) -> t.Optional[str]:
    if isinstance(dtype, exp.DType):
        dtype = dtype.into_expr()
    if dtype is None or dtype.is_type(exp.DType.UNKNOWN):
        return None
    return dtype.sql(dialect=dialect)


def _diff_contracts(
    before: list[ColumnContract], after: list[ColumnContract]
) -> list[ColumnChange]:
    changes: list[ColumnChange] = []
    used_before: set[int] = set()
    used_after: set[int] = set()

    # Pair up unexpanded stars by order of appearance; their covered columns are opaque
    before_stars = [i for i, c in enumerate(before) if c.is_star]
    after_stars = [i for i, c in enumerate(after) if c.is_star]
    for i, j in zip(before_stars, after_stars):
        used_before.add(i)
        used_after.add(j)
        changes.append(
            _star_change(
                before[i],
                after[j],
                "Star is not expanded on either side (missing source metadata); "
                "the columns it covers cannot be compared",
            )
        )

    star_unmatched = False
    for i in before_stars[len(after_stars) :]:
        used_before.add(i)
        star_unmatched = True
        changes.append(
            _star_change(
                before[i],
                None,
                "Star is not expanded under the before schema (missing source metadata); "
                "it cannot be mapped to columns under the after schema",
            )
        )
    for j in after_stars[len(before_stars) :]:
        used_after.add(j)
        star_unmatched = True
        changes.append(
            _star_change(
                None,
                after[j],
                "Star is not expanded under the after schema (missing source metadata); "
                "it cannot be mapped to columns under the before schema",
            )
        )

    # Match remaining columns by name, pairing the k-th occurrence with the k-th
    # occurrence so that duplicate output names are handled positionally
    before_by_name: dict[str, deque[int]] = defaultdict(deque)
    for i, column in enumerate(before):
        if i not in used_before:
            before_by_name[column.name].append(i)

    for j, after_column in enumerate(after):
        if j in used_after:
            continue
        queue = before_by_name.get(after_column.name)
        if queue:
            i = queue.popleft()
            used_before.add(i)
            used_after.add(j)
            change = _classify_pair(before[i], after_column)
            if change:
                changes.append(change)

    leftover_before = [i for i in range(len(before)) if i not in used_before]
    leftover_after = [j for j in range(len(after)) if j not in used_after]

    if star_unmatched:
        # An unexpanded star on one side makes it impossible to tell whether leftover
        # columns on the other side are real additions/removals or star-covered
        for i in leftover_before:
            changes.append(
                _metadata_limited_change(
                    before[i],
                    None,
                    "A star is not expanded under the after schema, so this column cannot be "
                    "confirmed as removed; it may still be covered by the star",
                )
            )
        for j in leftover_after:
            changes.append(
                _metadata_limited_change(
                    None,
                    after[j],
                    "A star is not expanded under the before schema, so this column cannot be "
                    "confirmed as added; it may have been covered by the star",
                )
            )
    else:
        remaining_after = set(leftover_after)
        for i in leftover_before:
            column = before[i]
            if i in remaining_after:
                remaining_after.discard(i)
                changes.append(_rename_change(column, after[i]))
            else:
                changes.append(
                    ColumnChange(
                        kind=ChangeKind.REMOVED,
                        reason=f"Column '{column.name}' at position {i} no longer appears "
                        "in the result under the after schema",
                        name_before=column.name,
                        position_before=i,
                        type_before=column.type,
                        expression_before=column.expression,
                    )
                )
        for j in leftover_after:
            if j in remaining_after:
                column = after[j]
                changes.append(
                    ColumnChange(
                        kind=ChangeKind.ADDED,
                        reason=f"Column '{column.name}' appears at position {j} "
                        "in the result under the after schema",
                        name_after=column.name,
                        position_after=j,
                        type_after=column.type,
                        expression_after=column.expression,
                    )
                )

    return sorted(
        changes,
        key=lambda c: (
            c.position_before if c.position_before is not None else c.position_after or 0,
            c.kind.value,
        ),
    )


def _classify_pair(before: ColumnContract, after: ColumnContract) -> t.Optional[ColumnChange]:
    kwargs: dict[str, t.Any] = {
        "name_before": before.name,
        "name_after": after.name,
        "position_before": before.position,
        "position_after": after.position,
        "type_before": before.type,
        "type_after": after.type,
        "expression_before": before.expression,
        "expression_after": after.expression,
    }

    if before.type is None or after.type is None:
        if before.type is None and after.type is None:
            side = "either side"
        else:
            side = "the before side" if before.type is None else "the after side"
        return ColumnChange(
            kind=ChangeKind.INDETERMINATE,
            reason=f"Type of column '{after.name}' is unknown on {side}; "
            "compatibility cannot be determined",
            metadata_limited=True,
            **kwargs,
        )

    if before.type != after.type:
        return ColumnChange(
            kind=ChangeKind.TYPE_CHANGED,
            reason=f"Column '{after.name}' changed type from {before.type} to {after.type}",
            **kwargs,
        )

    if before.position != after.position:
        return ColumnChange(
            kind=ChangeKind.MOVED,
            reason=f"Column '{after.name}' moved from position "
            f"{before.position} to {after.position}",
            **kwargs,
        )

    return None


def _rename_change(before: ColumnContract, after: ColumnContract) -> ColumnChange:
    reason = (
        f"Column at position {before.position} was renamed from "
        f"'{before.name}' to '{after.name}' (inferred from position)"
    )
    if before.type is None or after.type is None:
        reason += "; its type could not be determined on both sides"
    elif before.type != after.type:
        reason += f"; its type also changed from {before.type} to {after.type}"

    return ColumnChange(
        kind=ChangeKind.RENAMED,
        reason=reason,
        name_before=before.name,
        name_after=after.name,
        position_before=before.position,
        position_after=after.position,
        type_before=before.type,
        type_after=after.type,
        expression_before=before.expression,
        expression_after=after.expression,
    )


def _star_change(
    before: t.Optional[ColumnContract], after: t.Optional[ColumnContract], reason: str
) -> ColumnChange:
    return ColumnChange(
        kind=ChangeKind.INDETERMINATE,
        reason=reason,
        name_before=before.name if before else None,
        name_after=after.name if after else None,
        position_before=before.position if before else None,
        position_after=after.position if after else None,
        expression_before=before.expression if before else None,
        expression_after=after.expression if after else None,
        metadata_limited=True,
    )


def _metadata_limited_change(
    before: t.Optional[ColumnContract], after: t.Optional[ColumnContract], reason: str
) -> ColumnChange:
    return ColumnChange(
        kind=ChangeKind.INDETERMINATE,
        reason=reason,
        name_before=before.name if before else None,
        name_after=after.name if after else None,
        position_before=before.position if before else None,
        position_after=after.position if after else None,
        type_before=before.type if before else None,
        type_after=after.type if after else None,
        expression_before=before.expression if before else None,
        expression_after=after.expression if after else None,
        metadata_limited=True,
    )
