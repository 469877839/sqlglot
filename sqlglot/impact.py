"""
Offline impact analysis for upstream schema changes.

Given one query and two snapshots of the upstream schema (``before`` /
``after``), this module compares the query's *result column contract* against
each snapshot, without executing any SQL and without rewriting the query. It is
meant for data-mart workflows where an upstream table is about to change shape
and you need to know which existing SQL will produce a different output.

The analysis is purely static. Each schema snapshot is used independently to
qualify a private copy of the query (column resolution, ``SELECT *`` expansion,
``JOIN USING`` normalization, CTE / set-operation handling) and to infer output
types. The two resulting contracts are then aligned positionally and by
expression identity.

Unknown information is never treated as compatible. When an output name or type
cannot be derived (e.g. because a table is missing from the mapping, a star
cannot be expanded, or qualification fails), the affected columns are reported
as :class:`ImpactCategory.INDETERMINATE` instead of silently passing as
unchanged. Neither the input AST nor the input schema mappings are mutated.

Example:
    >>> from sqlglot.impact import analyze_impact
    >>> before = {"orders": {"id": "INT", "amount": "DECIMAL(10, 2)"}}
    >>> after = {"orders": {"id": "BIGINT", "amount": "DECIMAL(10, 2)", "status": "TEXT"}}
    >>> report = analyze_impact("SELECT id, amount FROM orders", before, after)
    >>> [change.category.value for change in report.changes]
    ['TYPE_CHANGED']
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass, field
from enum import auto

from sqlglot import expressions as exp
from sqlglot.errors import ImpactAnalysisError, OptimizeError
from sqlglot.helper import AutoName
from sqlglot.optimizer.annotate_types import annotate_types
from sqlglot.optimizer.qualify import qualify
from sqlglot.schema import Schema, ensure_schema

if t.TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlglot.dialects.dialect import DialectType


class ImpactCategory(AutoName):
    """How a single output column differs between the two schema snapshots."""

    ADDED = auto()
    """The column only exists in the ``after`` contract."""

    REMOVED = auto()
    """The column only exists in the ``before`` contract."""

    MOVED = auto()
    """The same output expression and name now appear at a different position."""

    RENAMED = auto()
    """The same output expression now exposes a different output name.

    Output names come from the query text, so a schema-only change against one
    fixed query generally cannot produce this category; an upstream column rename
    is reported as ``REMOVED`` + ``ADDED`` (or ``INDETERMINATE`` when metadata is
    missing) because there is no stable identity linking the old and new names.
    """

    TYPE_CHANGED = auto()
    """The same output expression now resolves to a different, fully known type."""

    INDETERMINATE = auto()
    """A difference cannot be ruled out because name or type metadata is missing."""

    @property
    def is_breaking(self) -> bool:
        """Whether consumers of the query output must react to this category.

        Additions are the only non-breaking change: positional consumers may still
        care about ``MOVED``, so it counts as breaking here.
        """
        return self is not ImpactCategory.ADDED


class UncertaintyReason(AutoName):
    """Why an output column could not be fully resolved against a schema snapshot."""

    UNEXPANDED_STAR = auto()
    """A projection star (``*`` or ``t.*``) could not be expanded from the schema."""

    UNKNOWN_TYPE = auto()
    """The output type resolves to ``UNKNOWN``, e.g. an unmodeled column or function."""

    QUALIFICATION_FAILED = auto()
    """Column qualification could not complete, so the query was analyzed without it."""


@dataclass(frozen=True)
class OutputColumn:
    """One output column of the query's result contract, resolved against a single schema snapshot.

    Attributes:
        position: 0-based ordinal of the column in the result set.
        name: the output name of the column (alias, column name, or ``*`` for an
            unexpanded star).
        expression: the qualified, type-annotated projection expression this column
            was derived from. Never mutated by the analysis.
        data_type: the inferred output type, or ``None`` when it could not be
            determined.
        resolved: whether the column is fully determined (no unexpanded star and a
            known type).
        reason: why the column is not resolved, or ``None`` when it is.
        sources: ``(table, column)`` pairs of the base columns this output reads,
            in traversal order. Empty for literals and unresolvable expressions.
    """

    position: int
    name: str
    expression: exp.Expr
    data_type: exp.DataType | None
    resolved: bool
    reason: UncertaintyReason | None = None
    sources: tuple[tuple[str, str], ...] = ()
    fingerprint: int = 0
    group_index: int = -1
    from_star: bool = False

    @property
    def type_sql(self) -> str | None:
        """The output type rendered as SQL, or ``None`` when unknown."""
        return self.data_type.sql() if self.data_type is not None else None

    def to_dict(self) -> dict[str, t.Any]:
        return {
            "position": self.position,
            "name": self.name,
            "expression": self.expression.sql(),
            "type": self.type_sql,
            "resolved": self.resolved,
            "reason": self.reason.value if self.reason else None,
            "sources": [list(source) for source in self.sources],
            "from_star": self.from_star,
        }


@dataclass(frozen=True)
class ColumnChange:
    """A single detected difference between the two result contracts.

    Attributes:
        category: the kind of difference.
        before: the column in the ``before`` contract, ``None`` for additions.
        after: the column in the ``after`` contract, ``None`` for removals.
        reason: populated for ``INDETERMINATE`` changes, explaining which side
            could not be resolved and why.
    """

    category: ImpactCategory
    before: OutputColumn | None = None
    after: OutputColumn | None = None
    reason: str | None = None

    @property
    def name(self) -> str:
        """The output name affected by this change (from whichever side exists)."""
        column = self.after or self.before
        return column.name if column else ""

    def to_dict(self) -> dict[str, t.Any]:
        return {
            "category": self.category.value,
            "before": self.before.to_dict() if self.before else None,
            "after": self.after.to_dict() if self.after else None,
            "reason": self.reason,
        }


@dataclass
class ImpactReport:
    """The result of comparing one query's contract against two schema snapshots.

    Attributes:
        sql: the analyzed query text.
        dialect: the dialect used for parsing and resolution.
        before_contract: output columns resolved against the ``before`` snapshot.
        after_contract: output columns resolved against the ``after`` snapshot.
        changes: detected differences, ordered by the position of the affected
            output (additions and removals first, then stable-position columns).
    """

    sql: str
    dialect: DialectType
    before_contract: list[OutputColumn] = field(default_factory=list)
    after_contract: list[OutputColumn] = field(default_factory=list)
    changes: list[ColumnChange] = field(default_factory=list)

    def changes_of(self, category: ImpactCategory) -> list[ColumnChange]:
        return [change for change in self.changes if change.category is category]

    def breaking_categories(self) -> list[ImpactCategory]:
        return list(
            dict.fromkeys(
                change.category for change in self.changes if change.category.is_breaking
            )
        )

    @property
    def has_breaking_changes(self) -> bool:
        return any(change.category.is_breaking for change in self.changes)

    @property
    def has_indeterminate_changes(self) -> bool:
        return any(change.category is ImpactCategory.INDETERMINATE for change in self.changes)

    def to_dict(self) -> dict[str, t.Any]:
        return {
            "sql": self.sql,
            "dialect": self.dialect,
            "before_contract": [column.to_dict() for column in self.before_contract],
            "after_contract": [column.to_dict() for column in self.after_contract],
            "changes": [change.to_dict() for change in self.changes],
        }

    def __str__(self) -> str:
        lines = [f"Query: {self.sql}"]
        if not self.changes:
            lines.append("Result contract is unchanged.")
            return "\n".join(lines)

        for change in self.changes:
            before = change.before
            after = change.after

            if change.category is ImpactCategory.ADDED and after:
                lines.append(
                    f"  ADDED          #{after.position} {after.name} {after.type_sql or 'UNKNOWN'}"
                )
            elif change.category is ImpactCategory.REMOVED and before:
                lines.append(
                    f"  REMOVED        #{before.position} "
                    f"{before.name} {before.type_sql or 'UNKNOWN'}"
                )
            elif change.category is ImpactCategory.MOVED and before and after:
                lines.append(
                    f"  MOVED          {after.name}: #{before.position} -> #{after.position}"
                )
            elif change.category is ImpactCategory.RENAMED and before and after:
                lines.append(
                    f"  RENAMED        #{after.position} {before.name} -> {after.name}"
                )
            elif change.category is ImpactCategory.TYPE_CHANGED and before and after:
                lines.append(
                    f"  TYPE_CHANGED   #{after.position} {after.name}: "
                    f"{before.type_sql} -> {after.type_sql}"
                )
            elif change.category is ImpactCategory.INDETERMINATE:
                lines.append(f"  INDETERMINATE  {change.name}: {change.reason}")

        return "\n".join(lines)


def analyze_impact(
    sql: str | exp.Expr,
    before_schema: dict[str, object] | Schema | None,
    after_schema: dict[str, object] | Schema | None,
    dialect: DialectType = None,
    db: str | None = None,
    catalog: str | None = None,
) -> ImpactReport:
    """
    Compare the result column contract of one query against two schema snapshots.

    The query is never executed and never rewritten; only its output contract
    (column names, positions and types) is compared. The input AST and the input
    schema mappings are not modified.

    Args:
        sql: the query to analyze, either as SQL text or as a parsed expression.
        before_schema: the upstream schema before the change, either a mapping of
            the form ``{table: {column: type}}`` (optionally nested by db/catalog)
            or a ``Schema`` instance.
        after_schema: the upstream schema after the change, same form as
            ``before_schema``.
        dialect: the SQL dialect used for parsing and name/type resolution.
        db: default database name used to qualify unqualified tables.
        catalog: default catalog name used to qualify unqualified tables.

    Returns:
        An ``ImpactReport`` describing both contracts and their differences.
    """
    expression = exp.maybe_parse(sql, dialect=dialect)
    if not isinstance(expression, exp.Query):
        raise ImpactAnalysisError(
            f"Impact analysis requires a query (SELECT / set operation), "
            f"got {type(expression).__name__}."
        )

    # Pre-qualification projections are the only structural representation stable
    # across snapshots: qualification of an unresolvable column drops its table
    # qualifier, which would make the same projection look like two different ones.
    projections = [hash(projection.unalias()) for projection in expression.selects]

    before = _build_contract(expression, projections, before_schema, dialect, db, catalog)
    after = _build_contract(expression, projections, after_schema, dialect, db, catalog)

    return ImpactReport(
        sql=expression.sql(dialect=dialect),
        dialect=dialect,
        before_contract=before,
        after_contract=after,
        changes=_diff_contracts(before, after),
    )


def _build_contract(
    expression: exp.Expr,
    fingerprints: Sequence[int],
    schema: dict[str, object] | Schema | None,
    dialect: DialectType,
    db: str | None,
    catalog: str | None,
) -> list[OutputColumn]:
    schema = ensure_schema(schema, dialect=dialect)
    original_selects = list(expression.selects)

    try:
        # qualify() rewrites in place, so always hand it a private copy
        qualified = qualify(
            expression.copy(),
            dialect=dialect,
            db=db,
            catalog=catalog,
            schema=schema,
            validate_qualify_columns=False,
            quote_identifiers=False,
        )
        fallback = False
    except OptimizeError:
        # e.g. a JOIN USING key that no longer exists in the snapshot; fall back
        # to table-level qualification so stars stay visible as unresolved
        qualified = qualify(
            expression.copy(),
            dialect=dialect,
            db=db,
            catalog=catalog,
            schema=schema,
            qualify_columns=False,
            validate_qualify_columns=False,
            quote_identifiers=False,
        )
        fallback = True

    try:
        annotated = annotate_types(qualified, schema=schema, dialect=dialect)
    except OptimizeError:
        annotated = qualified

    if isinstance(annotated, exp.Subquery):
        annotated = annotated.unnest()
    if not isinstance(annotated, exp.Query):
        raise ImpactAnalysisError(
            f"Cannot derive a result contract from {type(annotated).__name__}."
        )

    # Walk the resolved projections in lockstep with the original ones: a star
    # expands into one or more output columns, any other projection produces
    # exactly one. This keeps every output attributable to its original source
    # projection even after real star expansion.
    contract: list[OutputColumn] = []
    original_index = 0
    for projection in annotated.selects:
        if projection.is_star:
            group_index = min(original_index, len(original_selects) - 1)
            original_index += 1
        else:
            group_index = min(original_index, len(original_selects) - 1)
            if not original_selects[group_index].is_star:
                original_index += 1

        contract.append(
            _output_column(
                position=len(contract),
                group_index=group_index,
                fingerprint=fingerprints[group_index],
                original=original_selects[group_index],
                projection=projection,
                fallback=fallback,
            )
        )

    return contract


def _output_column(
    position: int,
    group_index: int,
    fingerprint: int,
    original: exp.Expr,
    projection: exp.Expr,
    fallback: bool,
) -> OutputColumn:
    if projection.is_star:
        # a star that survived qualification: its expansion depends on metadata
        # we do not have, so the columns behind it are indeterminate
        return OutputColumn(
            position=position,
            name=projection.output_name,
            expression=projection,
            data_type=None,
            resolved=False,
            reason=(
                UncertaintyReason.QUALIFICATION_FAILED
                if fallback
                else UncertaintyReason.UNEXPANDED_STAR
            ),
            sources=_star_sources(original),
            fingerprint=fingerprint,
            group_index=group_index,
            from_star=True,
        )

    data_type = projection.type
    known = data_type is not None and not data_type.is_type(exp.DType.UNKNOWN)

    return OutputColumn(
        position=position,
        name=projection.output_name,
        expression=projection,
        data_type=data_type if known else None,
        resolved=known and not fallback,
        reason=(
            UncertaintyReason.QUALIFICATION_FAILED
            if fallback
            else None if known else UncertaintyReason.UNKNOWN_TYPE
        ),
        sources=_column_sources(projection),
        fingerprint=fingerprint,
        group_index=group_index,
        from_star=original.is_star,
    )


def _star_sources(original: exp.Expr) -> tuple[tuple[str, str], ...]:
    # the original star's table qualifier, if any, identifies the source whose
    # metadata is missing
    return tuple(
        (column.table, "*") for column in original.find_all(exp.Column) if column.table
    )


def _column_sources(projection: exp.Expr) -> tuple[tuple[str, str], ...]:
    return tuple(
        (column.table, column.name) for column in projection.find_all(exp.Column) if column.table
    )


def _diff_contracts(
    before: Sequence[OutputColumn], after: Sequence[OutputColumn]
) -> list[ColumnChange]:
    changes: list[ColumnChange] = []
    before_groups = _group_contract(before)
    after_groups = _group_contract(after)

    for group_index in dict.fromkeys([*before_groups, *after_groups]):
        before_group = before_groups.get(group_index, [])
        after_group = after_groups.get(group_index, [])

        if before_group and before_group[0].from_star:
            changes.extend(_diff_star_group(before_group, after_group))
        else:
            changes.extend(_diff_projection_group(before_group, after_group))

    # keep a stable, position-oriented order: additions/removals first by their
    # side's ordinal, then paired columns by their after-snapshot ordinal
    changes.sort(key=lambda change: _change_position(change))
    return changes


def _group_contract(contract: Sequence[OutputColumn]) -> dict[int, list[OutputColumn]]:
    groups: dict[int, list[OutputColumn]] = {}
    for column in contract:
        groups.setdefault(column.group_index, []).append(column)
    return groups


def _diff_projection_group(
    before_group: Sequence[OutputColumn], after_group: Sequence[OutputColumn]
) -> list[ColumnChange]:
    # explicit projections are matched structurally; duplicates such as
    # SELECT 1, 1 pair up in projection order
    changes: list[ColumnChange] = []
    for before_column, after_column in zip(before_group, after_group):
        if not before_column.resolved or not after_column.resolved:
            changes.append(_indeterminate(before_column, after_column))
        else:
            changes.extend(_compare_pair(before_column, after_column))

    for column in before_group[len(after_group) :]:
        if column.resolved:
            changes.append(ColumnChange(ImpactCategory.REMOVED, before=column))
        else:
            changes.append(_indeterminate(column, None))
    for column in after_group[len(before_group) :]:
        if column.resolved:
            changes.append(ColumnChange(ImpactCategory.ADDED, after=column))
        else:
            changes.append(_indeterminate(None, column))
    return changes


def _diff_star_group(
    before_group: Sequence[OutputColumn], after_group: Sequence[OutputColumn]
) -> list[ColumnChange]:
    # columns behind a star are identity-based: same (source, name) means the
    # same output even though the star reorders or resizes around it
    if any(not column.resolved for column in [*before_group, *after_group]):
        # an unexpanded star hides an unknown number of columns, so nothing in
        # this group can be proven unchanged, added or removed
        unresolved_side = (
            "after" if any(not column.resolved for column in after_group) else "before"
        )
        changes = [
            _indeterminate(before_column, after_column)
            for before_column, after_column in zip(before_group, after_group)
        ]
        changes += [
            _star_occluded(column, unresolved_side)
            for column in before_group[len(after_group):]
        ]
        changes += [
            _star_occluded(column, unresolved_side)
            for column in after_group[len(before_group):]
        ]
        return changes

    before_by_name: dict[str, list[OutputColumn]] = {}
    after_by_name: dict[str, list[OutputColumn]] = {}
    for column in before_group:
        before_by_name.setdefault(column.name, []).append(column)
    for column in after_group:
        after_by_name.setdefault(column.name, []).append(column)

    changes: list[ColumnChange] = []
    for name in dict.fromkeys([*before_by_name, *after_by_name]):
        for before_column, after_column in zip(
            before_by_name.get(name, []), after_by_name.get(name, [])
        ):
            changes.extend(_compare_pair(before_column, after_column))

        unmatched_before = before_by_name.get(name, [])[len(after_by_name.get(name, [])):]
        unmatched_after = after_by_name.get(name, [])[len(before_by_name.get(name, [])):]
        changes += [
            ColumnChange(ImpactCategory.REMOVED, before=column) for column in unmatched_before
        ]
        changes += [
            ColumnChange(ImpactCategory.ADDED, after=column) for column in unmatched_after
        ]
    return changes


def _change_position(change: ColumnChange) -> tuple[int, int]:
    before = change.before.position if change.before else 10**9
    after = change.after.position if change.after else 10**9
    return (min(before, after), after if change.after else before)


def _compare_pair(before: OutputColumn, after: OutputColumn) -> list[ColumnChange]:
    if not before.resolved or not after.resolved:
        return [_indeterminate(before, after)]

    changes = []
    if before.name != after.name:
        changes.append(ColumnChange(ImpactCategory.RENAMED, before=before, after=after))
    if before.position != after.position:
        changes.append(ColumnChange(ImpactCategory.MOVED, before=before, after=after))
    if before.data_type != after.data_type:
        changes.append(ColumnChange(ImpactCategory.TYPE_CHANGED, before=before, after=after))
    return changes


def _indeterminate(
    before: OutputColumn | None, after: OutputColumn | None
) -> ColumnChange:
    unresolved = [
        (label, column)
        for label, column in (("before", before), ("after", after))
        if column is not None and not column.resolved
    ]
    details = "; ".join(
        f"{label} output #{column.position} {column.name!r} is unresolved: "
        f"{column.reason.value if column.reason else 'unknown'}"
        for label, column in unresolved
    )
    return ColumnChange(
        ImpactCategory.INDETERMINATE,
        before=before,
        after=after,
        reason=f"cannot determine whether this output changed ({details})",
    )


def _star_occluded(column: OutputColumn, occluding_side: str) -> ColumnChange:
    reason = (
        f"cannot determine whether this output changed (the {occluding_side} snapshot "
        f"leaves a star unexpanded, so this column may or may not exist there)"
    )
    if occluding_side == "after":
        return ColumnChange(ImpactCategory.INDETERMINATE, before=column, reason=reason)
    return ColumnChange(ImpactCategory.INDETERMINATE, after=column, reason=reason)
