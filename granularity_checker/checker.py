"""Conservative join fan-out (grain) checker.

The checker answers one narrow question: *before the aggregate expressions of
this query block run, can any join multiply the rows contributed by the
already-joined relations?*

Verdicts per join:

* ``SAFE``    - the right-hand relation can be proved unique on the join
                columns, so it contributes at most one row per left row.
* ``RISK``    - the right-hand relation's grain is known, and the join
                columns do not cover any of its unique keys: row
                multiplication is possible on the stated metadata.
* ``UNKNOWN`` - not enough information to prove uniqueness. The tool refuses
                to guess, and reports what evidence is missing.

A SAFE verdict is a syntactic/metadata proof about row multiplicity only. It
does not assert that any business metric is computed correctly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, traverse_scope

from .grain import Grain
from .paths import join_on_columns, relation_label, render_equalities
from .schema_model import DeclaredSchema

SAFE = "SAFE"
RISK = "RISK"
UNKNOWN = "UNKNOWN"


@dataclass
class JoinVerdict:
    scope_kind: str
    left_relation: str
    right_relation: str
    right_node_sql: str
    join_kind: str
    on_sql: str
    verdict: str
    reason: str
    equality_columns: list[str]
    right_proven_keys: list[tuple[str, ...]]
    location: str = ""


@dataclass
class CheckReport:
    sql: str
    verdicts: list[JoinVerdict] = field(default_factory=list)
    grains: dict[str, Grain] = field(default_factory=dict)
    degradations: list[str] = field(default_factory=list)
    parse_error: str | None = None

    @property
    def worst_verdict(self) -> str | None:
        order = {SAFE: 0, UNKNOWN: 1, RISK: 2}
        ranked = [v.verdict for v in self.verdicts]
        if not ranked:
            return None
        return max(ranked, key=lambda k: order[k])

    def format_text(self) -> str:
        lines: list[str] = []
        if self.parse_error:
            return f"PARSE ERROR: {self.parse_error}"
        lines.append(f"Overall worst verdict: {self.worst_verdict or 'NO JOINS'}")
        for v in self.verdicts:
            lines.append("")
            lines.append(
                f"[{v.verdict}] {v.join_kind} {v.left_relation} -> {v.right_relation}"
            )
            lines.append(f"    ON: {v.on_sql or '(no equi-join condition)'}")
            lines.append(f"    why: {v.reason}")
            if v.right_proven_keys:
                pretty = ", ".join("(" + ", ".join(k) + ")" for k in v.right_proven_keys)
                lines.append(f"    proven unique keys of {v.right_relation}: {pretty}")
            if v.location:
                lines.append(f"    at: {v.location}")
        for note in self.degradations:
            lines.append("")
            lines.append(f"[DEGRADED] {note}")
        lines.append("")
        lines.append(
            "Reminder: SAFE means 'no row multiplication provable on the "
            "declared keys', not 'the metric is correct'."
        )
        return "\n".join(lines)


def analyze_text(sql: str, schema: DeclaredSchema, dialect: str | None = None) -> CheckReport:
    try:
        expression = sqlglot.parse_one(sql, dialect=dialect)
    except sqlglot.errors.ParseError as exc:
        return CheckReport(sql=sql, parse_error=str(exc))
    return analyze(expression, schema, original_sql=sql)


def analyze(expression: exp.Query, schema: DeclaredSchema,
            original_sql: str | None = None) -> CheckReport:
    report = CheckReport(sql=original_sql or expression.sql())
    analyzer = _Analyzer(schema, report)
    analyzer.run(expression)
    return report


class _Analyzer:
    def __init__(self, schema: DeclaredSchema, report: CheckReport) -> None:
        self.schema = schema
        self.report = report
        # id(scope.expression) -> Grain proved for that relation's output.
        self._grain_by_expr: dict[int, Grain] = {}
        self._scope_by_expr: dict[int, Scope] = {}

    # ------------------------------------------------------------------ run
    def run(self, expression: exp.Query) -> None:
        recursive = self._find_recursive_cte(expression)
        if recursive:
            self.report.degradations.append(
                f"recursive CTE '{recursive}' is unsupported; its grain is "
                "treated as unknown and every join through it is UNKNOWN"
            )

        qualified = self._qualify_copy(expression)
        if qualified is None:
            return

        scopes = list(traverse_scope(qualified))
        for scope in scopes:
            self._scope_by_expr[id(scope.expression)] = scope

        for scope in scopes:  # post-order: children before parents
            self._analyze_scope(scope)

    def _qualify_copy(self, expression: exp.Query) -> exp.Query | None:
        copy = expression.copy()
        sqlglot_schema = {
            meta.name.lower(): {c.lower(): "?" for c in meta.columns}
            for meta in self.schema.tables.values()
        }
        try:
            return qualify(
                copy,
                schema=sqlglot_schema,
                validate_qualify_columns=False,
                quote_identifiers=False,
                identify=False,
            )
        except sqlglot.errors.OptimizeError as exc:
            self.report.degradations.append(
                f"name qualification failed ({exc}); results may be UNKNOWN"
            )
            return copy

    # -------------------------------------------------------------- per scope
    def _analyze_scope(self, scope: Scope) -> None:
        node = scope.expression

        if isinstance(node, exp.Union):
            grain = self._grain_for_set_operation(scope, node)
        elif isinstance(node, exp.Select):
            grain = self._grain_for_select(scope, node)
        else:  # pragma: no cover - defensive for future node kinds
            grain = Grain(degraded=True, degraded_reason=f"unsupported {type(node).__name__}")

        if scope.is_subquery and self._looks_correlated(scope):
            scope_label = self._scope_label(scope)
            self.report.degradations.append(
                f"correlated subquery {scope_label} references an outer scope; "
                "its grain cannot be established and joins through it are UNKNOWN"
            )
            grain = Grain(
                unique_keys=list(grain.unique_keys),
                evidence=list(grain.evidence),
                degraded=True,
                degraded_reason="correlated subquery: outer references present",
            )

        self._grain_by_expr[id(node)] = grain
        self.report.grains[self._scope_label(scope)] = grain

    def _scope_label(self, scope: Scope) -> str:
        node = scope.expression
        if isinstance(node, exp.Select):
            sources = list(scope.selected_sources)
            return f"select[{','.join(sources) or '?'}]"
        if isinstance(node, exp.Union):
            return f"setop[{node.key.lower()}]"
        return f"{type(node).__name__}"

    # --------------------------------------------------------------- grains
    def _grain_for_select(self, scope: Scope, node: exp.Select) -> Grain:
        if self._is_in_recursive_cte(scope):
            return Grain(degraded=True, degraded_reason="recursive CTE body")

        # First, derive the grain of the whole FROM/JOIN chain and record
        # verdicts along the way.
        chain_grain = self._analyze_join_chain(scope, node)

        group = node.args.get("group")
        distinct = node.args.get("distinct")
        evidence: list[str] = []

        if isinstance(group, exp.Group):
            keys = self._group_key_output_names(group, node)
            if keys:
                evidence.append(f"GROUP BY ({', '.join(keys)}) makes the grouping columns unique per group")
                return Grain(unique_keys=[tuple(keys)], evidence=evidence)
            evidence.append("GROUP BY uses expressions/ordinals: no plain-column grain can be proved")
            return Grain(evidence=evidence)

        if distinct is not None:
            cols = tuple(self._output_column_names(node))
            if cols:
                evidence.append(f"DISTINCT makes the full projection ({', '.join(cols)}) unique")
                return Grain(unique_keys=[cols], evidence=evidence)
            return Grain(evidence=["DISTINCT with unresolvable projection: grain unknown"])

        if node.args.get("limit") is not None:
            evidence.append("LIMIT does not establish a grain")

        return chain_grain

    def _grain_for_set_operation(self, scope: Scope, node: exp.Union) -> Grain:
        if self._is_in_recursive_cte(scope):
            return Grain(degraded=True, degraded_reason="recursive CTE body")

        left_grain = self._grain_by_expr.get(id(node.left))
        right_grain = self._grain_by_expr.get(id(node.right))
        if left_grain is None or right_grain is None:
            return Grain(evidence=["set operation branch grain unavailable"])

        if left_grain.degraded or right_grain.degraded:
            return Grain(
                degraded=True,
                degraded_reason="a set-operation branch is itself degraded",
            )

        distinct = bool(node.args.get("distinct"))
        cols = tuple(self._output_column_names(node))
        if not distinct:
            return Grain(
                evidence=[
                    f"{node.key.upper()} ALL keeps duplicate rows from the branches: grain unknown"
                ]
            )

        # Deduplicating set op: the full projection is unique, but only when
        # every branch's column types are comparable (we cannot verify that
        # here, so the full projection is the only defensible key).
        if cols:
            return Grain(
                unique_keys=[cols],
                evidence=[f"{node.key.upper()} (without ALL) deduplicates the full projection"],
            )
        return Grain(evidence=["set operation with unresolvable projection: grain unknown"])

    # ---------------------------------------------------------- join chain
    def _analyze_join_chain(self, scope: Scope, node: exp.Select) -> Grain:
        from_ = node.args.get("from_")
        if from_ is None or from_.this is None:
            return Grain(evidence=["no FROM clause: single-row projection"])

        # Grain of the FROM source.
        current = self._grain_of_source(scope, from_.this)
        left_label = relation_label(from_.this)

        for join in node.args.get("joins") or []:
            current = self._analyze_join(scope, node, join, current, left_label)
            right_node = join.this
            left_label = f"{left_label} ⨝ {relation_label(right_node)}"

        return current

    def _analyze_join(self, scope: Scope, select: exp.Select, join: exp.Join,
                      left_grain: Grain, left_label: str) -> Grain:
        right_node = join.this
        right_label = relation_label(right_node)
        join_kind = (join.kind or "INNER").upper() or "INNER"
        if join.method:
            join_kind = f"{join.method.upper()} {join_kind}"
        if join.side:
            join_kind = f"{join.side.upper()} {join_kind}"

        on = join.args.get("on")
        on_sql = on.sql() if on is not None else ""
        equalities = render_equalities(on)
        used = join_on_columns(on)

        right_grain = self._grain_of_source(scope, right_node)
        right_alias = self._source_alias(right_node)
        right_join_cols = tuple(sorted(used.get((right_alias or "").lower(), set())))
        fk_note = self._foreign_key_note(right_node, right_join_cols)

        verdict, reason = self._judge_join(
            join=join,
            on=on,
            right_grain=right_grain,
            right_label=right_label,
            right_join_cols=right_join_cols,
            fk_note=fk_note,
        )
        if fk_note:
            reason = f"{reason} | {fk_note}"

        self.report.verdicts.append(
            JoinVerdict(
                scope_kind=scope.scope_type.name,
                left_relation=left_label,
                right_relation=right_label,
                right_node_sql=right_node.sql(),
                join_kind=join_kind,
                on_sql=on_sql,
                verdict=verdict,
                reason=reason,
                equality_columns=equalities,
                right_proven_keys=list(right_grain.unique_keys),
                location=self._location_of(select),
            )
        )

        # Chain grain after this join: the left grain survives only when the
        # right side cannot multiply rows.
        if verdict == SAFE:
            return Grain(
                unique_keys=list(left_grain.unique_keys),
                evidence=list(left_grain.evidence)
                + [f"join to {right_label} proved non-expanding"],
            )
        return Grain(
            evidence=[
                f"join to {right_label} may multiply rows: downstream grain unknown"
            ],
        )

    def _foreign_key_note(self, right_node: exp.Expr,
                          right_join_cols: tuple[str, ...]) -> str | None:
        if not isinstance(right_node, exp.Table):
            return None
        meta = self.schema.table(right_node.name)
        if meta is None:
            return None
        cols = set(right_join_cols)
        for fk in meta.foreign_keys:
            if set(fk.key()) <= cols:
                return (
                    f"declared FK ({', '.join(fk.columns)}) -> "
                    f"{fk.ref_table}({', '.join(fk.ref_columns)}): this is the "
                    "many side of a relationship, which does not make it unique"
                )
        return None

    def _judge_join(self, join: exp.Join, on: exp.Expr | None, right_grain: Grain,
                    right_label: str, right_join_cols: tuple[str, ...],
                    fk_note: str | None = None) -> tuple[str, str]:
        method = (join.method or "").upper()
        if method == "CROSS" or (on is None and not join.args.get("using")):
            return (
                RISK,
                "cartesian-style join with no equality condition: every left row "
                "pairs with every right row",
            )

        if right_grain.degraded:
            return (
                UNKNOWN,
                f"grain of {right_label} is unknown ({right_grain.degraded_reason}); "
                "cannot prove the join non-expanding",
            )

        if not right_join_cols:
            return (
                UNKNOWN,
                f"no column equality references {right_label}; the join condition "
                "is not an equi-join we can reason about",
            )

        if right_grain.covers(right_join_cols):
            keys = [
                key for key in right_grain.unique_keys
                if set(key) <= set(right_join_cols)
            ]
            pretty = ", ".join("(" + ", ".join(k) + ")" for k in keys)
            return (
                SAFE,
                f"join columns ({', '.join(right_join_cols)}) cover the proven "
                f"unique key {pretty} of {right_label}: at most one matching row",
            )

        if right_grain.unique_keys:
            known = ", ".join("(" + ", ".join(k) + ")" for k in right_grain.unique_keys)
            return (
                RISK,
                f"join columns ({', '.join(right_join_cols)}) do not cover any "
                f"declared unique key of {right_label} (known keys: {known}): "
                "multiple right rows per left row are possible",
            )

        return (
            UNKNOWN,
            f"no unique key is declared or derivable for {right_label}; "
            "uniqueness of the join columns cannot be established",
        )

    # --------------------------------------------------------- source grains
    def _grain_of_source(self, scope: Scope, node: exp.Expr) -> Grain:
        if isinstance(node, exp.Table):
            resolved = self._resolve_table_source(scope, node)
            if isinstance(resolved, Scope):
                grain = self._grain_by_expr.get(id(resolved.expression))
                if grain is not None:
                    return grain
                return Grain(degraded=True, degraded_reason="CTE not analyzed yet")
            return self._grain_of_table(node)
        if isinstance(node, exp.Subquery):
            inner = node.this
            grain = self._grain_by_expr.get(id(inner))
            if grain is None:
                return Grain(degraded=True, degraded_reason="subquery not analyzed")
            return self._remap_derived_grain(grain, node)
        # VALUES / UDTF / etc.
        return Grain(degraded=True, degraded_reason=f"unsupported source {type(node).__name__}")

    def _resolve_table_source(self, scope: Scope, node: exp.Table):
        entry = scope.selected_sources.get(node.alias_or_name)
        if entry is not None:
            return entry[1]
        return None

    def _grain_of_table(self, table: exp.Table) -> Grain:
        name = table.name.lower()
        meta = self.schema.table(name)
        if meta is None:
            # Could be a CTE reference: the scope machinery resolves those to
            # Table nodes too, so look up the CTE grain by name.
            return Grain(
                degraded=True,
                degraded_reason=f"no metadata for table '{name}'",
            )
        keys = sorted(meta.normalized_unique_keys())
        evidence = [
            f"declared unique key ({', '.join(k)}) on {name}" for k in keys
        ]
        return Grain(unique_keys=keys, evidence=evidence)

    def _remap_derived_grain(self, grain: Grain, subquery: exp.Subquery) -> Grain:
        inner = subquery.this
        alias_cols = [c.lower() for c in subquery.alias_column_names]
        if isinstance(inner, exp.Select):
            outputs = [o.lower() for o in inner.named_selects]
        elif isinstance(inner, exp.Union):
            outputs = [o.lower() for o in self._output_column_names(inner)]
        else:
            outputs = []
        if alias_cols and len(alias_cols) == len(outputs):
            mapping = dict(zip(outputs, alias_cols))
        else:
            mapping = {o: o for o in outputs}
        return grain.rename_outputs(mapping)

    # ------------------------------------------------------- small utilities
    def _source_alias(self, node: exp.Expr) -> str | None:
        if isinstance(node, exp.Table):
            return node.alias_or_name
        if isinstance(node, exp.Subquery):
            return node.alias_or_name
        return None

    def _output_column_names(self, query: exp.Query) -> list[str]:
        if isinstance(query, exp.Select):
            return list(query.named_selects)
        if isinstance(query, exp.Union):
            return self._output_column_names(query.left)
        return []

    def _group_key_output_names(self, group: exp.Group, select: exp.Select) -> list[str]:
        """Output names of plain-column GROUP BY items.

        Only items that are plain columns contribute to a provable grain;
        expressions, ordinals and rollups yield no proof (conservative).
        """
        names: list[str] = []
        alias_by_column: dict[str, str] = {}
        for projection in select.expressions:
            if isinstance(projection, exp.Alias) and isinstance(projection.this, exp.Column):
                col = projection.this
                if col.table:
                    alias_by_column[f"{col.table.lower()}.{col.name.lower()}"] = projection.alias.lower()
                else:
                    alias_by_column[col.name.lower()] = projection.alias.lower()
            elif isinstance(projection, exp.Column):
                key = projection.name.lower()
                alias_by_column.setdefault(key, key)

        for item in group.expressions:
            if not isinstance(item, exp.Column):
                return []
            if item.table:
                key = f"{item.table.lower()}.{item.name.lower()}"
            else:
                key = item.name.lower()
            names.append(alias_by_column.get(key, item.name.lower()))
        return names

    def _looks_correlated(self, scope: Scope) -> bool:
        """Qualified column that belongs to an ancestor scope => correlated."""
        local = {name.lower() for name in scope.selected_sources}
        ancestor_names: set[str] = set()
        parent = scope.parent
        while parent is not None:
            ancestor_names.update(name.lower() for name in parent.selected_sources)
            parent = parent.parent
        for column in scope.columns:
            table = (column.table or "").lower()
            if table and table not in local and table in ancestor_names:
                return True
        return False

    def _find_recursive_cte(self, expression: exp.Expr) -> str | None:
        for with_ in expression.find_all(exp.With):
            if with_.args.get("recursive"):
                cte = with_.find(exp.CTE)
                return cte.alias if cte is not None else "?"
        return None

    def _is_in_recursive_cte(self, scope: Scope) -> bool:
        node: exp.Expr | None = scope.expression
        while node is not None:
            if isinstance(node, exp.CTE):
                with_ = node.parent
                if isinstance(with_, exp.With) and with_.args.get("recursive"):
                    return True
            node = node.parent
        return False

    def _location_of(self, select: exp.Select) -> str:
        meta = getattr(select, "meta", {}) or {}
        line = meta.get("line")
        col = meta.get("col")
        if line is not None:
            return f"line {line}, col {col}"
        return ""
