"""Query fingerprinting: group SQL statements into families by structure and binding.

Two statements share a fingerprint when they differ only in

- constant values (replaced by typed placeholders, e.g. ``:num`` / ``:str``),
- the spelling of local aliases (table / CTE / derived-table aliases that are
  bound to the same source at the same structural position),
- whitespace, comments and identifier case (per the dialect's rules).

Real identities are preserved: table names, column names, projection aliases,
join order, comparison operand order and projection order all stay in the
normalized form, so the output is readable enough to explain why statements
were grouped together.

Resolution is scope-aware (via ``sqlglot.optimizer.scope``), which handles CTE
shadowing, correlated subqueries and self-joins. When a binding cannot be
determined (e.g. an unqualified column with several candidate sources and no
schema), the statement is kept conservative: nothing is rewritten that could
merge two queries whose bindings might differ.
"""

from __future__ import annotations

import hashlib
import typing as t
from dataclasses import dataclass

from sqlglot import exp, parse_one
from sqlglot.dialects.dialect import Dialect, DialectType
from sqlglot.helper import find_new_name
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
from sqlglot.optimizer.qualify_tables import qualify_tables
from sqlglot.optimizer.scope import Scope, traverse_scope

__all__ = ["Fingerprint", "fingerprint", "same_family"]

# DML statements whose target / relation tables can carry an alias and whose
# columns may qualify through it. Other DML statements (e.g. COPY) do not
# alias their target and must be left alone.
_ALIASABLE_DML = (exp.Insert, exp.Update, exp.Delete, exp.Merge)


@dataclass(frozen=True)
class Fingerprint:
    """The result of fingerprinting one SQL statement.

    Attributes:
        normalized: Canonical SQL. Constants appear as typed placeholders
            (``:num``, ``:str``, ``:bool``, ``:param``) and local aliases are
            renamed to positional names (``_0``, ``_1``, ... for sources,
            ``_cte0``, ... for CTEs). Table and column identities are kept.
        digest: Stable hex digest of ``normalized``, for use as a map key.
        dialect: The dialect the statement was parsed with. Only compare
            fingerprints produced for the same dialect.
    """

    normalized: str
    digest: str
    dialect: str


def fingerprint(sql: str, dialect: DialectType = None) -> Fingerprint:
    """Compute the fingerprint of a single SQL statement.

    Args:
        sql: The SQL statement to fingerprint.
        dialect: The dialect to parse and render with. Fingerprints are only
            comparable within the same dialect.

    Returns:
        A ``Fingerprint`` with the normalized form and its digest.
    """
    dialect_obj = Dialect.get_or_raise(dialect)
    expression = parse_one(sql, dialect=dialect_obj)
    normalized = _normalize(expression, dialect_obj)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return Fingerprint(normalized=normalized, digest=digest, dialect=dialect_obj.__class__.__name__)


def same_family(left: str, right: str, dialect: DialectType = None) -> bool:
    """Whether two statements belong to the same query family under ``dialect``."""
    return fingerprint(left, dialect=dialect).digest == fingerprint(right, dialect=dialect).digest


def _normalize(expression: exp.Expression, dialect: Dialect) -> str:
    expression = normalize_identifiers(expression, dialect=dialect)

    # Structural normalization only (db/catalog qualification, join-construct
    # expansion); aliasing is handled below so CTE references are covered too.
    expression = qualify_tables(expression, dialect=dialect)

    scopes = traverse_scope(expression)
    if scopes:
        _rename_ctes(expression, scopes)
    if scopes or isinstance(expression, _ALIASABLE_DML):
        # A DML statement without a query body yields no scopes, but its target
        # and relation tables still need canonical aliases.
        _rename_sources(expression, scopes)

    _replace_constants(expression)
    _strip_comments(expression)

    # identify=True quotes every identifier, so the canonical form is
    # independent of how the input was quoted and always re-parseable.
    return expression.sql(dialect=dialect, identify=True)


def _rename_ctes(expression: exp.Expression, scopes: list[Scope]) -> None:
    """Rename every CTE to ``_cte<N>`` by structural position and update references.

    Numbering follows AST pre-order, which respects WITH-clause declaration
    order, so two queries whose CTEs bind the same way get the same names
    regardless of how the CTEs were originally called. References are found
    through scope sources, so a local alias that shadows a CTE name is left
    alone and a reference to a shadowed outer CTE is still updated. CTEs
    hanging off a DML/DDL statement (whose scopes are not one tree) are
    covered by enumerating the AST rather than any single scope root.
    """
    taken = {identifier.name for identifier in expression.find_all(exp.Identifier)}
    next_cte_name = _collision_free_name(taken, "_cte")
    new_name_by_id: dict[int, str] = {}

    for cte in expression.find_all(exp.CTE):
        new_name = next_cte_name()
        new_name_by_id[id(cte)] = new_name
        cte.args["alias"].set("this", exp.to_identifier(new_name))

    for scope in _preorder_scopes(scopes):
        for old_name, cte_source in scope.cte_sources.items():
            # The source expression sits inside the CTE's own subtree (for a
            # recursive CTE it is the base term of the union), so the nearest
            # CTE ancestor identifies the definition.
            cte_node = cte_source.expression.find_ancestor(exp.CTE)
            new_name = new_name_by_id.get(id(cte_node)) if cte_node else None
            if new_name is None:
                continue
            for table in scope.tables:
                # Match on the actual binding, not just the spelling: a real
                # table with the same name as an in-scope CTE is shadowing.
                if (
                    isinstance(table.parent, (exp.From, exp.Join))
                    and not table.db
                    and not table.catalog
                    and table.alias_or_name in scope.sources
                    and scope.sources[table.alias_or_name] is cte_source
                ):
                    table.set("this", exp.to_identifier(new_name))


def _rename_sources(expression: exp.Expression, scopes: list[Scope]) -> None:
    """Rename local source aliases to ``_0``, ``_1``, ... by structural position.

    Every source visible in a scope gets a canonical name: real tables, CTE
    references (with or without a local alias), derived tables and UDTFs.
    Column qualifiers that resolve to a renamed source are rewritten to the
    canonical name, which is what makes binding-equivalent aliases compare
    equal while different bindings (e.g. swapped self-join sides) stay apart.
    DML relation tables (``UPDATE ... FROM``, ``DELETE ... USING``,
    ``MERGE ... USING``) are standalone scopes in a DML statement and handled
    together with their statement's columns.
    """
    taken = {identifier.name for identifier in expression.find_all(exp.Identifier)}
    next_source_name = _collision_free_name(taken, "_")
    canonical: dict[str, str] = {}
    dml_statements = list(expression.find_all(*_ALIASABLE_DML))

    for scope in _preorder_scopes(scopes):
        if isinstance(scope.expression, exp.Table):
            # Standalone DML relation scope; its tables are renamed by the
            # DML pass below together with the statement's other relations.
            continue

        for table in scope.tables:
            if not isinstance(table.parent, (exp.From, exp.Join)):
                continue
            _assign(table, table.alias or table.name, canonical, next_source_name)

        for derived_table in scope.derived_tables:
            _assign(derived_table, derived_table.alias, canonical, next_source_name)

        for udtf in scope.udtfs:
            _assign(udtf, udtf.alias_or_name, canonical, next_source_name)

        for column in scope.columns:
            qualifier = column.table
            if qualifier and (new_name := canonical.get(qualifier)):
                column.set("table", exp.to_identifier(new_name))

    for statement in dml_statements:
        relations = list(_dml_relations(statement))
        for table in relations:
            _assign(table, table.alias or table.name, canonical, next_source_name)
        for column in statement.find_all(exp.Column):
            qualifier = column.table
            if qualifier and (new_name := canonical.get(qualifier)):
                column.set("table", exp.to_identifier(new_name))


def _dml_relations(statement: exp.DML) -> t.Iterator[exp.Table]:
    """Target and relation tables of a DML statement, in declaration order."""
    target = statement.this
    if isinstance(target, exp.Table):
        yield target
    elif isinstance(target, exp.Expression):
        yield from target.find_all(exp.Table)

    from_ = statement.args.get("from_")
    if isinstance(from_, exp.From):
        yield from _with_joined_tables(from_.this)

    using = statement.args.get("using")
    for relation in using if isinstance(using, list) else [using]:
        if isinstance(relation, exp.Expression):
            yield from _with_joined_tables(relation)


def _with_joined_tables(node: exp.Expression) -> t.Iterator[exp.Table]:
    if isinstance(node, exp.Table):
        yield node
        for join in node.args.get("joins") or []:
            yield from _with_joined_tables(join.this)
    else:
        yield from node.find_all(exp.Table)


def _assign(
    source: exp.Expression,
    current_name: str,
    canonical: dict[str, str],
    next_name: t.Callable[[], str],
) -> None:
    new_name = canonical.get(current_name)
    if new_name is None:
        new_name = next_name()
        canonical[current_name] = new_name

    alias = source.args.get("alias")
    if alias is None:
        source.set("alias", exp.TableAlias(this=exp.to_identifier(new_name)))
    else:
        alias.set("this", exp.to_identifier(new_name))


def _replace_constants(expression: exp.Expression) -> None:
    """Replace literal values and parameter markers with typed placeholders."""
    for node in list(expression.walk()):
        replacement: exp.Expression | None = None

        if isinstance(node, exp.Literal):
            if _is_structural_key(node):
                continue
            replacement = exp.Placeholder(this="num" if node.is_number else "str")
        elif isinstance(node, exp.Boolean):
            replacement = exp.Placeholder(this="bool")
        elif isinstance(node, (exp.Placeholder, exp.Parameter, exp.SessionParameter)):
            replacement = exp.Placeholder(this="param")

        if replacement is not None:
            node.replace(replacement)


def _is_structural_key(literal: exp.Literal) -> bool:
    """Whether a string literal names something instead of supplying a value.

    Property keys (``TBLPROPERTIES ('k' = ...)``) and JSON object keys are
    identifiers rendered from literals: placeholder-izing them changes the
    statement's structure, so they keep their spelling.
    """
    if not literal.is_string:
        return False
    parent = literal.parent
    return isinstance(parent, (exp.Property, exp.JSONKeyValue)) and parent.this is literal


def _strip_comments(expression: exp.Expression) -> None:
    """Drop source comments, which carry no query-family information."""
    for node in expression.walk():
        node.comments = None


def _preorder_scopes(scopes: list[Scope]) -> t.Iterator[Scope]:
    """Yield all scopes in depth-first pre-order, including disconnected roots.

    A DML/DDL statement yields several scopes that do not share a single
    yielded root, so iterate every scope whose parent is not in the set and
    walk its subtree in source order.
    """
    seen = set(scopes)
    roots = [scope for scope in scopes if scope.parent not in seen]

    def _walk(scope: Scope) -> t.Iterator[Scope]:
        yield scope
        for child in (
            *scope.cte_scopes,
            *scope.set_operation_scopes,
            *scope.table_scopes,
            *scope.subquery_scopes,
        ):
            yield from _walk(child)

    for root in roots:
        yield from _walk(root)


def _collision_free_name(taken: set[str], prefix: str) -> t.Callable[[], str]:
    """Name generator that never produces an identifier present in ``taken``."""
    counter = 0

    def _next() -> str:
        nonlocal counter
        name = find_new_name(taken, f"{prefix}{counter}")
        counter += 1
        taken.add(name)
        return name

    return _next
