from __future__ import annotations

import typing as t
from dataclasses import dataclass, field
from enum import auto

from sqlglot import expressions as exp
from sqlglot.errors import UnsupportedError
from sqlglot.helper import AutoName


class LossKind(AutoName):
    """The nature of the semantic loss introduced by a transformation."""

    PRECISION = auto()
    """The value's fidelity is reduced, e.g. a timestamp precision is truncated."""

    TIMEZONE = auto()
    """Time-zone awareness is dropped, e.g. `TIMESTAMPTZ` becomes `TIMESTAMP`."""

    NULL_ORDERING = auto()
    """A `NULLS FIRST`/`NULLS LAST` clause cannot be represented and is dropped."""

    INDEXING = auto()
    """An array position cannot be rewritten to the target's indexing convention."""

    SET_OPERATION = auto()
    """A set operation attribute (e.g. `ALL`) cannot be represented in the target."""


class Category(AutoName):
    """The transformation area a diagnostic belongs to."""

    NULL_ORDERING = auto()
    """Ordering of null values, e.g. `ORDER BY x NULLS LAST`."""

    TEMPORAL = auto()
    """Temporal precision and time-zone semantics, e.g. `TIMESTAMP(9)`, `TIMESTAMPTZ`."""

    ARRAY_INDEXING = auto()
    """Array subscript/position conventions, e.g. 1-based vs 0-based indexes."""

    SET_OPERATION = auto()
    """Set operation capabilities, e.g. `INTERSECT ALL` support differences."""


class ConversionPolicy(AutoName):
    """How a caller wants lossy transformations to be handled."""

    ALLOW = auto()
    """Generate the lossy SQL silently: neither diagnostics nor warnings are surfaced."""

    COLLECT = auto()
    """Generate the SQL and surface structured diagnostics (warnings are logged)."""

    REJECT = auto()
    """Raise `LossyConversionError` when a lossy transformation is required."""


@dataclass(frozen=True)
class ConversionDiagnostic:
    """A structured record of one lossy transformation.

    Attributes:
        category: the transformation area, see `Category`.
        node: the source AST node that the transformation applies to. If the node
            was produced in the middle of a multi-step transformation (e.g. an
            `exp.DataType` produced by a type-mapping transform), this is resolved
            to the originating node in the tree that was handed to the generator,
            using the node's provenance id. `None` if no provenance node exists.
        node_sql: the SQL of the source node, rendered with the generic dialect so
            it is independent of both the source and target dialects.
        target_dialect: the dialect being generated for.
        action: a human-readable description of what the generator did.
        loss: the nature of the lost semantics.
        message: the warning text that accompanies the diagnostic.
        source_id: the provenance id shared by every node derived from the same
            source node, which lets callers follow one node across multiple
            transformation steps.
    """

    category: Category
    node: exp.Expr | None
    node_sql: str
    target_dialect: str
    action: str
    loss: LossKind
    message: str
    source_id: str | None = None

    def to_dict(self) -> dict[str, t.Any]:
        return {
            "category": self.category.value,
            "node_sql": self.node_sql,
            "node_type": type(self.node).__name__ if self.node is not None else None,
            "target_dialect": self.target_dialect,
            "action": self.action,
            "loss": self.loss.value,
            "message": self.message,
            "source_id": self.source_id,
        }


@dataclass
class DiagnosticCollection:
    """The result of one generator run: the SQL and the diagnostics it produced."""

    sql: str
    diagnostics: list[ConversionDiagnostic] = field(default_factory=list)


class LossyConversionError(UnsupportedError):
    """Raised when a lossy conversion is performed under `ConversionPolicy.REJECT`."""

    def __init__(self, diagnostics: list[ConversionDiagnostic]) -> None:
        self.diagnostics = diagnostics
        super().__init__("\n\n".join(d.message for d in diagnostics))


SOURCE_ID_META_KEY = "sqlglot_source_id"
"""Expression meta key holding a node's provenance id."""


def assign_source_ids(expression: exp.Expr) -> exp.Expr:
    """Tag every node of `expression` with a stable provenance id and return it.

    Used internally by `Generator.generate` on the working copy, so the caller's AST
    is never mutated.
    """
    for i, node in enumerate(expression.walk(bfs=False)):
        node.meta[SOURCE_ID_META_KEY] = f"n{i}"
    return expression


class SourceResolver:
    """Maps nodes of a generating (possibly copied/transformed) tree back to source nodes.

    The target tree is a provenance-tagged copy of the source tree; a parallel walk
    pairs every target node (by provenance id) with the originating source node.
    Nodes introduced by transforms have no tag and resolve through their nearest
    tagged ancestor.
    """

    __slots__ = ("_source_nodes",)

    def __init__(self, source: exp.Expr, target: exp.Expr) -> None:
        self._source_nodes: dict[str, exp.Expr] = {}
        self._build_source_map(source, target)

    @staticmethod
    def _children(node: exp.Expr) -> list[exp.Expr]:
        children = []
        for value in node.args.values():
            if isinstance(value, list):
                children.extend(v for v in value if isinstance(v, exp.Expr))
            elif isinstance(value, exp.Expr):
                children.append(value)
        return children

    def _build_source_map(self, source: exp.Expr, target: exp.Expr) -> None:
        # A depth-first parallel walk pairs corresponding nodes of the source tree
        # and its tagged copy. Nodes inserted by transforms have no counterpart and
        # are skipped; they resolve through their nearest tagged ancestor instead.
        stack = [(source, target)]
        while stack:
            source_node, target_node = stack.pop()

            if isinstance(source_node, exp.Expr) and isinstance(target_node, exp.Expr):
                source_id = target_node.meta_get(SOURCE_ID_META_KEY)
                if source_id is not None:
                    self._source_nodes[source_id] = source_node

                source_children = self._children(source_node)
                target_children = self._children(target_node)
                if len(source_children) == len(target_children):
                    stack.extend(zip(source_children, target_children))

    def _nearest_id(self, node: exp.Expr | None) -> str | None:
        """Returns the provenance id of `node`, inherited from its nearest tagged ancestor."""
        current = node
        while current is not None:
            source_id = current.meta_get(SOURCE_ID_META_KEY)
            if source_id is not None:
                return t.cast(str, source_id)
            current = current.parent
        return None

    def resolve(self, node: exp.Expr | None) -> exp.Expr | None:
        """Returns the originating source node for `node`, or `None` if untraceable."""
        source_id = self._nearest_id(node)
        if source_id is None:
            return None
        return self._source_nodes.get(source_id)

    def source_id(self, node: exp.Expr | None) -> str | None:
        """Returns the provenance id of `node`, inherited from its nearest tagged ancestor."""
        return self._nearest_id(node)
