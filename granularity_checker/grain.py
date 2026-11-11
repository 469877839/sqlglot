"""Proven row-grain of a source (base table or derived relation).

The checker only reasons about what it can prove from the SQL text plus the
declared schema. Everything is deliberately conservative: a grain is a set of
column sets known to be unique on every row emitted by the source. When
nothing can be proved, ``unique_keys`` is empty (unknown, not "definitely
duplicated").
"""

from __future__ import annotations

from dataclasses import dataclass, field

# A "grain" maps a key name space ("output" columns of the relation) to
# column sets that are known unique. Columns are lower-cased.


@dataclass
class Grain:
    # Proven unique column combinations, expressed in the relation's own
    # output column names. Empty list = no guarantee proved.
    unique_keys: list[tuple[str, ...]] = field(default_factory=list)
    # Human-readable provenance notes, e.g. "GROUP BY (order_id)".
    evidence: list[str] = field(default_factory=list)
    # True when the relation's whole structure is unsupported (correlation,
    # recursion, ...): no grain can be assumed.
    degraded: bool = False
    degraded_reason: str | None = None

    @property
    def has_proof(self) -> bool:
        return bool(self.unique_keys) and not self.degraded

    def covers(self, columns: tuple[str, ...]) -> bool:
        """True if *columns* contain some proven unique key."""
        wanted = {c.lower() for c in columns}
        for key in self.unique_keys:
            if set(key) <= wanted:
                return True
        return False

    def rename_outputs(self, mapping: dict[str, str]) -> "Grain":
        """Map grain columns through an output-alias mapping.

        Keys that reference columns not present in the mapping are dropped:
        we would rather lose a proof than claim uniqueness on the wrong name.
        """
        mapped: list[tuple[str, ...]] = []
        for key in self.unique_keys:
            if all(c.lower() in mapping for c in key):
                mapped.append(tuple(mapping[c.lower()] for c in key))
        return Grain(unique_keys=mapped, evidence=list(self.evidence),
                     degraded=self.degraded, degraded_reason=self.degraded_reason)
