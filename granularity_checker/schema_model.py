"""User-supplied metadata: columns, unique keys, foreign keys."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ForeignKey:
    columns: tuple[str, ...]
    ref_table: str
    ref_columns: tuple[str, ...]

    def key(self) -> tuple[str, ...]:
        return tuple(c.lower() for c in self.columns)


@dataclass
class TableMeta:
    name: str
    columns: tuple[str, ...]
    # Each unique key is a tuple of column names. The primary key should be
    # declared here too. A single-column list means one key per row.
    unique_keys: tuple[tuple[str, ...], ...] = ()
    foreign_keys: tuple[ForeignKey, ...] = ()

    def normalized_unique_keys(self) -> set[tuple[str, ...]]:
        return {tuple(c.lower() for c in key) for key in self.unique_keys}


@dataclass
class DeclaredSchema:
    tables: dict[str, TableMeta] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict) -> "DeclaredSchema":
        tables: dict[str, TableMeta] = {}
        for name, spec in raw.items():
            fks = tuple(
                ForeignKey(
                    columns=tuple(fk["columns"]),
                    ref_table=fk["ref_table"],
                    ref_columns=tuple(fk["ref_columns"]),
                )
                for fk in spec.get("foreign_keys", [])
            )
            tables[name] = TableMeta(
                name=name,
                columns=tuple(spec.get("columns", [])),
                unique_keys=tuple(tuple(key) for key in spec.get("unique_keys", [])),
                foreign_keys=fks,
            )
        return cls(tables=tables)

    def table(self, name: str) -> TableMeta | None:
        if not name:
            return None
        return self.tables.get(name.lower())
