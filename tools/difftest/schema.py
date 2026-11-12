"""Small, typed schemas for the differential generator.

Three tables with deliberately different shapes: an integer key table, a text
table and a nullable value table. Data distributions are chosen to exercise
NULLs, duplicate rows and outer-join padding.
"""

from __future__ import annotations

INT = "INT"
TEXT = "TEXT"

SCHEMAS: dict[str, dict[str, dict[str, str]]] = {
    "small": {
        "t1": {"a": INT, "b": INT},
        "t2": {"id": INT, "a": INT},
        "t3": {"c": INT, "d": TEXT},
    },
    "nullable": {
        "t1": {"a": INT, "b": INT},
        "t2": {"id": INT, "a": INT},
        "t3": {"c": INT, "d": TEXT},
    },
}

# Values per column (ordered, so the seed fully determines the data).
COLUMN_VALUES: dict[str, dict[str, list]] = {
    "small": {
        "t1.a": [0, 1, 1, 2, 3],
        "t1.b": [10, 20, 20, 30, 30],
        "t2.id": [0, 1, 2, 2, 4],
        "t2.a": [1, 1, 2, 0, 2],
        "t3.c": [0, 1, 2, 2, 5],
        "t3.d": ["x", "y", "y", None, "z"],
    },
    "nullable": {
        "t1.a": [None, 0, 1, 1, 2, None, 3],
        "t1.b": [10, None, 20, 30, 30, None, 10],
        "t2.id": [0, 1, 2, 2, 4, None],
        "t2.a": [1, None, 2, 0, 2, 1],
        "t3.c": [None, 0, 1, 2, 2, 5],
        "t3.d": ["x", None, "y", "y", None, "z"],
    },
}

TABLES = ("t1", "t2", "t3")


def ddl_for(schema: dict[str, dict[str, str]]) -> list[str]:
    return [
        f"CREATE TABLE {table} ({', '.join(f'{col} {type_}' for col, type_ in cols.items())})"
        for table, cols in schema.items()
    ]


def inserts_for(profile: str, schema: dict[str, dict[str, str]]) -> list[str]:
    values = COLUMN_VALUES[profile]
    statements = []
    for table, cols in schema.items():
        rows = list(zip(*(values[f"{table}.{col}"] for col in cols)))
        literals = ", ".join(
            "(" + ", ".join(_literal(value) for value in row) + ")" for row in rows
        )
        statements.append(f"INSERT INTO {table} VALUES {literals}")
    return statements


def _literal(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    return str(value)


def columns_for(schema: dict[str, dict[str, str]]):
    for table, cols in schema.items():
        for col, type_ in cols.items():
            yield table, col, type_


def qualified_columns(schema: dict[str, dict[str, str]], type_: str | None = None):
    return [
        (table, col, ctype)
        for table, col, ctype in columns_for(schema)
        if type_ is None or ctype == type_
    ]
