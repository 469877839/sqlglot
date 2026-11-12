"""Result comparison for differential testing.

Without ORDER BY results are compared as multisets (bag semantics): row
order is irrelevant but duplicate multiplicities must match. With ORDER BY
the row sequence is compared element-wise, preserving sequence semantics.
"""

from __future__ import annotations

import math
import struct
import typing as t
from collections import Counter

Row = tuple
Rows = list


def normalize_value(value: t.Any) -> t.Any:
    """Map a SQLite result value to a canonical, hashable form."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return ("float", "nan")
        if value == int(value) and abs(value) < 2**62:
            # SQLite may surface integral floats (e.g. AVG of ints) where the
            # other side produced an int; treat them as equal numerically.
            return int(value)
        return ("float", struct.pack("<d", value))
    if isinstance(value, bytes):
        return ("bytes", value)
    return ("text", str(value))


def normalize_row(row: t.Iterable) -> Row:
    return tuple(normalize_value(v) for v in row)


def rows_equal(expected: Rows, actual: Rows, ordered: bool) -> bool:
    """Compare two result sets.

    Args:
        expected: rows from the original query.
        actual: rows from the optimized query.
        ordered: True when the query has a top-level ORDER BY; in that case
            the comparison is sequence-sensitive. Otherwise rows are compared
            as a multiset.
    """
    left = [normalize_row(r) for r in expected]
    right = [normalize_row(r) for r in actual]
    if ordered:
        return left == right
    return Counter(left) == Counter(right)
