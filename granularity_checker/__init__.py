"""Conservative SQL granularity / join fan-out checker built on sqlglot."""

from .checker import (
    CheckReport,
    DeclaredSchema,
    JoinVerdict,
    analyze,
    analyze_text,
)

__all__ = [
    "CheckReport",
    "DeclaredSchema",
    "JoinVerdict",
    "analyze",
    "analyze_text",
]
