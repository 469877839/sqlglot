"""Differential tester for the sqlglot optimizer against SQLite.

Generates small SQLite-executable SELECT statements, runs the original and
the optimized form against the same in-memory database and compares results.
Failures are reduced to a smaller reproducing example.
"""

__version__ = "0.1.0"
