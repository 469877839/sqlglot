# Conservative SQL granularity checker (SQLGlot-based)

Answers one narrow question about a query: **before the aggregate expressions
of each query block run, can a join multiply the rows of the relations already
joined?**

The checker never guesses. A join is only cleared when uniqueness of the
right-hand relation on the join columns can be *proved* from:

1. unique keys declared in the supplied schema, or
2. the SQL structure itself (`GROUP BY`, `DISTINCT`, deduplicating set
   operations).

## Verdicts

| Verdict   | Meaning                                                                          |
|-----------|----------------------------------------------------------------------------------|
| `SAFE`    | Right side is provably unique on the join columns; at most one row per left row. |
| `RISK`    | The right side's grain is known and the join columns cover none of its keys.      |
| `UNKNOWN` | Not enough evidence (missing metadata, non-equi join, unsupported construct).     |

`SAFE` is a proof about **row multiplicity only**. It does not mean the
business metric is correct — filters, expressions, join semantics and data
quality are out of scope.

## Inputs

```python
from granularity_checker import DeclaredSchema, analyze_text

schema = DeclaredSchema.from_dict({
    "orders": {
        "columns": ["order_id", "customer_id", "amount"],
        "unique_keys": [["order_id"]],
        "foreign_keys": [
            {"columns": ["customer_id"], "ref_table": "customers",
             "ref_columns": ["customer_id"]}
        ],
    },
    "order_items": {
        "columns": ["item_id", "order_id", "qty"],
        "unique_keys": [["item_id"]],
        "foreign_keys": [
            {"columns": ["order_id"], "ref_table": "orders",
             "ref_columns": ["order_id"]}
        ],
    },
})

report = analyze_text(
    "SELECT sum(o.amount) FROM orders o "
    "JOIN order_items i ON i.order_id = o.order_id",
    schema,
)
print(report.format_text())
```

Foreign keys do **not** make a relation unique: an FK identifies the *many*
side, so it is surfaced as supporting evidence but never as a proof of
non-expansion.

## What is tracked

- Table and relation aliases, including explicit derived-table column aliases.
- CTEs (non-recursive), derived tables and nested subqueries, bottom-up.
- `GROUP BY` plain columns (grouping columns are unique per output group).
- `DISTINCT` (the full projection becomes unique; joining on a subset of it is
  still `RISK`).
- `UNION / INTERSECT / EXCEPT` without `ALL` (deduplicated full projection) vs
  their `ALL` variants (grain unknown).
- Composite unique keys: join columns must *cover* a full key.
- Join kind, including cartesian joins (`CROSS JOIN`, no `ON`).

## Explicit degradation

The following are out of scope and always produce an explicit `DEGRADED` note
plus `UNKNOWN` for joins that depend on them, rather than a silent guess:

- **Correlated subqueries** — any subquery carrying a qualified reference to an
  ancestor scope (uncorrelated subqueries are analyzed normally).
- **Recursive CTEs** — `WITH RECURSIVE` bodies have no statically provable
  grain.
- `GROUP BY` on expressions or ordinals, non-equi join predicates, and sources
  without metadata similarly yield `UNKNOWN` with a stated reason.

## Demo and tests

```bash
python -m granularity_checker.demo        # paired cases + in-memory SQLite proof
python -m pytest granularity_checker -q   # verdict test suite
```

The SQLite demo seeds two orders (one with two item rows) and shows
`sum(orders.amount)` inflating from `300` to `400` under the risky join, while
the pre-aggregated variant returns `300`.

### Paired cases (one difference flips the verdict)

- A: join on the child unique key vs. its non-unique FK column.
- B: raw child join vs. the same child pre-aggregated in a derived table.
- C: same SQL with vs. without a unique-key declaration.
- D: `UNION` vs. `UNION ALL` inside the derived table.
- E: plain projection vs. `DISTINCT` projection of a keyless log.
