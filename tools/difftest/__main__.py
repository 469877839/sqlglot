"""CLI for the optimizer differential tester.

    python -m tools.difftest run      --seed-start 0 --seeds 500
    python -m tools.difftest replay   counterexample.json
    python -m tools.difftest selftest

Exit code is 0 when no mismatch is found (or the replayed counterexample no
longer reproduces), 1 otherwise, 2 on usage errors.
"""

from __future__ import annotations

import argparse
import collections
import sys
import time

from .gen import generate_case
from .report import case_from_payload, read_counterexample, write_counterexample
from .rules import RULE_SETS, resolve_rules
from .runner import MISMATCH, OK, SKIPPED, Case, Outcome, run_case
from .shrink import shrink_case

PROFILES = ("small", "nullable")


def _resolve_ruleset(name: str):
    if name not in RULE_SETS:
        print(f"error: unknown rule set {name!r}; choices: {', '.join(sorted(RULE_SETS))}")
        raise SystemExit(2)
    return RULE_SETS[name]


def cmd_run(args: argparse.Namespace) -> int:
    rules = _resolve_ruleset(args.rules)
    profiles = (args.profile,) if args.profile else PROFILES
    stats: collections.Counter = collections.Counter()
    skip_reasons: collections.Counter = collections.Counter()
    mismatches: list[tuple[str, int, Case, Outcome]] = []
    started = time.time()

    total = 0
    for profile in profiles:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            total += 1
            case = generate_case(seed, profile)
            outcome = run_case(case, rules)
            stats[outcome.status] += 1
            if outcome.status == SKIPPED:
                skip_reasons[outcome.reason.split(":")[0]] += 1
            elif outcome.status == MISMATCH:
                mismatches.append((profile, seed, case, outcome))
                if args.fail_fast:
                    break
        if args.fail_fast and mismatches:
            break

    elapsed = time.time() - started
    print(
        f"ran {total} cases in {elapsed:.2f}s | "
        f"ok={stats[OK]} mismatch={stats[MISMATCH]} skipped={stats[SKIPPED]}"
    )
    for reason, count in sorted(skip_reasons.items()):
        print(f"  skipped[{reason}] = {count}")

    if not mismatches:
        return 0

    for index, (profile, seed, case, outcome) in enumerate(mismatches[: args.max_reports]):
        print(f"\n[mismatch] profile={profile} seed={seed}")
        print(f"  query     : {case.query}")
        print(f"  optimized : {outcome.optimized_sql}")
        print(f"  original rows  : {outcome.original_rows}")
        print(f"  optimized rows : {outcome.optimized_rows}")
        if args.shrink_budget > 0:
            reduced, spent = shrink_case(
                case,
                is_failure=lambda c: run_case(c, rules).status == MISMATCH,
                budget=args.shrink_budget,
            )
            reduced_outcome = run_case(reduced, rules)
            print(f"  reduced ({spent} evaluations):")
            print(f"    inserts : {reduced.inserts}")
            print(f"    query   : {reduced.query}")
            print(f"    optimized: {reduced_outcome.optimized_sql}")
            if args.out_dir:
                import os

                os.makedirs(args.out_dir, exist_ok=True)
                path = os.path.join(args.out_dir, f"counterexample-{profile}-{seed}.json")
                write_counterexample(
                    path, reduced, reduced_outcome, rules, spent, args.shrink_budget
                )
                print(f"    written : {path}")
    return 1


def cmd_replay(args: argparse.Namespace) -> int:
    payload = read_counterexample(args.file)
    try:
        rules = resolve_rules(payload["rules"])
    except KeyError as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
    case = case_from_payload(payload)
    outcome = run_case(case, rules)
    print(f"replayed {args.file}: status={outcome.status}")
    if outcome.status == MISMATCH:
        print(f"  original rows  : {outcome.original_rows}")
        print(f"  optimized rows : {outcome.optimized_rows}")
        return 1
    if outcome.reason:
        print(f"  reason: {outcome.reason}")
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """Prove harness + shrinker catch deliberately injected faulty rules."""
    failures = []
    for bug_name, rule_set in (
        ("drop_outer_join_BUG", "bug-drop-outer-join"),
        ("drop_distinct_BUG", "bug-drop-distinct"),
        ("invert_is_null_BUG", "bug-invert-is-null"),
    ):
        rules = _resolve_ruleset(rule_set)
        found = None
        for profile in PROFILES:
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                case = generate_case(seed, profile)
                if run_case(case, rules).status == MISMATCH:
                    found = (profile, seed, case)
                    break
            if found:
                break
        if not found:
            failures.append(f"{bug_name}: no mismatch found in scanned seeds")
            print(f"[selftest] {bug_name}: NOT detected")
            continue
        profile, seed, case = found
        before = (len(case.query), sum(len(i) for i in case.inserts))
        reduced, spent = shrink_case(
            case,
            is_failure=lambda c: run_case(c, rules).status == MISMATCH,
            budget=args.shrink_budget,
        )
        after = (len(reduced.query), sum(len(i) for i in reduced.inserts))
        still_fails = run_case(reduced, rules).status == MISMATCH
        print(
            f"[selftest] {bug_name}: detected at profile={profile} seed={seed}; "
            f"shrunk query {before[0]}->{after[0]} chars, data {before[1]}->{after[1]} chars "
            f"in {spent} evaluations; still failing: {still_fails}"
        )
        if not still_fails:
            failures.append(f"{bug_name}: shrinker lost the failure")
        if args.keep_dir:
            import os

            os.makedirs(args.keep_dir, exist_ok=True)
            outcome = run_case(reduced, rules)
            path = os.path.join(args.keep_dir, f"selftest-{bug_name}.json")
            write_counterexample(path, reduced, outcome, rules, spent, args.shrink_budget)
            print(f"           counterexample: {path}")
    if failures:
        for failure in failures:
            print(f"SELFTEST FAILURE: {failure}")
        return 1
    print("selftest passed: injected faults detected and reduced")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tools.difftest", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="generate and execute differential cases")
    run.add_argument("--seed-start", type=int, default=0)
    run.add_argument("--seeds", type=int, default=500, help="seeds per profile")
    run.add_argument("--profile", choices=PROFILES, default=None)
    run.add_argument("--rules", default="default", help=f"rule set: {', '.join(sorted(RULE_SETS))}")
    run.add_argument(
        "--shrink-budget", type=int, default=200,
        help="max reducer evaluations per mismatch (0 disables shrinking)",
    )
    run.add_argument("--max-reports", type=int, default=5)
    run.add_argument("--out-dir", default=None, help="directory for counterexample JSON files")
    run.add_argument("--fail-fast", action="store_true")
    run.set_defaults(func=cmd_run)

    replay = sub.add_parser("replay", help="replay a counterexample file")
    replay.add_argument("file")
    replay.set_defaults(func=cmd_replay)

    selftest = sub.add_parser("selftest", help="verify harness catches injected faulty rules")
    selftest.add_argument("--seed-start", type=int, default=0)
    selftest.add_argument("--seeds", type=int, default=60)
    selftest.add_argument("--shrink-budget", type=int, default=200)
    selftest.add_argument("--keep-dir", default=None, help="keep reduced counterexamples here")
    selftest.set_defaults(func=cmd_selftest)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
