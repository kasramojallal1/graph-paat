"""Run graph-paat over real repositories and record exactly what it produced.

Unit tests use code small enough to know the answer by hand. This does the
opposite: it runs the whole tool over hundreds of thousands of lines of real
Python and writes down every number. On the next run those numbers are compared.

**An unexplained movement is a bug until proven otherwise.** That rule is the
whole point. Every real defect found so far -- a builtin resolved to a
same-named vendored function, an id suffix clashing with a real symbol, edges
counted twice -- showed up first as a count that moved for no stated reason.

Corpora are configured per machine in `corpora.json` (see `corpora.sample.json`),
because they are installed packages rather than fixtures we ship.

    python -m tests.corpus_runner            # compare against baselines
    python -m tests.corpus_runner --record   # accept current numbers as truth
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from graphpaat.ids import Collisions
from graphpaat.parse import parse_corpus_files
from graphpaat.resolve import resolve

HERE = Path(__file__).parent
BASELINES = HERE / "baselines"
CONFIG = HERE / "corpora.json"


def measure(root: Path) -> dict:
    """Every number one build produces. Deliberately exhaustive: a metric that
    is not recorded is a regression that cannot be seen."""
    files, failed = parse_corpus_files(root)

    nodes, edges, collisions = [], [], Collisions()
    for parsed in files:
        for node in parsed.nodes:
            collisions.claim(node.id, f"{node.file}:L{node.line}")
            nodes.append(node)
        edges.extend(parsed.edges)

    call_edges, refusals = resolve(files)
    edges.extend(call_edges)

    resolved = [e for e in call_edges if e.resolved]
    return {
        "files_parsed": len(files),
        "files_failed": len(failed),
        "nodes_total": len(nodes),
        "nodes_by_kind": dict(Counter(n.kind for n in nodes)),
        "edges_total": len(edges),
        "edges_by_relation": dict(Counter(e.relation for e in edges)),
        "edges_unresolved": sum(1 for e in edges if not e.resolved),
        "resolved_by_rule": dict(Counter(e.reason for e in resolved)),
        "refusals_by_reason": dict(refusals),
        "id_collisions": len(collisions.collided()),
        "ids_unique": len({n.id for n in nodes}),
    }


def flatten(data: dict, prefix: str = "") -> dict[str, int]:
    """Nested counts to flat `a.b` keys, so a diff can name exactly what moved."""
    out: dict[str, int] = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{name}."))
        else:
            out[name] = value
    return out


def compare(name: str, current: dict, baseline: dict) -> list[str]:
    now, before = flatten(current), flatten(baseline)
    problems = []
    for key in sorted(set(now) | set(before)):
        a, b = before.get(key), now.get(key)
        if a == b:
            continue
        if a is None:
            problems.append(f"  {name}: NEW  {key} = {b}")
        elif b is None:
            problems.append(f"  {name}: GONE {key} (was {a})")
        else:
            delta = b - a
            problems.append(f"  {name}: {key}  {a} -> {b}  ({delta:+d})")
    return problems


def load_corpora() -> dict[str, Path]:
    if not CONFIG.exists():
        print(f"no {CONFIG.name}; copy corpora.sample.json and set the paths",
              file=sys.stderr)
        raise SystemExit(2)
    entries = json.loads(CONFIG.read_text())
    return {name: Path(path) for name, path in entries.items()}


def main(argv: list[str]) -> int:
    record = "--record" in argv
    only = [a for a in argv if not a.startswith("--")]
    BASELINES.mkdir(exist_ok=True)

    corpora = load_corpora()
    if only:
        corpora = {k: v for k, v in corpora.items() if k in only}

    problems: list[str] = []
    for name, root in sorted(corpora.items()):
        if not root.exists():
            print(f"{name:12s} SKIP  (not installed at {root})")
            continue
        current = measure(root)
        path = BASELINES / f"{name}.json"

        if record or not path.exists():
            path.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n")
            print(f"{name:12s} RECORDED  {current['nodes_total']} nodes, "
                  f"{current['edges_total']} edges")
            continue

        found = compare(name, current, json.loads(path.read_text()))
        if found:
            problems.extend(found)
            print(f"{name:12s} CHANGED   {len(found)} metric(s) moved")
        else:
            print(f"{name:12s} ok        {current['nodes_total']} nodes, "
                  f"{current['edges_total']} edges")

    if problems:
        print("\nMetrics moved. Each one is a bug until explained:\n")
        print("\n".join(problems))
        print("\nIf every change above is intended, re-run with --record.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
