"""Command line entry point. Step 1 exposes only `build`."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from . import store
from . import instructions
from .build import assemble
from .query import match, ranked_names, render, vocabulary


def build(root: Path, out: Path | None = None) -> int:
    built = assemble(root)
    nodes, edges = built.nodes, built.edges
    call_edges, refusals = built.call_edges, built.refusals
    collisions, failed = built.collisions, built.failed
    kinds = Counter(n.kind for n in nodes)
    rels = Counter(e.relation for e in edges)

    from . import languages
    print(f"corpus:    {root.resolve()}")
    print(f"languages: {', '.join(languages.names())}")
    # A file we skipped because a grammar is missing is invisible loss, which
    # is the thing this tool exists to refuse. Say it.
    absent = languages.missing()
    if absent:
        for name, why in absent.items():
            skipped = len([p for p in root.rglob(f"*.{name}")])
            print(f"  ! {name}: not read ({why})"
                  + (f" - {skipped} file(s) skipped" if skipped else ""))
    print(f"nodes:     {len(nodes)}")
    for kind, count in kinds.most_common():
        print(f"  {kind:10s} {count}")

    print(f"edges:     {len(edges)}")
    for rel, count in rels.most_common():
        print(f"  {rel:10s} {count}")

    # Report per relation. Lumping calls and imports together made the resolved
    # count read as a calls figure when it was not.
    for relation in ("calls", "imports"):
        group = [e for e in call_edges if e.relation == relation]
        ok = [e for e in group if e.resolved]
        print(f"\n{relation}: {len(ok)} resolved, {len(group) - len(ok)} unresolved")
        for how, count in Counter(e.reason for e in ok).most_common():
            print(f"  {count:6d}  {how}")
    print("\nrefusals, by reason:")
    for why, count in refusals.most_common(8):
        print(f"  {count:6d}  {why}")

    # Every run says what it could not do. graphify computes this and drops it;
    # D6 says a gap the user cannot see is worse than one they can.
    print(f"unique ids: {len(set(n.id for n in nodes))}")
    lines = collisions.report()
    if lines:
        print(f"\n{len(lines)} id collisions (a distinct symbol is lost in each):")
        for line in lines[:10]:
            print(f"  {line}")
        if len(lines) > 10:
            print(f"  ... and {len(lines) - 10} more")
    if failed:
        print(f"\n{len(failed)} files could not be parsed:")
        for f in failed[:10]:
            print(f"  {f}")

    # A missing networkx degrades the build rather than failing it: the map
    # still works without knowing the shape of the codebase.
    if built.grouping_error:
        print(f"\nno grouping: {built.grouping_error}")
    else:
        print(f"\ngroups:    {len(built.groups['groups'])} "
              f"({built.groups['ungrouped']} nodes in none)")
        for g in built.groups["groups"][:5]:
            print(f"  {g['size']:6d}  {g['name']}")

    path = store.write(root, nodes, edges, collisions, failed, out=out,
                       groups=built.groups, gods=built.gods)
    print(f"\nwritten: {path}")
    return 0


def vocab(out: Path | None, contains: str | None, limit: int) -> int:
    """Publish the graph's names so the calling agent can expand a question
    against them (D9). Optionally filtered, because 11,000 names is a lot to
    hand a model that only needs the ones near one topic."""
    graph = store.read(Path("."), out=out)
    names = ranked_names(graph)
    if contains:
        names = [n for n in names if contains.lower() in n.lower()]
    print(f"{len(names)} names" + (f" containing '{contains}'" if contains else "")
          + ", most connected first")
    for name in names[:limit]:
        print(f"  {name}")
    if len(names) > limit:
        print(f"  ... {len(names) - limit} more (raise --limit, or filter with --contains)")
    return 0


def query(terms: list[str], out: Path | None, budget: int, depth: int,
          seeds_wanted: int, per_node: int) -> int:
    graph = store.read(Path("."), out=out)
    seeds, more = match(graph, terms, limit=seeds_wanted)
    print(render(graph, seeds, budget=budget, depth=depth, more=more, per_node=per_node))
    return 0


def overview(out: Path | None, top: int) -> int:
    """What are the main parts of this codebase, and what does it lean on?

    The first question anyone asks of an unfamiliar repository, and the one a
    map of individual symbols cannot answer.
    """
    data = store.read(Path("."), out=out).get("overview", {})
    groups = data.get("groups", [])
    if not groups:
        print("no grouping in this graph - rebuild with networkx installed")
        return 1
    print(f"{len(groups)} groups ({data.get('ungrouped', 0)} nodes in none)\n")
    for g in groups[:top]:
        folders = ", ".join(g["folders"][:2])
        print(f"  {g['size']:6d}  {g['name']}")
        if folders and folders != ".":
            print(f"          {folders}")
    print("\nmost connected symbols:")
    for n in data.get("god_nodes", [])[:top]:
        print(f"  {n['connections']:6d}  {n['label']:32s} {n['file']}:L{n['line']}")
    return 0


def install(root: Path, hosts: list[str], all_hosts: bool, remove: bool) -> int:
    """Put the instructions where an assistant will read them."""
    if remove:
        done = instructions.uninstall(root)
        for path, what in done:
            print(f"  {what:24s} {path}")
        if not done:
            print("nothing installed here")
        return 0
    done = instructions.install(root, hosts or None,
                                existing_only=not (hosts or all_hosts))
    for path, what in done:
        print(f"  {what:24s} {path}")
    if all(w.startswith("skipped") for _, w in done):
        print("\nNo instructions file found. Name a host to create one, e.g.:")
        print("  graph-paat install --host claude")
        print(f"  hosts: {', '.join(instructions.TARGETS)}")
    return 0


USAGE = """usage:
  graph-paat build <path> [--out <dir>]
  graph-paat overview [--top N] [--out <dir>]
  graph-paat install [--host claude|agents|gemini|cursor|copilot] [--all] [--remove]
  graph-paat vocab [--contains <text>] [--limit N] [--out <dir>]
  graph-paat query <term> [<term>...] [--budget N] [--depth N] [--seeds N]
                                     [--per-node N] [--out <dir>]"""


def _take(rest: list[str], flag: str, cast=str, default=None):
    """Pull `--flag value` out of the argument list, returning the value."""
    if flag not in rest:
        return default, rest
    i = rest.index(flag)
    return cast(rest[i + 1]), rest[:i] + rest[i + 2:]


def main(argv: list[str] | None = None) -> int:
    """Entry point. Every expected failure leaves as a message, not a traceback.

    Following the published instructions literally, an agent that queries
    before building got a Python stack trace. The store already raises a
    message saying exactly what to run; nothing was catching it.
    """
    try:
        return _run(argv if argv is not None else sys.argv[1:])
    except (FileNotFoundError, ValueError) as exc:
        print(f"graph-paat: {exc}", file=sys.stderr)
        return 1
    except BrokenPipeError:
        return 0          # piping into `head` is normal usage, not an error


def _run(argv: list[str]) -> int:
    if not argv or argv[0] not in ("build", "vocab", "query", "overview", "install"):
        print(USAGE, file=sys.stderr)
        return 2
    command, rest = argv[0], argv[1:]
    out, rest = _take(rest, "--out", Path)

    if command == "build":
        return build(Path(rest[0] if rest else "."), out=out)
    if command == "install":
        hosts: list[str] = []
        while "--host" in rest:
            host, rest = _take(rest, "--host")
            hosts.append(host)
        all_hosts = "--all" in rest
        remove = "--remove" in rest
        rest = [a for a in rest if a not in ("--all", "--remove")]
        return install(Path(rest[0] if rest else "."), hosts, all_hosts, remove)
    if command == "overview":
        top, rest = _take(rest, "--top", int, 10)
        return overview(out, top)
    if command == "vocab":
        contains, rest = _take(rest, "--contains")
        limit, rest = _take(rest, "--limit", int, 60)
        return vocab(out, contains, limit)
    budget, rest = _take(rest, "--budget", int, 2000)
    depth, rest = _take(rest, "--depth", int, 2)
    seeds_wanted, rest = _take(rest, "--seeds", int, 6)
    per_node, rest = _take(rest, "--per-node", int, 10)
    if not rest:
        print(USAGE, file=sys.stderr)
        return 2
    return query(rest, out, budget, depth, seeds_wanted, per_node)


if __name__ == "__main__":
    raise SystemExit(main())
