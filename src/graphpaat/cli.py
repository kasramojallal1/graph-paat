"""Command line entry point. Step 1 exposes only `build`."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from . import attach
from . import documents as docs
from . import instructions
from . import store
from .build import assemble, with_documents
from .query import match, ranked_names, render, vocabulary, word_vocabulary


def deep(built, root: Path, out: Path | None, budget: int,
         ask_budget: int = attach.DEFAULT_ASK_TOKENS) -> dict:
    """The document lane, in the two-step shape D15 requires.

    graph-paat holds no API key and calls no provider. It writes down the prose
    that needs reading; the assistant already running it reads that file and
    writes its answer back; this ingests the answer. First run produces the
    question, second run consumes the answer.

    Returns the coverage block for the store, and prints what happened.
    """
    reading = docs.read(root, budget_tokens=budget)
    ask = attach.build_ask(reading, built.payload(), budget_tokens=ask_budget)
    folder = (Path(out) if out is not None else Path.cwd() / store.OUT_DIR).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    ask_path = folder / attach.ASK_FILE
    ask_path.write_text(ask.text, encoding="utf-8")

    read_tokens = reading.tokens_read
    print(f"\ndocuments: {len(reading.documents)} read "
          f"({read_tokens:,} tokens of prose), "
          f"{sum(len(d.sections) for d in reading.documents)} passages")
    # Every gap gets a line. A capped read that stays quiet reads as a full one.
    if reading.skipped:
        by_reason = Counter(why for _, _, why in reading.skipped)
        for why, count in by_reason.most_common():
            held = sum(tok for _, tok, w in reading.skipped if w == why)
            print(f"  ! {count} not read, {why} ({held:,} tokens)")
        for path, tok, why in reading.skipped[:5]:
            print(f"      {path} ({tok:,} tokens, {why})")
        if len(reading.skipped) > 5:
            print(f"      ... and {len(reading.skipped) - 5} more")
    for path, why in reading.unreadable[:5]:
        print(f"  ! {path}: {why}")
    if ask.skipped_as_index:
        print(f"  {ask.skipped_as_index} passages skipped: they name more than "
              f"{attach.INDEX_MENTIONS} symbols, so they are a reference table, "
              f"not an explanation")
    if ask.skipped_over_budget:
        print(f"  ! {ask.skipped_over_budget} passages not put to the model, over "
              f"the {ask_budget:,}-token ask budget ({ask.tokens_over_budget:,} "
              f"tokens). Raise --ask-budget to include them.")

    coverage = {
        "read": [d.path for d in reading.documents],
        "skipped": [{"path": p, "tokens": tok, "why": w}
                    for p, tok, w in reading.skipped],
        "unreadable": [{"path": p, "why": w} for p, w in reading.unreadable],
        "tokens_read": read_tokens,
        "tokens_skipped": reading.tokens_skipped,
        "passages_asked": ask.sections,
        "passages_over_ask_budget": ask.skipped_over_budget,
        "passages_dropped_as_index": ask.skipped_as_index,
        "ask_digest": ask.digest,
    }

    answer_path = folder / attach.ANSWER_FILE
    answer, why = attach.load_answer(answer_path, ask.digest)
    if why:
        print(f"\n{ask.sections} passages need reading (~{ask.tokens:,} tokens).")
        print(f"  {why}")
        print(f"\n  1. read  {ask_path}")
        print(f"  2. write {answer_path}")
        print("  3. run this same command again")
        print("\nThe map below is the code lane only until you do.")
        coverage["state"] = "waiting for an answer"
        return coverage

    ingested = attach.ingest(answer, ask.manifest, reading)
    with_documents(built, reading, ingested)
    print(f"  {ingested.attached} passages attached to a symbol, "
          f"{ingested.empty} described nothing")
    for reason, count in sorted((ingested.rejected or {}).items()):
        # A refusal is a result. The one that matters is an id the model typed
        # rather than picked -- it would have become a node nothing could reach.
        print(f"  ! {count} rejected: {reason}")
    coverage["state"] = "attached"
    coverage["attached"] = ingested.attached
    coverage["described_nothing"] = ingested.empty
    coverage["rejected"] = ingested.rejected or {}
    return coverage


def build(root: Path, out: Path | None = None, deep_lane: bool = False,
          prose_budget: int = docs.DEFAULT_BUDGET_TOKENS,
          ask_budget: int = attach.DEFAULT_ASK_TOKENS) -> int:
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

    coverage = (deep(built, root, out, prose_budget, ask_budget)
                if deep_lane else None)
    # Re-read from the build: the document lane appends to both lists.
    path = store.write(root, built.nodes, built.edges, collisions, failed, out=out,
                       groups=built.groups, gods=built.gods, documents=coverage)
    print(f"\nwritten: {path}")
    return 0


def vocab(out: Path | None, contains: str | None, limit: int,
          as_words: bool = False) -> int:
    """Publish the graph's names so the calling agent can expand a question
    against them (D9). Optionally filtered, because 11,000 names is a lot to
    hand a model that only needs the ones near one topic."""
    graph = store.read(Path("."), out=out)
    if as_words:
        # The whole list, unfiltered and unlimited: it exists to be read in one
        # go and turned into search terms, so truncating it defeats the point.
        found = word_vocabulary(graph)
        if contains:
            found = [w for w in found if contains.lower() in w]
        print(f"{len(found)} words used in names in this codebase.")
        print("Pick the ones that match your question and pass them to `query`.")
        print("A word that is not here cannot be found - do not invent one.\n")
        print("\n".join(found))
        return 0
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
          seeds_wanted: int, per_node: int, claims: bool = False) -> int:
    graph = store.read(Path("."), out=out)
    seeds, more = match(graph, terms, limit=seeds_wanted, claims=claims)
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
  graph-paat build <path> [--deep] [--prose-budget N] [--ask-budget N] [--out <dir>]
  graph-paat overview [--top N] [--out <dir>]
  graph-paat install [--host claude|agents|gemini|cursor|copilot] [--all] [--remove]
  graph-paat vocab --words [--out <dir>]        every word used in a name
  graph-paat vocab [--contains <text>] [--limit N] [--out <dir>]
  graph-paat query <term> [<term>...] [--budget N] [--depth N] [--seeds N]
                                     [--per-node N] [--claims] [--out <dir>]

  --deep    also read the repository's documents (.md, .rst, .txt, .pdf). Two
            steps: the first run writes the passages that need reading, you
            answer them, the second run folds the answers in.
  --claims  let a sentence from a document be an answer in its own right,
            rather than only a signpost to the code it describes."""


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
        prose_budget, rest = _take(rest, "--prose-budget", int,
                                   docs.DEFAULT_BUDGET_TOKENS)
        ask_budget, rest = _take(rest, "--ask-budget", int,
                                 attach.DEFAULT_ASK_TOKENS)
        deep_lane = "--deep" in rest
        rest = [a for a in rest if a != "--deep"]
        return build(Path(rest[0] if rest else "."), out=out,
                     deep_lane=deep_lane, prose_budget=prose_budget,
                     ask_budget=ask_budget)
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
        return vocab(out, contains, limit, as_words="--words" in rest)
    budget, rest = _take(rest, "--budget", int, 2000)
    depth, rest = _take(rest, "--depth", int, 2)
    seeds_wanted, rest = _take(rest, "--seeds", int, 6)
    per_node, rest = _take(rest, "--per-node", int, 10)
    claims = "--claims" in rest
    rest = [a for a in rest if a != "--claims"]
    if not rest:
        print(USAGE, file=sys.stderr)
        return 2
    return query(rest, out, budget, depth, seeds_wanted, per_node, claims=claims)


if __name__ == "__main__":
    raise SystemExit(main())
