# graph-paat

**Give an AI agent a small map of a codebase too large to read.**

A coding agent can read a repository that fits in its context window. Past that, it cannot —
and no amount of grepping fixes not fitting. `graph-paat` reads the repository once, builds a
graph of what is in it and how those things connect, and answers a question with a few
hundred tokens instead of a few million.

```
$ graph-paat query MinHashLSH

graph: 11020 nodes | seeds: 1 | shown: 1 | ~185 tokens

MinHashLSH    graphify/_minhash.py:L84    [class, ast]
    part of       _minhash.py    graphify/_minhash.py:L1
    rationale_for [claim] "Band-hashing LSH — same API as datasketch.MinHashLSH..."
    contains      __init__       graphify/_minhash.py:L87
    contains      insert         graphify/_minhash.py:L92
    contains      query          graphify/_minhash.py:L101
    called by     deduplicate_entities    graphify/dedup.py:L503
```

That answer is 185 tokens: where the class lives, what it claims about itself, what is inside
it, and who uses it. Reading the file it describes costs 1,035 tokens. The repository it came
from is 1.78 million.

## Install

```bash
pip install -e .
```

No runtime dependencies. The parser is Python's own `ast` module.

## Use

```bash
graph-paat build /path/to/repo          # read the repo, write the graph
graph-paat vocab --contains auth        # what names exist in the graph
graph-paat query login verify           # a map of those names and their links
```

`build` runs once. `vocab` and `query` read what it wrote. A full build of Django — 879
files, 155,000 lines — takes about a second.

### For an agent

The intended loop is two calls. The agent has a language model; `graph-paat` does not.

1. `graph-paat vocab` publishes the names the graph actually contains, so the agent can work
   out that a question about "authentication" is a question about `login` and `Session`.
2. `graph-paat query login Session` returns the map.

Nothing here calls a model, needs an API key, or gives a different answer on two runs.

## What it records

**Nodes** — every file, class, function, method, and docstring.
**Edges** — `contains`, `calls`, `imports`, and `rationale_for` (a docstring explaining the
thing it sits on).

**And what it could not work out.** Every call the resolver refuses is counted with a reason,
and where a plausible target exists it is drawn as an unresolved edge:

```
login()  calls  verify()     resolved
login()  calls  ?connect     UNRESOLVED (receiver type unknown, auth.py:44)
```

This is deliberate. A missing edge and a nonexistent relationship look identical in a graph
that only records successes, so the map has to say which one it is. The same applies to
identity: when two different symbols would take the same name, `graph-paat` reports it rather
than silently keeping one.

The written graph carries a `coverage` block — files that failed to parse, id collisions,
unresolved edge counts — so a gap is still visible when someone reads the file a week later.

A docstring is printed as `[claim]`, not as fact. Comments go stale; parsers do not.

## Status

Early. It reads **Python only**, and it has no test suite yet — the roadmap puts that first.
It has been verified by hand against `graphify`, `django`, and `requests`.

## Licence

MIT.
