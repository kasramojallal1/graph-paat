# graph-paat

**Give an AI agent a small map of a codebase too large to read.**

A coding agent can read a repository that fits in its context window. Past that, it cannot —
and no amount of grepping fixes not fitting. `graph-paat` reads the repository once, builds a
graph of what is in it and how those things connect, and answers a question with a few
hundred tokens instead of a few million.

```
$ graph-paat query QuerySet

graph: 15709 nodes | seeds: 1 | shown: 1 | ~138 tokens

QuerySet    db/models/query.py:L293    [class]    part of: db/models · QuerySet
    part of     query.py        db/models/query.py:L1
    contains    order_by        db/models/query.py:L1695
    contains    values_list     db/models/query.py:L1364
    contains    bulk_create     db/models/query.py:L757
    contains    select_related  db/models/query.py:L1575
    contains    delete          db/models/query.py:L1164
    ... 103 more links (raise --per-node)
```

That is Django's `QuerySet` in 138 tokens: where it lives, which part of the codebase it
belongs to, and the seven of its 111 methods that the rest of the code actually uses. Reading
`query.py` instead costs about 22,000 tokens. Django is 155,000 lines.

Which seven is the point. A class with 111 methods cannot be summarised by the first ten
alphabetically — that gives `_add_hints` and `_batched_insert`. Links are ranked by how
connected their target is, so the answer is `order_by` and `bulk_create`.

## Install

```bash
pip install -e .
```

The parser is Python's own `ast` module, so reading a repository needs nothing installed.
`networkx` is used for grouping only; without it the graph still builds.

## Use

```bash
graph-paat build /path/to/repo          # read the repo, write the graph
graph-paat overview                     # what are the main parts of this codebase
graph-paat vocab --contains auth        # what names exist in the graph
graph-paat query login verify           # a map of those names and their links
```

`overview` answers the question you ask first about an unfamiliar repository, and the one a
map of individual symbols cannot. On Django:

```
182 groups (243 nodes in none)

    2811  utils · ValidationError
     849  db/models · QuerySet
     564  db/migrations · MigrationAutodetector
     380  contrib/admin · ModelAdmin
     336  db/backends/oracle · DatabaseOperations

most connected symbols:
     QuerySet, Query, ValidationError, ImproperlyConfigured, GEOSGeometryBase
```

Groups come out of the call graph, not the folder layout — but each is named after the
directory most of its members live in and its most connected symbol, because a group called
`47` tells a reader nothing.

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
that only records successes, so the map has to say which one it is.

One exception, and it is a judgement: no gap is drawn for the names of built-in container
methods — `get`, `items`, `copy`, `update`. `config.get("timeout")` is a dictionary access,
not a link worth chasing, and marking every one of them crowded out the gaps that mean
something. They are still counted in the coverage report. A call that genuinely resolves to a
method named `get` is unaffected. The same applies to
identity: when two different symbols would take the same name, `graph-paat` reports it rather
than silently keeping one.

The written graph carries a `coverage` block — files that failed to parse, id collisions,
unresolved edge counts — so a gap is still visible when someone reads the file a week later.

A docstring is printed as `[claim]`, not as fact. Comments go stale; parsers do not.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

78 unit tests covering every identity rule, every resolution rule, and every reason the
resolver refuses a call. Several exist because the bug they describe shipped once: a builtin
`set()` resolved to a same-named function in a vendored package, a docstring id that clashed
with a real symbol named `_doc`, edges collected twice because they are reachable from both
ends, and output written into the repository being analysed.

### Against real repositories

Unit tests use code small enough to know the answer by hand. That is not enough: some defects
only appear at scale. So the tool is also run over six large installed packages, and every
number it produces is recorded and compared on the next run.

```bash
cp tests/corpora.sample.json tests/corpora.json   # set paths for your machine
python -m tests.corpus_runner                     # compare against baselines
python -m tests.corpus_runner --record            # accept current numbers
```

| corpus | what it stresses |
|---|---|
| `requests` | small enough to verify entirely by hand |
| `rich` | modern typed classes, properties |
| `pydantic` | metaclasses and generated code |
| `django` | classic OO, decorators, platform branches |
| `numpy` | C extensions, dynamic imports |
| `torch` | scale — 890,000 lines |

**A metric that moves without an explanation is a bug until proven otherwise.** Reintroducing
a naming bug that had already been fixed moves `id_collisions` on Django from 54 to 151, and
the runner names the metric and the delta. The same bug is invisible on `requests`, which is
why one corpus is not enough.

## Status

Early. It reads **Python only**. Verified by hand against Django and several large packages.

## Licence

MIT.
