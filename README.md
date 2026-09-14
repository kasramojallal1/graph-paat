# graph-paat

**Give an AI agent a small map of a codebase too large to read.**

A coding agent can read a repository that fits in its context window. Past that, it cannot —
and no amount of grepping fixes not fitting. `graph-paat` reads the repository once, builds a
graph of what is in it and how those things connect, and answers a question with a few
hundred tokens instead of a few million.

```
$ graph-paat query QuerySet

graph: 15709 nodes | seeds: 1 | shown: 1 | ~138 tokens

QuerySet    db/models/query.py:L293    [class · read from code]    part of: db/models · QuerySet
    part of     query.py        db/models/query.py:L1      [file · read from code]
    contains    order_by        db/models/query.py:L1695   [method · read from code]
    contains    values_list     db/models/query.py:L1364   [method · read from code]
    contains    bulk_create     db/models/query.py:L757    [method · read from code]
    contains    select_related  db/models/query.py:L1575   [method · read from code]
    contains    delete          db/models/query.py:L1164   [method · read from code]
    ... 103 more links (raise --per-node)
```

That is Django's `QuerySet` in 138 tokens: where it lives, which part of the codebase it
belongs to, and the seven of its 111 methods that the rest of the code actually uses. Reading
`query.py` instead costs about 22,000 tokens. Django is 155,000 lines.

Which seven is the point. A class with 111 methods cannot be summarised by the first ten
alphabetically — that gives `_add_hints` and `_batched_insert`. Links are ranked by how
connected their target is, so the answer is `order_by` and `bulk_create`.

## Does it find the right file?

The test that counts is one nobody here wrote. [SWE-bench Lite](https://www.swebench.com/)
is 300 real GitHub issues from twelve Python projects, filed by their developers years before
this tool existed, and for each one the right answer is recorded: the file the maintainers
actually changed to fix it. The 189 issues from Django (114) and sympy (75) — 155,000 and
753,000 lines — were asked using only the issue's one-line title, and scored on whether that
file is in the top three results. The same issues, titles and scoring were run against
[graphify](https://github.com/Graphify-Labs/graphify) on its own graphs of the same two packages.

| | graph-paat | graphify |
|---|---|---|
| Django — 114 issues | **51%** | 37% |
| sympy — 75 issues | 48% | **52%** |
| **all 189** | **50%** | 43% |

Half the time, from one line written by a stranger, the file that needed changing is in the
top three. sympy is the one it loses: a codebase whose function names are things like
`_eigenvals`, where the English lives in the docstrings rather than the identifiers.

The issues ship with the repository, so the graph-paat column takes one command per package
and no network:

```bash
python benchmarks/swe_bench.py score django /path/to/site-packages/django
python benchmarks/swe_bench.py score sympy  /path/to/site-packages/sympy
```

`--graphify path/to/graphify-out/graph.json` scores a graphify build of the same package
with the same rule (the table is graphify 0.9.49; older releases rank differently), and
`fetch` re-downloads the 300 issues from HuggingFace.

## Install

```bash
pip install -e .
```

Reading Python needs nothing installed — the standard library's own parser does it.
`networkx` is used for grouping. Every other language needs its tree-sitter grammar, and
reading PDFs needs a PDF library:

```bash
pip install -e ".[all]"      # or ".[go]" / ".[typescript]" / ".[rust]" / ".[pdf]" ...
```

The extras are `go`, `typescript`, `javascript`, `java`, `csharp`, `rust`, `ruby`, `php`,
`swift`, `kotlin`, `bash`, `lua`, `c` and `cpp`.

Without one, those files are skipped and the build says so rather than quietly leaving them
out.

## Use

```bash
graph-paat build /path/to/repo          # read the repo, write the graph
graph-paat overview                     # what are the main parts of this codebase
graph-paat vocab --contains auth        # what names exist in the graph
graph-paat query login verify           # a map of those names and their links
graph-paat build /path/to/repo --deep   # also read the repo's own documents
```

### Every line says where it came from

A parsed function is a fact. A sentence someone wrote is a claim that may have been true when
it was written. The map never blurs the two:

```
Console                             console.py:L593   [class · read from code]
"the entry point for all output"    README.md:L12     [claim · from a document]
```

An agent that cannot tell those apart reads a five-year-old README line as something a parser
checked today, which is exactly the mistake a map is supposed to prevent.

### Reading the repository's prose — `--deep`

Off by default. With it, `graph-paat` also reads `.md`, `.rst`, `.txt` and PDF files and
attaches what they say to the symbols they describe. Useful where the names are opaque but the
documentation is good.

**It never calls a model, and there is no API key anywhere in this project.** The agent running
the tool already has a model, so the tool writes down the passages that need reading, the agent
reads them and writes its answer back, and the second run folds it in:

```bash
graph-paat build . --deep     # writes graph-paat-out/documents-to-read.md
                              # ... you answer it in document-answers.json ...
graph-paat build . --deep     # folds the answers into the map
```

**The agent picks from a list; it never types an identifier.** Each passage comes with the
names that already exist in the graph, and an id that is not on that list is rejected rather
than created — so a document about something not in the map attaches to nothing, which is the
right answer. Prose is capped per repository and the build prints exactly which files the cap
excluded and what they held.

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

### Let an agent find it

```bash
graph-paat install                 # add instructions to CLAUDE.md, AGENTS.md, … if present
graph-paat install --host cursor   # or create one for a specific assistant
graph-paat install --remove        # take them out again
```

Writes a short marked block into whichever instructions file your assistant already reads,
so it reaches for the tool on its own instead of waiting to be told. Only files that already
exist are touched unless you name a host, the block sits between markers so a reinstall
replaces exactly what the last one wrote, and `--remove` restores the file as it was.

### For an agent

The intended loop is two calls. The agent has a language model; `graph-paat` does not.

1. `graph-paat vocab` publishes the names the graph actually contains, so the agent can work
   out that a question about "authentication" is a question about `login` and `Session`.
2. `graph-paat query login Session` returns the map.

Nothing here calls a model, needs an API key, or gives a different answer on two runs.

## What it records

**Nodes** — every file, class, function, method, and docstring.
**Edges** — `contains`, `calls`, `imports`, `inherits`, and `rationale_for` (a docstring
explaining the thing it sits on).

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

840 tests covering every identity rule, every resolution rule, every language, and every
reason the resolver refuses a call. Several exist because the bug they describe shipped once: a builtin
`set()` resolved to a same-named function in a vendored package, a docstring id that clashed
with a real symbol named `_doc`, edges collected twice because they are reachable from both
ends, and output written into the repository being analysed.

### Against real repositories

Unit tests use code small enough to know the answer by hand. That is not enough: some defects
only appear at scale. So the tool is also run over forty real repositories across all
fifteen languages, and every number it produces is recorded and compared on the next run.

```bash
cp tests/corpora.sample.json tests/corpora.json   # set paths for your machine
python -m tests.corpus_runner                     # compare against baselines
python -m tests.corpus_runner --record            # accept current numbers
```

| language | corpora |
|---|---|
| Python | requests, rich, pydantic, django, numpy, pandas, scipy, sympy, torch, transformers |
| Go | uuid, logrus, grpc |
| TypeScript | redux, rxjs, zod |
| JavaScript | axios, three.js |
| Java | gson, commons-lang |
| C# | Serilog, Newtonsoft.Json |
| Rust | ripgrep, tokio |
| C / C++ | curl, redis / fmt, googletest |
| Ruby, PHP, Swift, Kotlin, Bash, Lua | two each — jekyll, rack; guzzle, slim; Alamofire, swift-algorithms; okhttp, coroutines; bats, bash-it; lazy.nvim, telescope |

`requests` is small enough to verify entirely by hand; `torch` is 890,000 lines.

**A metric that moves without an explanation is a bug until proven otherwise.** Reintroducing
a naming bug that had already been fixed moves `id_collisions` on Django from 54 to 151, and
the runner names the metric and the delta. The same bug is invisible on `requests`, which is
why one corpus is not enough.

## Status

Early. It reads **fifteen languages**: Python, Go, TypeScript, JavaScript, Java, C#, Rust,
Ruby, PHP, Swift, Kotlin, Bash, Lua, C and C++. Verified by hand against Django, pydantic,
grpc, redux and several other large packages, and regression-checked against forty recorded
corpora.

Every language produces the same shapes. A Go struct, a TypeScript interface and a Python
class are all `class` nodes; Go embedding, TypeScript `extends` and `implements`, and Python
subclassing are all `inherits`; a docstring, a `//` doc comment and a `/** */` block are all
claims. Nothing downstream — grouping, ranking, query — knows more than one language exists,
which was the real test of whether the model was language-neutral.

How well calls resolve depends on how much the language declares. Go states a method's
receiver outright, so every call on it resolves exactly. TypeScript usually annotates
parameters, which is nearly as good. Python has to infer the same fact from an assignment,
and often cannot.

## Licence

MIT.
