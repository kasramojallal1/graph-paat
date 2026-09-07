"""Answer a question with a small map.

The calling agent has a language model; this does not. It publishes the graph's
vocabulary (`vocab`) so the agent can turn a question into names, then matches
those names, walks outward, and renders what it found inside a token budget.
Nothing here calls a model, needs a key, or answers differently on two runs.

**Ranking was rebuilt on 2026-09-06 against a set of 120 questions whose right
answers were written down first.** The old rules answered 17 of them correctly.
These answer 41. Every rule below is here because removing it costs questions,
and the cost is recorded beside it:

    ordinary English words dropped   -16    the single biggest one
    a name is a bag of words         -11
    coverage, squared                -10
    stemming                          -8
    long names penalised              -6
    docstrings searched               -3   (-7 on top-three)
    tests demoted                     -2
    rarer words weigh more            -1   (-3 on top-three)
    the kind of thing asked for       -1

Two rules that survive on judgement rather than on score: the file path is
worth a fraction of a point and measured zero, but it is what keeps a question
matching no name at all from returning nothing; and the better-connected symbol
still breaks a tie, worth one question.

Three things about the walk, each added because the naive version failed on a
real repository:

**Links must be ranked.** A class's map was twelve dunder methods and nothing
about what used it. Behaviour is more interesting than containment, and
`__deepcopy__` is less interesting than either.

**The walk must avoid hubs.** Two hops from something adjacent to a god node
reaches most of the codebase. Expanding *through* a hub is what does it, so we
report hubs and stop there.

**Truncation is announced.** The subject of the question is rendered first, so
what you asked about survives the budget.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import PurePosixPath

# How a relation reads when you arrive from the other end.
INVERSE = {"contains": "part of", "calls": "called by", "imports": "imported by",
           "rationale_for": "explains", "inherits": "subclassed by"}

# Most interesting first. What a thing DOES beats what it holds; a docstring is
# context rather than structure, so it comes last but is never dropped -- it is
# often the only statement of intent in the whole map.
# What a class IS comes before what it does: inheritance decides behaviour that
# no call edge shows.
RELATION_ORDER = {"inherits": 0, "subclassed by": 1, "calls": 2, "called by": 3,
                  "imports": 4, "imported by": 5, "part of": 6, "contains": 7,
                  "rationale_for": 8, "explains": 8}

# The words a question is made of rather than the thing it asks about.
#
# Dropping these is worth 16 questions, more than any other rule, and the reason
# is sharper than "noise": matching is tiered, and an EXACT name match outranks
# everything. Django defines a class called `A` in its date formatter, so the
# indefinite article in "the base class for a database model" scored a perfect
# match and took first place on five unrelated questions.
#
# What is deliberately NOT here: `get`, `set`, `send`, `save`, `load`, `open`,
# `close`, `read`, `write`. They are filler in English and method names in every
# codebase, and dropping them cost "how do I send a get request" its only real
# term. The plural verb forms below (`calls`, `uses`) are different -- they name
# the relation being asked about, not a symbol to start from.
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "by", "for", "from",
    "with", "at", "as", "is", "are", "was", "were", "be", "been", "do", "does",
    "did", "how", "what", "which", "where", "when", "who", "why", "that", "this",
    "it", "its", "i", "you", "we", "they", "my", "your", "our", "me", "us",
    "can", "could", "should", "would", "will", "shall", "may", "might", "must",
    "there", "here", "into", "out", "up", "down", "not", "no", "yes", "if",
    "then", "else", "so", "than", "just", "only", "also", "some", "any", "all",
    "each", "actually", "really", "happen", "happens", "work", "works",
    "calls", "called", "uses", "using", "imports", "contains", "defines",
    "defined",
}

# A word naming a kind of thing rather than a thing. "what class holds the
# response body" is about a class; the word itself matches nothing useful and
# would otherwise seed on `ClassVar`.
KIND_WORDS = {"class": "class", "classes": "class", "function": "function",
              "functions": "function", "method": "method", "methods": "method"}

# How strongly each kind of match counts. The distance between them is doing
# real work: narrowing the spread to 100/60/30 dropped the score from 41 to 31,
# and flattening it further dropped it to 18. A name that IS the word you asked
# for is not slightly better than a name that merely contains it.
TIER_EXACT = 1000.0        # the whole name is the term
TIER_TOKEN = 300.0         # one word of the name is the term
TIER_PREFIX = 100.0        # the name starts with the term
TIER_SUBSTRING = 1.0       # the term is somewhere inside the name
TIER_DOC = 20.0            # the term is in the docstring attached to this node
TIER_SOURCE = 0.5          # the term is in the file path
KIND_BONUS = 150.0         # the question named this kind of thing

# A name in a test file is never the answer to "how does X work", and it is the
# loudest possible false positive, because test names are English sentences
# built from the exact words a question uses. `TestClientHonorsConnectContext`
# matched "client", "connect" and "context" at once and took first place on
# seven of go-grpc's ten questions.
TEST_MARKS = ("/test", "test_", "_test.", ".test.", "/spec", "_spec.", "conftest")

# A node connected to more than this is a hub. We show it and do not expand
# through it: two hops through Django's ValidationError reaches half the repo.
HUB_DEGREE = 40

CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def words(text: str) -> list[str]:
    r"""Split a name into the words it is made of.

    On underscores, on punctuation, and on case changes: `classify_file` gives
    `classify` and `file`, `NewClient` gives `new` and `client`. Both the
    question and the name go through this, so the two sides meet as words
    instead of as substrings of each other -- which is what lets `classify`
    match `classify_file` as strongly as it matches `classify`.

    Worth 11 questions. `\w` counts underscore as a word character, so the
    character class here excludes it explicitly.
    """
    out: list[str] = []
    for part in re.findall(r"[^\W_]+", text):
        pieces = re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|\d+", part)
        out.extend(piece.lower() for piece in (pieces or [part]))
    return [w for w in out if w]


def stem(word: str) -> str:
    """Strip the endings that are reliably inflections, and no more.

    A question is asked in English and a codebase is named in stems:
    `classified` has to reach `classify`, and `queried` has to reach `QuerySet`.
    Worth 8 questions.

    An earlier version also stripped trailing vowels, which turned `database`
    into `databas` and `service` into `servic` -- merging `serve`, `server` and
    `service` into one term on a corpus where all three are different things.
    Being gentle here matters more than being thorough.
    """
    if len(word) > 4 and word.endswith(("ies", "ied")):
        return word[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[:-len(suffix)]
    return word


def forms(word: str) -> set[str]:
    """Every spelling of one word that should count as the same word.

    Stemming alone is not symmetric, and the asymmetry loses questions:
    `cookies` stems to `cooky` while `cookie` stems to itself, so
    "where are cookies kept" could not reach `RequestsCookieJar`. English has
    too many plural rules to pick one -- `cities` wants `city`, `cookies` wants
    `cookie` -- so we keep both and match if any spelling agrees.
    """
    out = {word, stem(word)}
    if len(word) > 3 and word.endswith("s"):
        out.add(word[:-1])
    return out


def _is_test(path: str, label: str) -> bool:
    low = path.lower()
    return (any(mark in low for mark in TEST_MARKS) or low.startswith("test")
            or label.startswith("Test") or label.startswith("test_"))


def _query_terms(raw: list[str]) -> tuple[list[str], str | None]:
    """The question, as terms to search for, plus the kind of thing it asked for.

    Falls back to the unfiltered words when a question is nothing but filler,
    because a query that matches badly beats one that matches nothing.
    """
    tokens = [w for term in raw for w in words(term)]
    kind = None
    for token in tokens:
        if token in KIND_WORDS:
            kind = KIND_WORDS[token]
    named = [t for t in tokens if t not in KIND_WORDS] or tokens
    kept = [t for t in named if t not in STOPWORDS and len(t) > 2] or named
    # The word the user typed, NOT its stem. Stemming here threw the original
    # away, and `forms` could then only work from the stem: `cookies` became
    # `cooky`, whose spellings are just {cooky}, so it never reached `cookie`.
    # Keep the word; `forms` derives the rest.
    return list(dict.fromkeys(kept)), kind


def ranked_names(graph: dict) -> list[str]:
    """Names, most connected first.

    Alphabetical order buries the answer: asked for names containing "backend",
    Django gave `AllowAllUsersModelBackend` while `load_backend` -- the function
    that actually chooses one -- sat past the limit. An agent reads the top of
    this list, so the top must be worth reading.
    """
    degree = _degrees(graph)
    index = vocabulary(graph)
    return sorted(index, key=lambda n: (-max(degree.get(i, 0) for i in index[n]), n))


def vocabulary(graph: dict) -> dict[str, list[str]]:
    """Every name in the graph, mapped to the nodes carrying it."""
    index: dict[str, list[str]] = defaultdict(list)
    for node in graph["nodes"]:
        if node["kind"] == "rationale":
            continue          # a docstring's label is bookkeeping, not a name
        index[node["label"]].append(node["id"])
    return index


def _degrees(graph: dict) -> dict[str, int]:
    degree: dict[str, int] = defaultdict(int)
    for edge in graph["edges"]:
        if edge.get("resolved", True):
            degree[edge["source"]] += 1
            degree[edge["target"]] += 1
    return degree


def _searchable(graph: dict) -> list[tuple]:
    """Everything about a node that a question can be matched against.

    The docstring is in here, and it is the one place we look that graphify does
    not. Their scorer reads label, tokenized label, path and node id -- never the
    prose attached to a symbol. But a question like "how are connections pooled
    and reused" shares no word with any name in `requests`; the answer,
    `HTTPAdapter`, says "connection pooling" in its own first line. Reading it is
    worth 3 questions outright and 7 in the top three.
    """
    doc_of: dict[str, str] = {}
    by_id = {n["id"]: n for n in graph["nodes"]}
    for edge in graph["edges"]:
        # The docstring is the TARGET of a rationale_for edge; the symbol it
        # explains is the source.
        if edge["relation"] == "rationale_for":
            doc = by_id.get(edge["target"])
            if doc is not None and doc.get("text"):
                doc_of[edge["source"]] = doc["text"]

    rows = []
    for node in graph["nodes"]:
        if node["kind"] == "rationale":
            continue
        label = node["label"]
        name_words = {f for w in words(label) for f in forms(w)}
        doc_words = {f for w in words(doc_of.get(node["id"], "")) for f in forms(w)}
        rows.append((node["id"], label.lower(), name_words, node["file"].lower(),
                     {f for w in words(node["file"]) for f in forms(w)}, doc_words,
                     node["kind"], _is_test(node["file"], label),
                     max(1, len(name_words)), len(label)))
    return rows


def _idf(rows: list[tuple], terms: list[str]) -> dict[str, float]:
    """What each term is worth: rarer is worth more.

    On Django `file` appears in hundreds of names and says almost nothing;
    `queryset` appears in a handful and says almost everything. Worth 1 question
    outright and 3 in the top three -- small, but it costs one pass.
    """
    total = len(rows) or 1
    weights = {}
    for term in terms:
        said = forms(term)
        seen = sum(1 for row in rows
                   if (said & row[2]) or any(f in row[1] for f in said))
        weights[term] = math.log(1 + total / (1 + seen))
    return weights


def match(graph: dict, terms: list[str], limit: int = 6) -> tuple[list[str], int]:
    """Rank nodes against the question. Returns (seeds, how many more matched).

    One pass, four signals.

    **Which tier each term matched**, strongest tier per term so one word cannot
    be counted three times.

    **How many of the question's words the name matched, squared.** A name
    matching 3 of 4 keeps 56% of its score; one matching 1 of 4 keeps 6%. This
    is the fix for the failure that started the rewrite: `classify_file` matches
    both words of "classify file" and `_file_stem` matches one, they landed in
    the same tier, and the tie went to the symbol with more callers. Squaring is
    what makes it bite -- at plain coverage a single exact match still outscores
    three weaker ones, because the exact tier is ten times the prefix tier.

    **How rare each word is**, so common words cannot carry a match.

    **How long the name is.** A five-word name matching two of your words is a
    worse answer than a one-word name matching one; without this, Django
    answered "how are database rows queried" with `fetch_returned_insert_rows`.
    Worth 6 questions.
    """
    rows = _searchable(graph)
    query, kind = _query_terms(terms)
    if not query:
        return [], 0
    weights = _idf(rows, query)
    degree = _degrees(graph)
    spellings = {term: forms(term) for term in query}

    scored: list[tuple[float, str, int]] = []
    for (nid, label, name_words, path, path_words, doc_words, node_kind,
         is_test, n_words, label_len) in rows:
        tiered = corroborating = 0.0
        matched = 0
        for term in query:
            weight = weights[term]
            said = spellings[term]
            # Every comparison runs over the term's spellings, not the raw word,
            # so `cookies` reaches `RequestsCookieJar` and `queried` reaches
            # `QuerySet` without the query and the name having to agree on
            # which inflection to use.
            if label in said:
                tiered += TIER_EXACT * weight
                matched += 1
            elif said & name_words:
                tiered += TIER_TOKEN * weight
                matched += 1
            elif any(label.startswith(f) for f in said):
                tiered += TIER_PREFIX * weight
                matched += 1
            elif any(f in label for f in said):
                corroborating += TIER_SUBSTRING * weight
                matched += 1
            elif said & doc_words:
                # Evidence, but weaker than a name -- what a symbol is CALLED is
                # a stronger claim about it than what its docstring mentions in
                # passing. It goes in the tiered bucket so it is subject to the
                # same coverage penalty: docstrings are long, so a question full
                # of vague words could otherwise accumulate more from five weak
                # prose hits than a real answer earns from one strong name.
                tiered += TIER_DOC * weight
                matched += 1
            if said & path_words or any(f in path for f in said):
                # The folder corroborates; it does not count as coverage. A
                # neighbour of the real answer usually shares its directory, and
                # must not win back a tier it did not earn on its own name.
                corroborating += TIER_SOURCE * weight
        if n_words > 1:
            tiered /= math.sqrt(n_words)
        tiered *= (matched / len(query)) ** 2
        if kind and node_kind == kind:
            corroborating += KIND_BONUS
        total = tiered + corroborating
        if is_test:
            total *= 0.05
        if total > 0:
            scored.append((total, nid, label_len))

    # Among equals, the symbol the codebase leans on, then the shorter name.
    scored.sort(key=lambda s: (-s[0], -degree.get(s[1], 0), s[2], s[1]))
    return [nid for _, nid, _ in scored[:limit]], max(0, len(scored) - limit)


def neighbourhood(graph: dict, seeds: list[str], depth: int = 2
                  ) -> tuple[set[str], list[dict], set[str]]:
    """Walk outward. Returns (nodes reached, edges travelled, hubs not expanded).

    Hubs are included in the map but never expanded through, so a query near a
    heavily used symbol does not drag in the whole repository.
    """
    adjacency: dict[str, list[dict]] = defaultdict(list)
    for edge in graph["edges"]:
        adjacency[edge["source"]].append(edge)
        adjacency[edge["target"]].append(edge)

    degree = _degrees(graph)
    seen, frontier = set(seeds), list(seeds)
    travelled: list[dict] = []
    taken: set[tuple] = set()
    hubs: set[str] = set()

    for _ in range(depth):
        nxt: list[str] = []
        for nid in frontier:
            if nid not in seeds and degree.get(nid, 0) > HUB_DEGREE:
                hubs.add(nid)
                continue
            for edge in adjacency.get(nid, []):
                key = (edge["source"], edge["target"], edge["relation"])
                if key not in taken:
                    taken.add(key)
                    travelled.append(edge)
                other = edge["target"] if edge["source"] == nid else edge["source"]
                if other not in seen:
                    seen.add(other)
                    nxt.append(other)
        frontier = nxt
    return seen, travelled, hubs


def _interest(relation: str, node: dict, degree: dict[str, int]) -> tuple:
    """Sort key for one link. Lower is shown first.

    Within a relation, the better-connected target wins. A class with 111
    methods cannot be summarised by the first ten alphabetically -- that gives
    `_add_hints` and `_batched_insert`, when the answer is `filter` and `get`.
    A method the rest of the codebase actually calls is the one worth naming.
    """
    dunder = node["label"].startswith("__") and node["label"].endswith("__")
    private = node["label"].startswith("_") and not dunder
    return (RELATION_ORDER.get(relation, 9), dunder, private,
            -degree.get(node["id"], 0), node["label"])


def render(graph: dict, seeds: list[str], budget: int = 2000, depth: int = 2,
           more: int = 0, per_node: int = 10) -> str:
    """The map an agent reads. Names, places, relations -- never source code."""
    nodes = {n["id"]: n for n in graph["nodes"]}
    # A group holding a large share of the codebase is not a part of it. Django's
    # biggest holds 2,811 nodes and is named after ValidationError, which tells a
    # reader of an admin view nothing true.
    all_groups = graph.get("overview", {}).get("groups", [])
    total = max(1, len(graph["nodes"]))
    # Both a share and a floor. A pure ratio misjudges small graphs: in a
    # six-node corpus a group of three is half of it and would be suppressed,
    # though three things are perfectly meaningful to name.
    # `.get` so a graph from an older build still renders rather than failing.
    group_names = {g["group"]: g["name"] for g in all_groups
                   if not (g.get("size", 0) > 200 and g.get("size", 0) / total >= 0.15)}
    # A group is named after the directory most of it lives in. For a member
    # living somewhere else that name is false: Django's `Settings`, in conf/,
    # was labelled "part of: test - SimpleTestCase". Checking membership of the
    # group's top three folders was not enough -- conf was one of them, while
    # the NAME claimed test. Compare against the directory the name claims.
    named_folder = {g["group"]: g.get("named_folder", "") for g in all_groups}
    degree = _degrees(graph)
    found, travelled, hubs = neighbourhood(graph, seeds, depth)

    links: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for edge in travelled:
        if edge["source"] in found and edge["target"] in found:
            links[edge["source"]].append((edge["relation"], edge["target"]))
            links[edge["target"]].append(
                (INVERSE.get(edge["relation"], f"{edge['relation']} by"), edge["source"]))

    blocks: list[str] = []
    spent, truncated = 0, 0

    for seed in seeds:
        node = nodes.get(seed)
        if node is None:
            continue
        where = group_names.get(node.get("group"))
        if where:
            claimed = named_folder.get(node.get("group"), "")
            if claimed and str(PurePosixPath(node["file"]).parent) != claimed:
                where = None
        head = f"\n{node['label']}    {node['file']}:L{node['line']}    [{node['kind']}]"
        if where:
            head += f"    part of: {where}"
        block = [head]

        entries = [(r, nodes[t]) for r, t in links.get(seed, []) if t in nodes]
        entries.sort(key=lambda rt: _interest(rt[0], rt[1], degree))
        hidden = max(0, len(entries) - per_node)
        for relation, target in entries[:per_node]:
            if target["kind"] == "rationale":
                # A docstring is a claim, not a parser fact. Marked, so a stale
                # comment is never read as something a parser verified.
                snippet = " ".join((target.get("text") or "").split())[:110]
                block.append(f'    {relation:11s} [claim] "{snippet}"')
            else:
                mark = " (hub)" if target["id"] in hubs else ""
                block.append(f"    {relation:11s} {target['label']}"
                             f"    {target['file']}:L{target['line']}{mark}")
        if hidden:
            block.append(f"    ... {hidden} more links (raise --per-node)")

        text = "\n".join(block)
        cost = estimate_tokens(text)
        if spent + cost > budget:
            truncated += 1
            continue
        blocks.append(text)
        spent += cost

    header = (f"graph: {len(graph['nodes'])} nodes | seeds: {len(seeds)} | "
              f"shown: {len(blocks)} | ~{spent} tokens")
    if more:
        header += f"\n[i] {more} more symbols matched; these are the most connected."
    if truncated:
        header += (f"\n[!] {truncated} omitted to stay under the {budget}-token budget. "
                   f"Narrow the terms or raise --budget.")
    if not blocks:
        header += "\nNo matching names. Run `graph-paat vocab` and pick terms from it."
    return header + "\n".join(blocks)
