"""Answer a question with a small map.

The calling agent has a language model; this does not. It publishes the graph's
vocabulary (`vocab`) so the agent can turn a question into names, then matches
those names, walks outward, and renders what it found inside a token budget.
Nothing here calls a model, needs a key, or answers differently on two runs.

Three things separate a useful map from a dump, and each was added because the
naive version failed visibly on a real repository:

**Seeds must be ranked.** Asking Django for `get` matches 30 symbols. Returning
them in whatever order they were found spends the whole budget on an arbitrary
one. Exact matches beat loose ones, and within a tier the better-connected
symbol wins, because a name that many things use is usually the one being asked
about.

**Links must be ranked.** A class's map was twelve dunder methods and nothing
about what used it. Behaviour is more interesting than containment, and
`__deepcopy__` is less interesting than either.

**The walk must avoid hubs.** Two hops from something adjacent to a god node
reaches most of the codebase. Expanding *through* a hub is what does it, so we
report hubs and stop there.
"""
from __future__ import annotations

from collections import defaultdict

# How a relation reads when you arrive from the other end.
INVERSE = {"contains": "part of", "calls": "called by", "imports": "imported by",
           "rationale_for": "explains"}

# Most interesting first. What a thing DOES beats what it holds; a docstring is
# context rather than structure, so it comes last but is never dropped -- it is
# often the only statement of intent in the whole map.
RELATION_ORDER = {"calls": 0, "called by": 1, "imports": 2, "imported by": 3,
                  "part of": 4, "contains": 5, "rationale_for": 6, "explains": 6}

# Words that describe the RELATION being asked about, not a symbol to look for.
# "what calls login" must not seed on `calls`, which is a method name in most
# codebases and would seat an unrelated root.
INTENT_WORDS = {"call", "calls", "called", "use", "uses", "using", "import",
                "imports", "contain", "contains", "define", "defines", "where",
                "what", "which", "does", "the", "and", "for", "from", "with"}

# A node connected to more than this is a hub. We show it and do not expand
# through it: two hops through Django's ValidationError reaches half the repo.
HUB_DEGREE = 40

CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


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


def match(graph: dict, terms: list[str], limit: int = 6) -> tuple[list[str], int]:
    """Rank nodes matching the terms. Returns (seeds, how many more matched).

    Tiers: exact name, then case-insensitive, then substring. Within a tier the
    better-connected node comes first -- when a name is used in thirty places,
    the one the rest of the codebase leans on is the likely subject.
    """
    index = vocabulary(graph)
    lowered: dict[str, list[str]] = defaultdict(list)
    for name, ids in index.items():
        lowered[name.lower()].extend(ids)
    degree = _degrees(graph)

    real_terms = [t for t in terms if t.lower() not in INTENT_WORDS] or terms
    tiers: dict[str, int] = {}

    def offer(ids, tier):
        for i in ids:
            if tier < tiers.get(i, 99):
                tiers[i] = tier

    for term in real_terms:
        if term in index:
            offer(index[term], 0)
        if term.lower() in lowered:
            offer(lowered[term.lower()], 1)
        needle = term.lower()
        for name, ids in index.items():
            if needle in name.lower():
                offer(ids, 2)

    ranked = sorted(tiers, key=lambda i: (tiers[i], -degree.get(i, 0), i))
    return ranked[:limit], max(0, len(ranked) - limit)


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
    group_names = {g["group"]: g["name"]
                   for g in graph.get("overview", {}).get("groups", [])}
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
