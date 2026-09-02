"""Answer a question with a small map.

D9 settled how this works, and it is a two-layer design borrowed from graphify
once we understood it properly. graphify's CLI does no model call at query time
-- `serve.py` is word matching and graph scoring. The model step lives one layer
up, in their SKILL.md: *"expand the question against the graph's own vocabulary
so a wording mismatch does not collapse the answer to noise."*

So: the calling agent has a model and does the expansion. We publish the
vocabulary (`vocab`) and do the walking (`query`). Nothing here calls a model,
needs a key, or gives a different answer on two runs.
"""
from __future__ import annotations

from collections import defaultdict

# How a relation reads when you arrive from the other end. Without this the
# reverse of `contains` printed as "contains by".
INVERSE = {"contains": "part of", "calls": "called by", "imports": "imported by"}

# A token is roughly four characters of prose or code. Good enough to hold a
# budget to within a few percent, and it costs nothing to compute -- the point
# is to never blow the caller's context, not to bill them.
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def vocabulary(graph: dict) -> dict[str, list[str]]:
    """Every name in the graph, mapped to the node ids that carry it.

    This is what the agent reads to bridge English to code. Names are returned
    rather than printed so `query` can reuse the same index.
    """
    index: dict[str, list[str]] = defaultdict(list)
    for node in graph["nodes"]:
        if node["kind"] == "rationale":
            continue          # a docstring's "label" is bookkeeping, not a name
        index[node["label"]].append(node["id"])
    return index


def match(graph: dict, terms: list[str]) -> list[str]:
    """Node ids whose name matches one of the given terms.

    Exact match first, then case-insensitive, then substring. Ordered so a
    precise term is never beaten by a loose match on a longer name -- asking
    for `load` should not lead with `load_cached_semantic_entry`.
    """
    index = vocabulary(graph)
    lowered = {name.lower(): ids for name, ids in index.items()}
    seeds: list[str] = []
    seen: set[str] = set()

    def take(ids):
        for i in ids:
            if i not in seen:
                seen.add(i)
                seeds.append(i)

    for term in terms:
        if term in index:
            take(index[term])
        elif term.lower() in lowered:
            take(lowered[term.lower()])
        else:
            for name, ids in index.items():
                if term.lower() in name.lower():
                    take(ids)
    return seeds


def neighbourhood(graph: dict, seeds: list[str], depth: int = 2) -> tuple[set[str], list[dict]]:
    """Walk outward from the seeds, collecting nodes and the edges travelled.

    The edges are returned as well as the nodes, because D2 chose a map WITH
    relations. graphify's own query output is a flat list of node lines -- it
    tells you `login` and `verify` are both nearby and never that one calls the
    other. Carrying the edges is our difference, and it is only affordable
    because the budget below drops the far ones first.
    """
    adjacency: dict[str, list[dict]] = defaultdict(list)
    for edge in graph["edges"]:
        adjacency[edge["source"]].append(edge)
        adjacency[edge["target"]].append(edge)

    seen = set(seeds)
    frontier = list(seeds)
    travelled: list[dict] = []
    # An edge is reachable from both of its ends, so without this it is
    # collected twice and printed twice -- which is what the first run did.
    taken: set[tuple] = set()
    for _ in range(depth):
        nxt: list[str] = []
        for nid in frontier:
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
    return seen, travelled


def render(graph: dict, seeds: list[str], budget: int = 2000, depth: int = 2) -> str:
    """The map an agent reads. Never source code (D2) -- names, places, arrows."""
    nodes = {n["id"]: n for n in graph["nodes"]}
    found, travelled = neighbourhood(graph, seeds, depth)

    # Both directions. An edge is a fact about both of its ends: `cache.py
    # contains file_hash` is equally the answer to "where does file_hash live".
    # Showing only the outgoing direction hid that on the first run.
    links: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for edge in travelled:
        if edge["source"] in found and edge["target"] in found:
            links[edge["source"]].append((edge["relation"], edge["target"]))
            inverse = INVERSE.get(edge["relation"], f"{edge['relation']} by")
            links[edge["target"]].append((inverse, edge["source"]))

    lines: list[str] = []
    spent = 0
    truncated = 0

    for seed in seeds:
        node = nodes.get(seed)
        if node is None:
            continue
        block = [f"\n{node['label']}    {node['file']}:L{node['line']}"
                 f"    [{node['kind']}, {node['origin']}]"]
        for relation, target in links.get(seed, [])[:12]:
            t = nodes.get(target)
            if t is None:
                continue
            if t["kind"] == "rationale":
                # A docstring is a claim, not a parser fact. L3 found graphify
                # records that distinction carefully and then prints none of it,
                # so a comment reads as verified. Ours says which it is.
                snippet = " ".join((t.get("text") or "").split())[:100]
                block.append(f"    {relation:10s} [claim] \"{snippet}\"")
            else:
                block.append(f"    {relation:10s} {t['label']}    {t['file']}:L{t['line']}")
        text = "\n".join(block)
        cost = estimate_tokens(text)
        if spent + cost > budget:
            truncated += 1
            continue
        lines.append(text)
        spent += cost

    header = (f"graph: {len(graph['nodes'])} nodes | seeds: {len(seeds)} | "
              f"shown: {len(lines)} | ~{spent} tokens")
    if truncated:
        header += (f"\n[!] {truncated} matched symbols omitted to stay under the "
                   f"{budget}-token budget. Narrow the terms or raise --budget.")
    if not lines:
        header += "\nNo matching names. Run `graph-paat vocab` and pick terms from it."
    return header + "\n".join(lines)
