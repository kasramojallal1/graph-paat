"""Group the graph into parts, and find the symbols everything depends on.

Two separate questions that get asked together.

**God nodes** are just a ranking by how many connections a symbol has. Cheap and
surprisingly good: on a real codebase the top of that list is the shared utility,
the router, the id minter -- genuinely the spine.

**Communities** are harder. Louvain groups nodes that connect to each other more
densely than to everything else, which finds structure the folder layout does not
show. Two things have to be handled or the result is unusable:

1. **Too many groups.** Left alone, clustering a codebase yields roughly as many
   groups as files, which is not a summary of anything. `resolution` below 1.0
   produces fewer, larger groups; the default here is tuned for "the main parts
   of a codebase", not for maximum modularity.

2. **Groups arrive unnamed.** A number is useless to a reader. Each group is
   named from the directory most of its members live in, plus its most connected
   symbol -- so `db/models · Query` rather than `community 47`.

Determinism matters more than it looks: an agent that asks the same question
twice must not get different group names. Nodes and edges are inserted in sorted
order and the seed is fixed.
"""
from __future__ import annotations

from collections import Counter
from pathlib import PurePosixPath

# Below 1.0 yields fewer, larger communities. At the library default of 1.0 a
# codebase fragments into about as many groups as it has files.
DEFAULT_RESOLUTION = 0.4


def _graph(nodes: list[dict], edges: list[dict]):
    import networkx as nx

    g = nx.Graph()
    # Sorted insertion so the partition cannot depend on dict ordering.
    g.add_nodes_from(sorted(n["id"] for n in nodes))
    weighted: Counter = Counter()
    for e in sorted(edges, key=lambda e: (e["source"], e["target"], e["relation"])):
        if not e.get("resolved", True):
            continue          # a gap is not evidence that two things belong together
        if e["source"] in g and e["target"] in g and e["source"] != e["target"]:
            weighted[(e["source"], e["target"])] += 1
    for (a, b), w in weighted.items():
        g.add_edge(a, b, weight=w)
    return g


def god_nodes(nodes: list[dict], edges: list[dict], top: int = 15) -> list[dict]:
    """The most connected symbols: what the rest of the codebase leans on."""
    degree: Counter = Counter()
    ids = {n["id"] for n in nodes}
    for e in edges:
        if not e.get("resolved", True):
            continue
        for end in (e["source"], e["target"]):
            if end in ids:
                degree[end] += 1
    by_id = {n["id"]: n for n in nodes}
    ranked = sorted(degree.items(), key=lambda kv: (-kv[1], kv[0]))
    out = []
    for nid, count in ranked:
        node = by_id[nid]
        if node["kind"] in ("file", "rationale"):
            continue          # a file is connected to everything it contains
        out.append({"id": nid, "label": node["label"], "file": node["file"],
                    "line": node["line"], "connections": count})
        if len(out) == top:
            break
    return out


def communities(nodes: list[dict], edges: list[dict],
                resolution: float = DEFAULT_RESOLUTION) -> tuple[dict[str, int], dict]:
    """Returns (node id -> group number, {"groups": [...], "ungrouped": n}).

    Raises ImportError with a usable message when networkx is absent.
    """
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "grouping needs networkx: pip install networkx") from exc

    g = _graph(nodes, edges)
    if g.number_of_edges() == 0:
        return {}, {"groups": [], "ungrouped": len(nodes)}

    groups = nx.community.louvain_communities(
        g, resolution=resolution, seed=42, weight="weight")
    # Largest first, and ties broken by name, so group numbers are stable.
    groups = sorted(groups, key=lambda s: (-len(s), sorted(s)[0]))
    # A node with no resolved edges is its own group of one. On a large corpus
    # that is hundreds of "groups" that summarise nothing, so they are reported
    # as an ungrouped count instead of padding the list.
    grouped = [s for s in groups if len(s) > 1]
    ungrouped = sum(len(s) for s in groups if len(s) <= 1)

    by_id = {n["id"]: n for n in nodes}
    degree = dict(g.degree())
    membership: dict[str, int] = {}
    summaries: list[dict] = []

    for number, members in enumerate(grouped):
        for nid in members:
            membership[nid] = number
        summaries.append(_summarise(number, members, by_id, degree))
    return membership, {"groups": summaries, "ungrouped": ungrouped}


def _summarise(number: int, members: set, by_id: dict, degree: dict) -> dict:
    real = [by_id[m] for m in members if m in by_id and by_id[m]["kind"] != "rationale"]
    # A file node is connected to everything it contains, so it wins on degree
    # while naming nothing. The hub should be a symbol someone can look up.
    symbols = [n for n in real if n["kind"] != "file"] or real
    # Where these things live. The most common directory names the group far
    # better than a number does, and it costs nothing to compute.
    folders = Counter(str(PurePosixPath(n["file"]).parent) for n in real)
    folder = folders.most_common(1)[0][0] if folders else "?"
    hub = max(symbols, key=lambda n: (degree.get(n["id"], 0), n["id"]), default=None)
    name = folder if folder not in (".", "") else ""
    if hub is not None:
        name = f"{name} · {hub['label']}" if name else hub["label"]
    return {
        "group": number,
        "name": name,
        "size": len(real),
        "hub": hub["label"] if hub else None,
        "folders": [f for f, _ in folders.most_common(3)],
    }
