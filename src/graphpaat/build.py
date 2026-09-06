"""Turn a directory into a finished graph, in one place.

Three callers need the same graph and used to each assemble it themselves: the
`build` command, the corpus runner that records baselines, and the question
runner that scores answers. Three copies of the same twenty lines is how the
thing being measured drifts away from the thing being shipped -- the score would
still be produced, and it would no longer be a score of what users run.

So the assembly lives here and reports nothing. Printing belongs to the CLI,
comparing to the runner, scoring to the question set.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .cluster import communities, god_nodes
from .ids import Collisions
from .parse import Edge, Node, parse_corpus_files
from .resolve import resolve


@dataclass
class Built:
    """One finished build. Everything any caller has ever needed from one."""
    root: Path
    nodes: list[Node]
    edges: list[Edge]
    call_edges: list[Edge]
    refusals: object                    # Counter[str]
    collisions: Collisions
    failed: list[str]
    files_parsed: int
    groups: dict | None = None
    gods: list | None = None
    grouping_error: str = ""

    def payload(self) -> dict:
        """The same shape `store.read` returns, without going through disk.

        A query scored against this is scored against exactly what a user gets
        from `graph-paat query`, which is the only reason the score means
        anything.
        """
        return {
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
            "overview": {"groups": (self.groups or {}).get("groups", []),
                         "ungrouped": (self.groups or {}).get("ungrouped", 0),
                         "god_nodes": self.gods or []},
        }


def assemble(root: Path, *, cluster: bool = True) -> Built:
    """Parse, resolve, and (unless told not to) cluster.

    Clustering is optional because it needs `networkx`. A missing grouping
    degrades the graph rather than failing the build -- the map still works
    without knowing the shape of the codebase.
    """
    parsed_files, failed = parse_corpus_files(root)

    nodes: list[Node] = []
    edges: list[Edge] = []
    collisions = Collisions()
    for parsed in parsed_files:
        for node in parsed.nodes:
            collisions.claim(node.id, f"{node.file}:L{node.line}")
            nodes.append(node)
        edges.extend(parsed.edges)

    # Resolution runs over the whole corpus at once: a call in one file usually
    # lands in another, so it cannot be done per file.
    call_edges, refusals = resolve(parsed_files)
    edges.extend(call_edges)

    built = Built(root=root, nodes=nodes, edges=edges, call_edges=call_edges,
                  refusals=refusals, collisions=collisions, failed=failed,
                  files_parsed=len(parsed_files))
    if not cluster:
        return built

    # Clustering needs the finished graph, so it runs last.
    node_dicts = [asdict(n) for n in nodes]
    edge_dicts = [asdict(e) for e in edges]
    try:
        membership, built.groups = communities(node_dicts, edge_dicts)
        built.gods = god_nodes(node_dicts, edge_dicts)
        for node in nodes:
            node.group = membership.get(node.id)
    except ImportError as exc:          # pragma: no cover - depends on install
        built.grouping_error = str(exc)
    return built
