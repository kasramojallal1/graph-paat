"""Write the graph to disk, and read it back.

Small file, but it is a seam -- the handoff between building and querying --
and the study's flattest finding was that every problem in this kind of system
lives between stages, not inside one. So the shape written here is the contract,
and it is stated in one place.

The one deliberate difference from graphify: **the artifact records what the
run failed to do.** Their `graph.json` carries `nodes`, `links`, `hyperedges`
and a commit hash -- nothing else. They compute `failed_sources`, use it
internally in the shrink guard, and drop it (L3). Anyone reading the file a week
later sees a graph that looks complete. Ours carries a `coverage` block, so a
gap survives being scrolled past.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .parse import Edge, Node

OUT_DIR = "graph-paat-out"
GRAPH_FILE = "graph.json"
SCHEMA = 1      # bump when the shape below changes; readers check it


def out_path(root: Path, out: Path | None = None) -> Path:
    """Where the graph lives.

    Default is `graph-paat-out/` in the CURRENT directory, **not** inside the
    corpus. This is graphify's #1774, which we reproduced on the first run: the
    output is an artifact, and writing it into the tree being analysed pollutes
    a repo the user may not own or may have checked out read-only. Our own
    reference clone of graphify is exactly that.

    `out` overrides it, so one machine can graph several corpora side by side.
    """
    base = Path(out) if out is not None else Path.cwd() / OUT_DIR
    return base.resolve() / GRAPH_FILE


def write(root: Path, nodes: list[Node], edges: list[Edge],
          collisions, failed: list[str], out: Path | None = None,
          groups: dict | None = None, gods: list | None = None) -> Path:
    path = out_path(root, out)
    path.parent.mkdir(parents=True, exist_ok=True)

    collided = collisions.collided()
    payload = {
        "schema": SCHEMA,
        "corpus": str(root.resolve()),
        # The shape of the codebase, computed once at build time so a query
        # never has to cluster 50,000 nodes to answer one question.
        "overview": {"groups": (groups or {}).get("groups", []),
                     "ungrouped": (groups or {}).get("ungrouped", 0),
                     "god_nodes": gods or []},
        "nodes": [asdict(n) for n in nodes],
        "edges": [asdict(e) for e in edges],
        # Everything the run could not do, in the artifact rather than the
        # terminal. This block is the point of the file's docstring.
        "coverage": {
            "files_parsed": len({n.file for n in nodes}),
            "files_failed": failed,
            "nodes_total": len(nodes),
            "ids_unique": len({n.id for n in nodes}),
            "collisions": {nid: where for nid, where in collided.items()},
            "unresolved_edges": sum(1 for e in edges if not e.resolved),
        },
    }
    # Written whole then moved, so an interrupted run cannot leave a half-file
    # that every later read fails on -- graphify's #2405 is exactly that bug in
    # their cache, where a corrupt entry re-extracts forever.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(path)
    return path


def read(root: Path, out: Path | None = None) -> dict:
    path = out_path(root, out)
    if not path.exists():
        raise FileNotFoundError(
            f"no graph at {path} - run `graph-paat build {root}` first")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise ValueError(
            f"graph at {path} uses schema {data.get('schema')}, this build expects "
            f"{SCHEMA} - rebuild it")
    return data
