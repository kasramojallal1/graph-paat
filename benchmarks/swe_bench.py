"""Score graph-paat against SWE-bench Lite.

SWE-bench Lite is 300 GitHub issues from twelve Python projects, each paired
with the patch that closed it. Nobody here wrote the questions: each one is an
issue title typed by a developer years ago, and the right answer is not an
opinion, it is the file the maintainers changed. Every instance touches exactly
one file, and the patch's hunk headers name the functions it changed.

    python benchmarks/swe_bench.py fetch
        Pull the 300 instances from HuggingFace and write `swe_bench_lite.json`
        next to this script. Needs the network once; the file ships with the
        repository so that scoring does not.

    python benchmarks/swe_bench.py score django /path/to/site-packages/django
        Build the graph of that package, ask every issue in the set that
        belongs to it, and print how often the right file -- and the right
        function -- is in the top three.

    python benchmarks/swe_bench.py score django /path/to/django \\
        --graphify /path/to/graphify-out/graph.json
        The same issues put to a graphify build of the same package through
        `graphify query`, so the two columns are scored identically.

The question is the issue's first line. `--full` asks with the whole issue
instead -- stack traces and all -- which is what an agent pasting a ticket
would do, and scores worse.

The instances are the test split of `princeton-nlp/SWE-bench_Lite` (MIT), kept
in full so `--full` works offline.

An issue whose file is not under the package root is skipped and counted: the
installed version has moved it. The package root is the directory the graph is
built from, so the gold path is trimmed from the left until it exists there.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
INSTANCES = HERE / "swe_bench_lite.json"
DATASET = "princeton-nlp/SWE-bench_Lite"
ROWS_URL = ("https://datasets-server.huggingface.co/rows"
            f"?dataset={DATASET}&config=default&split=test")
TOP_N = 3

_FILE = re.compile(r"^diff --git a/(\S+) b/", re.M)
_HUNK_DEF = re.compile(r"^@@[^@\n]*@@\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)", re.M)
# What `graphify query` prints for each result.
_GRAPHIFY_NODE = re.compile(r"^NODE (.+?) \[src=(\S+) loc=L(\d+)")


def fetch() -> int:
    rows: list[dict] = []
    offset = 0
    while True:
        req = urllib.request.Request(f"{ROWS_URL}&offset={offset}&length=100",
                                     headers={"User-Agent": "graph-paat-benchmark"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            page = json.load(resp)
        rows.extend(r["row"] for r in page["rows"])
        offset += 100
        if offset >= page["num_rows_total"]:
            break

    out = []
    for r in rows:
        files = _FILE.findall(r["patch"])
        if len(set(files)) != 1:
            continue  # Lite promises one file per patch; hold it to that
        out.append({
            "repo": r["repo"],
            "id": r["instance_id"],
            "commit": r["base_commit"],
            "file": files[0],
            "funcs": sorted(set(_HUNK_DEF.findall(r["patch"]))),
            "problem": r["problem_statement"],
        })
    INSTANCES.write_text(json.dumps(out, indent=1))
    print(f"{len(out)} instances from {len(rows)} rows -> {INSTANCES}")
    return 0


def _gold_path(root: Path, patch_path: str) -> str | None:
    """The gold file relative to the package root, or None if it is not there."""
    parts = patch_path.split("/")
    for i in range(len(parts)):
        rel = "/".join(parts[i:])
        if (root / rel).is_file():
            return rel
    return None


def _ours(root: Path):
    sys.path.insert(0, str(HERE.parent / "src"))
    from graphpaat.build import assemble
    from graphpaat.query import match

    graph = assemble(root).payload()
    node_of = {n["id"]: n for n in graph["nodes"]}

    def ask(text: str) -> list[tuple[str, str]]:
        seeds, _ = match(graph, text.split(), limit=TOP_N)
        return [(node_of[s]["file"], node_of[s]["label"]) for s in seeds if s in node_of]
    return ask


def _theirs(graph_json: Path):
    exe = shutil.which("graphify")
    if exe is None:
        sys.exit("`graphify` is not on PATH; install it to score that column")

    def ask(text: str) -> list[tuple[str, str]]:
        result = subprocess.run([exe, "query", text, "--graph", str(graph_json)],
                                capture_output=True, text=True)
        found = []
        for line in result.stdout.splitlines():
            hit = _GRAPHIFY_NODE.match(line)
            if hit:
                # Their labels carry syntax -- `get()`, `.send()` -- ours do not.
                label = hit.group(1).strip().lstrip(".").removesuffix("()")
                found.append((hit.group(2), label))
            if len(found) >= TOP_N:
                break
        return found
    return ask


def score(repo: str, root: Path, graphify: Path | None, full: bool) -> int:
    if not INSTANCES.exists():
        sys.exit(f"{INSTANCES} is missing; run `fetch` first")
    items = [o for o in json.loads(INSTANCES.read_text()) if repo in o["repo"]]
    if not items:
        sys.exit(f"no instances match {repo!r}")
    root = root.resolve()

    usable, skipped = [], 0
    for o in items:
        rel = _gold_path(root, o["file"])
        if rel is None:
            skipped += 1
            continue
        usable.append((o, rel))

    ask = _theirs(graphify) if graphify else _ours(root)
    who = "graphify" if graphify else "graph-paat"

    file_hits = symbol_hits = 0
    for o, rel in usable:
        text = o["problem"] if full else o["problem"].splitlines()[0]
        hits = ask(text)
        in_file = [label for path, label in hits if path == rel or path.endswith("/" + rel)]
        file_hits += bool(in_file)
        symbol_hits += any(label in o["funcs"] for label in in_file)

    n = len(usable)
    print(f"{items[0]['repo']}  {n} issues"
          + (f"  ({skipped} skipped: file not in this install)" if skipped else ""))
    print(f"  {who:10s} asked with {'the whole issue' if full else 'the title'}:"
          f"  right file in top {TOP_N}: {file_hits:3d}/{n} ({100 * file_hits // n}%)"
          f"   right function: {symbol_hits:3d}/{n} ({100 * symbol_hits // n}%)")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="download the instances")
    s = sub.add_parser("score", help="score one package")
    s.add_argument("repo", help="which project's issues, e.g. django or sympy")
    s.add_argument("root", type=Path, help="the package directory to build from")
    s.add_argument("--graphify", type=Path, metavar="GRAPH_JSON",
                   help="score a graphify build of the same package instead")
    s.add_argument("--full", action="store_true", help="ask with the whole issue text")
    a = p.parse_args(argv)
    if a.cmd == "fetch":
        return fetch()
    return score(a.repo, a.root, a.graphify, a.full)


if __name__ == "__main__":
    sys.exit(main())
