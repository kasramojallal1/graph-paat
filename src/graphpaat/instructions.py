"""The instructions an agent reads, and putting them where it will look.

graph-paat only gets used if someone types its name. This file is what changes
that: a short block of text installed where an assistant already reads project
instructions, so it reaches for the tool on its own when a repository is too
large to read.

Two rules shape the text below.

**It costs context.** Every assistant that loads it pays for it on every turn,
so it says the minimum that makes the tool usable and stops.

**It has to say how to read the output, not just how to run it.** A map that
marks a docstring as a claim and an unresolved call as a gap is only useful to
someone told what those marks mean. Left unexplained, an agent reads a stale
comment as a verified fact -- which is the failure the marks exist to prevent.
"""
from __future__ import annotations

from pathlib import Path

BEGIN = "<!-- graph-paat:begin -->"
END = "<!-- graph-paat:end -->"

INSTRUCTIONS = """## graph-paat: navigating a large codebase

When a repository is too large to read comfortably, do not grep through it.
Build a map once and query it.

```bash
graph-paat build .          # once per repo; seconds, no network, no API key
graph-paat overview         # the main parts, and the most connected symbols
graph-paat vocab --contains auth    # names that actually exist in this graph
graph-paat query login Session      # a map of those names and how they connect
```

**Use `vocab` before `query`.** The graph knows symbol names, not English. A
question about "authentication" is a question about whatever this codebase
calls it -- `vocab` tells you, and `query` only matches names that exist.

**Pass several terms from one question together.** Terms reinforce each other:
`query QuerySet filter` finds the ORM's filter rather than an unrelated one of
the same name.

Reading the output:

- Every line gives `file:line`. Open only those files.
- `[claim]` is a docstring. It is what the code says about itself, not
  something a parser verified. Comments go stale; treat it as a lead.
- `?name` is a call the parser could not resolve, with the reason. It marks a
  real gap rather than hiding it -- so absence of an edge is not proof that
  nothing is there.
- `(hub)` marks a symbol connected to much of the codebase. The map stops
  there rather than dragging in everything behind it.

`--budget N` caps the answer size; `--per-node N` caps links shown per symbol.

The graph is written to `graph-paat-out/` beside where you run `build`. Add it
to `.gitignore`; rebuilding it takes seconds.
"""

# Where each assistant reads project instructions. Kept to files that are
# genuinely read, rather than every host a name could be invented for.
TARGETS: dict[str, str] = {
    "claude": "CLAUDE.md",
    "agents": "AGENTS.md",            # Codex, Amp, and others share this one
    "gemini": "GEMINI.md",
    "cursor": ".cursor/rules/graph-paat.mdc",
    "copilot": ".github/copilot-instructions.md",
}


def block() -> str:
    """The instructions wrapped in markers, so a reinstall can replace exactly
    what a previous one wrote and nothing else."""
    return f"{BEGIN}\n{INSTRUCTIONS.rstrip()}\n{END}\n"


def merge(existing: str, new_block: str) -> str:
    """Add or replace our block, leaving every other line untouched.

    Never rewrites a file we do not own. A user's instructions file is theirs;
    we occupy a marked region of it and nothing more.
    """
    if BEGIN in existing and END in existing:
        head, _, rest = existing.partition(BEGIN)
        _, _, tail = rest.partition(END)
        return (head.rstrip("\n") + "\n\n" + new_block + tail.lstrip("\n")).lstrip("\n")
    if not existing.strip():
        return new_block
    return existing.rstrip("\n") + "\n\n" + new_block


def install(root: Path, hosts: list[str] | None = None,
            existing_only: bool | None = None) -> list[tuple[Path, str]]:
    """Write the block into each host's instructions file under `root`.

    With no hosts named, only files that already exist are touched: creating
    five files in someone's repository because they ran one command is not a
    reasonable thing to do. Naming a host asks for that file, so it is created.

    `existing_only` overrides that default in either direction.

    Returns (path, what happened) per target.
    """
    if existing_only is None:
        existing_only = hosts is None
    chosen = hosts or list(TARGETS)
    done: list[tuple[Path, str]] = []
    for host in chosen:
        rel = TARGETS.get(host)
        if rel is None:
            done.append((root / host, "unknown host"))
            continue
        path = root / rel
        if not path.exists() and existing_only:
            done.append((path, "skipped (does not exist)"))
            continue
        before = path.read_text(encoding="utf-8") if path.exists() else ""
        after = merge(before, block())
        if before == after:
            done.append((path, "already current"))
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(after, encoding="utf-8")
        done.append((path, "updated" if BEGIN in before else
                     ("created" if not before else "added")))
    return done


def uninstall(root: Path) -> list[tuple[Path, str]]:
    """Remove our block wherever it was installed, leaving the rest intact."""
    done: list[tuple[Path, str]] = []
    for rel in TARGETS.values():
        path = root / rel
        if not path.exists():
            continue
        before = path.read_text(encoding="utf-8")
        if BEGIN not in before:
            continue
        head, _, rest = before.partition(BEGIN)
        _, _, tail = rest.partition(END)
        head, tail = head.strip("\n"), tail.strip("\n")
        # Rejoin with a blank line when there is content on both sides. Simply
        # concatenating ran the following heading onto the preceding paragraph,
        # which changes how the file renders -- removing our block should leave
        # the file as it was, not merely without us in it.
        joined = f"{head}\n\n{tail}" if head and tail else (head or tail)
        path.write_text(joined + "\n" if joined else "", encoding="utf-8")
        done.append((path, "removed"))
    return done
