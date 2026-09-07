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

**Why the vocabulary step is written as an order rather than a suggestion.**
Measured 2026-09-07 over 40 questions on four libraries averaging 700,000
lines: an agent that picks its search terms from the graph's own word list
first scores 10 first places against 3, and 25 in the top three against 3. That
is a larger gain than every ranking rule in `query.py` put together, and it is
free. An earlier version of this text mentioned `vocab` in one advisory line
and the step was skipped, which is exactly what the numbers above cost.

The benchmark itself stays out of the text. An assistant needs the instruction,
not the evidence for it.
"""
from __future__ import annotations

from pathlib import Path

BEGIN = "<!-- graph-paat:begin -->"
END = "<!-- graph-paat:end -->"

INSTRUCTIONS = """## graph-paat: navigating a codebase too large to read

Build a map once, then ask it questions. Do not grep a repository this size.

```bash
graph-paat build .      # once per repo; seconds, no network, no API key
graph-paat overview     # the main parts, and what everything leans on
```

### A question takes two steps. Do not skip the first.

**1. Turn the question into words this codebase uses.**

```bash
graph-paat vocab --words
```

Every word appearing in a name here, meant to be read in one go. Pick up to a
dozen that fit your question.

- **Only words on that list can be found. Do not invent one.**
- A concept with no word on the list: drop it. A remembered synonym matches
  nothing and dilutes the terms that would have worked.
- Nothing matches at all: say so and stop. A confident wrong map is worse than
  no map.

You are bridging vocabulary, not spelling -- "classified" already finds
`classify_file`. What you cannot guess is a codebase that calls authentication
`Guardian`, and only this list tells you that.

**2. Ask with those words, together.**

```bash
graph-paat query cache past key values dynamic
```

One query, not one per word. A symbol matching several of your words outranks
one matching a single word perfectly -- usually the difference between the
answer and a same-named decoy.

### Reading what comes back

- Every line gives `file:line`. Open only those files.
- `[claim]` is a docstring: what the code says about itself, not something a
  parser verified. A lead, not a fact.
- `?name` is a call that could not be resolved, with the reason. A missing edge
  is not proof that nothing is there.
- `(hub)` is wired into much of the codebase; the map stops there rather than
  dragging in everything behind it.
- Answer looks wrong? Go back to step 1 and pick different words. That is
  almost always where it went wrong.

`--budget N` caps answer size; `--per-node N` caps links per symbol. The graph
lands in `graph-paat-out/`; add it to `.gitignore` and rebuild freely.
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
