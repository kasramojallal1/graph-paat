"""The data every language produces, and the machinery around it.

A node, an edge, a call site, one file's worth of them. Nothing here knows
about any particular language: each language lives in `languages/` and hands
back these same shapes, so resolution, storage and query never learn a second
vocabulary.

`DOC_SUFFIX` is "#doc" rather than "__doc" because `#` cannot appear in an
identifier. With an underscore the suffix was ambiguous against real code: a
module docstring minted `<prefix>__doc`, and a function genuinely named `_doc`
minted the same string.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .ids import Collisions, file_prefix

DOC_SUFFIX = "#doc"

# Directories that are never source. Kept small on purpose: a long list here
# silently shrinks the corpus, and the whole stance is that invisible loss is
# the enemy.
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".tox",
             "build", "dist"}


@dataclass
class Node:
    id: str
    label: str
    kind: str            # file | class | function | method | rationale
                         #   | document | claim   (the --deep lane)
    file: str            # relative to the corpus root
    line: int
    # Provenance is recorded at creation, never inherited. "ast" is something a
    # parser verified this build; "doc" is a sentence a person wrote, which may
    # have been true once. The difference is visible on every line of every
    # answer -- see `query.provenance`.
    origin: str = "ast"
    text: str | None = None   # rationale and claim nodes carry their prose
    group: int | None = None  # which community, filled in after clustering
    bases: list[str] | None = None   # class nodes: what it inherits from


@dataclass
class CallSite:
    """A call we saw but have not resolved yet.

    Extraction records what it SAW; resolution decides what it MEANS. Keeping
    them apart is deliberate: every hard bug in this kind of system has lived
    at the handoff between stages, and a handoff you can print is one you can
    debug.
    """
    caller: str                     # node id of the function doing the calling
    file: str
    line: int
    name: str                       # the name being called: foo, or the .foo part
    receiver: str | None = None     # x in x.foo()
    receiver_is_self: bool = False  # True for self.x.foo()
    on_self: bool = False           # True for self.foo()


@dataclass
class Edge:
    source: str
    target: str
    relation: str        # contains | calls | imports
    file: str
    line: int
    origin: str = "ast"
    resolved: bool = True
    reason: str | None = None   # why an unresolved edge could not be resolved


@dataclass
class ParsedFile:
    path: str
    prefix: str
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    calls: list[CallSite] = field(default_factory=list)
    imports: dict[str, str] = field(default_factory=dict)      # local name -> dotted path
    var_types: dict[str, str] = field(default_factory=dict)    # "scope::var" -> ClassName
    attr_types: dict[str, str] = field(default_factory=dict)   # "Class::attr" -> ClassName
    owner_of: dict[str, str] = field(default_factory=dict)     # method id -> class name
    defined_classes: set = field(default_factory=set)
    import_sites: list = field(default_factory=list)   # (dotted module, line)



def collect(root: Path) -> list[Path]:
    """Every source file under root in a language we can read, minus the noise."""
    from .languages import EXTENSIONS
    out = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in EXTENSIONS or not path.is_file():
            continue
        # A declaration file restates the names of the module beside it, and
        # file_prefix drops the extension -- so foo.ts and foo.d.ts would mint
        # identical ids and collide by construction. Same reason .pyi is out.
        if path.name.endswith((".d.ts", ".d.mts", ".d.cts")):
            continue
        parts = path.relative_to(root).parts[:-1]
        if any(d in SKIP_DIRS or d.startswith(".") for d in parts):
            continue
        out.append(path)
    return out


def parse_file(path: Path, root: Path) -> ParsedFile | None:
    """Parse one file with whichever language owns its extension.

    Returns None when the file cannot be read or parsed. A None is a real loss
    and the caller counts it: a gap the user cannot see is worse than one they
    can.
    """
    from .languages import for_path
    language = for_path(path)
    if language is None:
        return None
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    # A language may state that the extension is part of a file's identity --
    # C's `url.c` and `url.h` are two files, not one. The default is off and
    # nothing here knows which languages set it.
    parsed = ParsedFile(
        path=str(path.relative_to(root)),
        prefix=file_prefix(path, root,
                           keep_extension=getattr(language, "KEEP_EXTENSION", False)))
    # A node for the file itself, so "what is in this file" is a graph question
    # rather than a filesystem one.
    parsed.nodes.append(Node(
        id=parsed.prefix, label=path.name, kind="file",
        file=parsed.path, line=1,
    ))
    if language.parse(source, parsed) is False:
        return None
    return parsed


def parse_corpus_files(root: Path) -> tuple[list[ParsedFile], list[str]]:
    """Parse every file, returning the per-file results untouched.

    Resolution needs all files before it can start -- a call in one file often
    lands in another -- so the corpus is assembled first and resolved second.
    """
    root = root.resolve()
    parsed_files: list[ParsedFile] = []
    failed: list[str] = []
    for path in collect(root):
        try:
            parsed = parse_file(path, root)
        except RecursionError:
            # A single deeply nested expression -- a generated table, a long
            # chain of binary operators -- exhausts the interpreter stack while
            # walking the tree. Python's own parser produced the tree happily;
            # it is the walk that cannot finish.
            #
            # Found on sympy, where it killed a 1,532-file build outright. That
            # is the worst failure this tool can have: it exists for repositories
            # too large to read, and one file made all of them unreadable. A file
            # we cannot walk is a gap to report, exactly like one we cannot parse.
            parsed = None
        if parsed is None:
            failed.append(str(path.relative_to(root)))
            continue
        parsed_files.append(parsed)
    return parsed_files, failed


def parse_corpus(root: Path) -> tuple[list[Node], list[Edge], Collisions, list[str]]:
    """Parse every file under root, flattened."""
    parsed_files, failed = parse_corpus_files(root)
    nodes: list[Node] = []
    edges: list[Edge] = []
    collisions = Collisions()
    for parsed in parsed_files:
        for node in parsed.nodes:
            collisions.claim(node.id, f"{node.file}:L{node.line}")
            nodes.append(node)
        edges.extend(parsed.edges)
    return nodes, edges, collisions, failed
