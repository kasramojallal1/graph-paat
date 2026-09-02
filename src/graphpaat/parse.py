"""Turn a folder of Python into nodes.

`DOC_SUFFIX` is "#doc" rather than "__doc" because `#` cannot appear in a Python
identifier. With an underscore the suffix was ambiguous against real code: a
module docstring minted `<prefix>__doc`, and a function genuinely named `_doc`
minted the same string. Found in graphify's own tests/test_partial_cache.py,
which has both.

Step 1 of the thin loop. Reads files, parses them, mints ids. It deliberately
does NOT resolve calls or build a graph -- those are later steps, and keeping
them out means the seam between parsing and resolving stays visible rather than
being smeared across one big function.

Python's own `ast` module does the parsing (D8: Python only, no dependency).
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from .ids import Collisions, file_prefix, mint

# Directories that are never source. Kept small on purpose: a long list here
# silently shrinks the corpus, and D6's whole stance is that invisible loss is
# the enemy.
DOC_SUFFIX = "#doc"

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".tox", "build", "dist"}


@dataclass
class Node:
    id: str
    label: str
    kind: str            # file | class | function | method | rationale
    file: str            # relative to the corpus root
    line: int
    origin: str = "ast"  # provenance is recorded at creation, never inherited
    text: str | None = None   # rationale nodes carry their docstring
    group: int | None = None  # which community, filled in after clustering
    bases: list[str] | None = None   # class nodes: what it inherits from


@dataclass
class CallSite:
    """A call we saw but have not resolved yet.

    Extraction records what it SAW; resolution decides what it MEANS. Keeping
    them apart is deliberate: the study found every problem in this kind of
    system lives at the handoff between stages, and a handoff you can print is
    one you can debug.
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
    """Every .py file under root, minus the obvious noise."""
    out = []
    for p in sorted(root.rglob("*.py")):
        parts = p.relative_to(root).parts[:-1]
        if any(d in SKIP_DIRS or d.startswith(".") for d in parts):
            continue
        out.append(p)
    return out


class _Walker(ast.NodeVisitor):
    """Walks one file's syntax tree, emitting a node per definition.

    Tracks the enclosing class so a method's id can be qualified by it -- the
    fourth id rule, and the one that prevents two `__init__`s colliding.
    """

    def __init__(self, parsed: ParsedFile) -> None:
        self.parsed = parsed
        self.scope: list[str] = []          # enclosing classes and functions
        self.in_class = False               # is the immediate parent a class?
        # id of the thing we are currently inside, so `contains` can be emitted
        # without a second pass. The file node is the outermost container.
        self.container: list[str] = [parsed.prefix]

    def _emit(self, label: str, kind: str, node: ast.AST,
              bases: list[str] | None = None) -> str:
        nid = mint(self.parsed.prefix, label, self.scope)
        self.parsed.edges.append(Edge(
            source=self.container[-1], target=nid, relation="contains",
            file=self.parsed.path, line=node.lineno,
        ))
        self.parsed.nodes.append(Node(
            id=nid, label=label, kind=kind,
            file=self.parsed.path, line=node.lineno, bases=bases,
        ))
        return nid

    def _emit_docstring(self, owner_id: str, node: ast.AST) -> None:
        """D5: a docstring becomes a node explaining the thing it sits on.

        Comments are the other half of D5 and Python's parser discards them, so
        they need `tokenize` and are deliberately left to a later step.
        """
        doc = ast.get_docstring(node)
        if not doc:
            return
        doc_id = f"{owner_id}{DOC_SUFFIX}"
        self.parsed.nodes.append(Node(
            id=doc_id, label=f"docstring of {owner_id}",
            kind="rationale", file=self.parsed.path, line=node.lineno,
            text=doc.strip(),
        ))
        self.parsed.edges.append(Edge(
            source=owner_id, target=doc_id, relation="rationale_for",
            file=self.parsed.path, line=node.lineno,
        ))

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.parsed.defined_classes.add(node.name)
        nid = self._emit(node.name, "class", node, bases=_base_names(node))
        self._emit_docstring(nid, node)
        self.scope.append(node.name)
        self.container.append(nid)
        was_in_class, self.in_class = self.in_class, True
        self.generic_visit(node)
        self.in_class = was_in_class
        self.container.pop()
        self.scope.pop()

    def _visit_callable(self, node) -> None:
        # A property's setter and deleter are not separate entities. Callers
        # write `obj.choices` for both, so the pair is one node -- the getter.
        # Emitting both minted one id twice and reported it as a lost symbol,
        # which was 100 false alarms on Django. Skipping them is not data loss;
        # treating them as distinct was the error.
        if _is_property_accessor(node):
            self.scope.append(node.name)
            was_in_class, self.in_class = self.in_class, False
            self.generic_visit(node)
            self.in_class = was_in_class
            self.scope.pop()
            return
        # "method" only when the immediate parent is a class. A function nested
        # inside a method is a nested function, not another method.
        kind = "method" if self.in_class else "function"
        nid = self._emit(node.name, kind, node)
        owner = self._enclosing_class()
        if owner:
            self.parsed.owner_of[nid] = owner
        self._emit_docstring(nid, node)
        self.scope.append(node.name)
        self.container.append(nid)
        was_in_class, self.in_class = self.in_class, False
        self.generic_visit(node)
        self.in_class = was_in_class
        self.container.pop()
        self.scope.pop()

    visit_FunctionDef = _visit_callable
    visit_AsyncFunctionDef = _visit_callable

    # ---- evidence for resolution -------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            local = alias.asname or alias.name.split(".")[0]
            self.parsed.imports[local] = alias.name
            self.parsed.import_sites.append((alias.name, node.lineno))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            local = alias.asname or alias.name
            self.parsed.imports[local] = f"{module}.{alias.name}" if module else alias.name
        # One edge per import STATEMENT, not per imported name. graphify splits
        # `import x` and `from x import y` into two relations; our own S1 notes
        # flagged that as duplicating one fact, so we keep a single `imports`.
        if module:
            self.parsed.import_sites.append((module, node.lineno))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        cls = _constructed_class(node.value)
        if cls:
            for target in node.targets:
                self._record_type(target, cls)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # An annotation is free type information and more reliable than an
        # inferred one: `db: Database` is the author telling us directly.
        cls = _annotated_class(node.annotation) or _constructed_class(node.value)
        if cls:
            self._record_type(node.target, cls)
        self.generic_visit(node)

    def visit_arguments(self, node: ast.arguments) -> None:
        args = list(node.args) + list(node.posonlyargs) + list(node.kwonlyargs)
        for arg in args:
            cls = _annotated_class(arg.annotation) if arg.annotation else None
            if cls:
                self.parsed.var_types[f"{self._scope_key()}::{arg.arg}"] = cls
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        caller = self.container[-1]
        func = node.func
        if isinstance(func, ast.Name):
            self.parsed.calls.append(CallSite(
                caller=caller, file=self.parsed.path, line=node.lineno, name=func.id))
        elif isinstance(func, ast.Attribute):
            receiver, is_self, on_self = _receiver(func.value)
            self.parsed.calls.append(CallSite(
                caller=caller, file=self.parsed.path, line=node.lineno,
                name=func.attr, receiver=receiver,
                receiver_is_self=is_self, on_self=on_self))
        self.generic_visit(node)

    # ---- helpers -----------------------------------------------------

    def _scope_key(self) -> str:
        return ".".join(self.scope)

    def _record_type(self, target, cls: str) -> None:
        if isinstance(target, ast.Name):
            self.parsed.var_types[f"{self._scope_key()}::{target.id}"] = cls
        elif (isinstance(target, ast.Attribute)
              and isinstance(target.value, ast.Name)
              and target.value.id == "self"):
            # self.db = Database() -- a class-level attribute, usable from any
            # method of that class, so it is keyed by class not by scope.
            owner = self._enclosing_class()
            if owner:
                self.parsed.attr_types[f"{owner}::{target.attr}"] = cls

    def _enclosing_class(self) -> str | None:
        for name in reversed(self.scope):
            if name in self.parsed.defined_classes:
                return name
        return None


def _base_names(node: ast.ClassDef) -> list[str]:
    """What a class inherits from, by name.

    Recorded because a name alone cannot tell you what a class IS. Django's
    `ImproperlyConfigured` does not end in Error, but it inherits from
    Exception -- and an exception is connected to everything that raises it,
    which makes it a bad name for the region it happens to hub.
    """
    names = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            names.append(base.attr)
    return names


def _is_property_accessor(node) -> bool:
    """True for `@choices.setter` / `@x.deleter` -- the non-getter half of a
    property. The getter, decorated with plain `@property`, is the node."""
    for dec in getattr(node, "decorator_list", []):
        if isinstance(dec, ast.Attribute) and dec.attr in ("setter", "deleter"):
            return True
    return False


def _constructed_class(value) -> str | None:
    """`Database()` -> "Database". `mod.Database()` -> "Database".

    The capital-letter test is a convention, not a rule -- Python has no way to
    know a callable is a constructor. It is right nearly always and wrong for a
    factory function named like a class, which is a tradeoff worth stating.
    """
    if isinstance(value, ast.Call):
        func = value.func
        if isinstance(func, ast.Name) and func.id[:1].isupper():
            return func.id
        if isinstance(func, ast.Attribute) and func.attr[:1].isupper():
            return func.attr
    return None


def _annotated_class(annotation) -> str | None:
    if isinstance(annotation, ast.Name) and annotation.id[:1].isupper():
        return annotation.id
    if isinstance(annotation, ast.Attribute) and annotation.attr[:1].isupper():
        return annotation.attr
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        head = annotation.value.split("[")[0].strip()
        return head if head[:1].isupper() else None
    return None


def _receiver(value) -> tuple[str | None, bool, bool]:
    """Who is being called on. Returns (name, is_self_attribute, is_self)."""
    if isinstance(value, ast.Name):
        return (value.id, False, value.id == "self")
    if (isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name)
            and value.value.id == "self"):
        return (value.attr, True, False)
    return (None, False, False)


def parse_file(path: Path, root: Path) -> ParsedFile | None:
    """Parse one file. Returns None if it cannot be read or parsed.

    A None here is a real loss and the caller is expected to count it -- D6:
    a gap the user cannot see is worse than one they can.
    """
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except (SyntaxError, OSError, ValueError):
        return None

    parsed = ParsedFile(path=str(path.relative_to(root)), prefix=file_prefix(path, root))
    # One node for the file itself, so "what is in this file" is a graph
    # question rather than a filesystem one.
    parsed.nodes.append(Node(
        id=parsed.prefix, label=path.name, kind="file",
        file=parsed.path, line=1,
    ))
    module_doc = ast.get_docstring(tree)
    if module_doc:
        parsed.nodes.append(Node(
            id=f"{parsed.prefix}{DOC_SUFFIX}", label=f"docstring of {parsed.prefix}",
            kind="rationale", file=parsed.path, line=1, text=module_doc.strip(),
        ))
        parsed.edges.append(Edge(
            source=parsed.prefix, target=f"{parsed.prefix}{DOC_SUFFIX}",
            relation="rationale_for", file=parsed.path, line=1,
        ))
    _Walker(parsed).visit(tree)
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
        parsed = parse_file(path, root)
        if parsed is None:
            failed.append(str(path.relative_to(root)))
            continue
        parsed_files.append(parsed)
    return parsed_files, failed


def parse_corpus(root: Path) -> tuple[list[Node], list[Edge], Collisions, list[str]]:
    """Parse every file under root.

    Returns nodes, edges, the collision record, and the files that failed. All
    four are returned rather than printed: L3 found that graphify computes
    `failed_sources` and then never writes it anywhere the user can see it.
    """
    root = root.resolve()
    nodes: list[Node] = []
    edges: list[Edge] = []
    collisions = Collisions()
    failed: list[str] = []

    for path in collect(root):
        parsed = parse_file(path, root)
        if parsed is None:
            failed.append(str(path.relative_to(root)))
            continue
        for node in parsed.nodes:
            collisions.claim(node.id, f"{node.file}:L{node.line}")
            nodes.append(node)
        edges.extend(parsed.edges)
    return nodes, edges, collisions, failed
