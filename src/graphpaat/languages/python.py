"""Python, read with the standard library's own parser.

No dependency and no grammar to install: `ast` ships with the interpreter, is
exact, and is the reference implementation of the language. Any other language
needs tree-sitter; Python does not, and paying for a grammar to read the
language we are written in would be silly.
"""
from __future__ import annotations

import ast

from ..ids import mint
from ..parse import DOC_SUFFIX, CallSite, Edge, Node, ParsedFile

# `.py` only. A `.pyi` stub declares the same names as the module beside it and
# `file_prefix` drops the extension, so foo.py and foo.pyi mint identical ids --
# they collide by construction. A stub also has no bodies, so nothing resolves
# from it. Adding them cost numpy 176 files of pure duplication.
EXTENSIONS = {".py"}


def parse(source: str, parsed: ParsedFile) -> bool:
    """Fill `parsed` from `source`. False means the file could not be parsed."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False
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
    return True


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
        """A docstring becomes a node explaining the thing it sits on.

        Comments are the other half of that and Python's parser discards them, so
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
        module = self._absolute_module(node.module or "", node.level)
        for alias in node.names:
            local = alias.asname or alias.name
            self.parsed.imports[local] = f"{module}.{alias.name}" if module else alias.name
        # One edge per import STATEMENT, not per imported name.
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

    def _absolute_module(self, module: str, level: int) -> str:
        """Anchor a relative import to the package doing the importing.

        `from .main import BaseModel` inside `v1/env_settings.py` means
        `v1.main`, not the top-level `main`. Ignoring the level made pydantic's
        v1 classes appear to inherit from the v2 BaseModel, because both files
        are called main.py and the top-level one matched first.

        `level` is the number of leading dots: 1 is this package, 2 is its
        parent, and so on.
        """
        if not level:
            return module
        # The file's own prefix minus its module name is the package it lives
        # in; each extra dot walks one more level up.
        package = self.parsed.prefix.split("_")[:-1]
        if level > 1:
            package = package[: -(level - 1)] or []
        return "_".join(package + ([module.replace(".", "_")] if module else []))

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


