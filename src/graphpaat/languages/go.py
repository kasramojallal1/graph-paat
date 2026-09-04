"""Go, read with tree-sitter.

Go maps onto the same shapes as Python, and one difference makes it easier
rather than harder: **a method's receiver is declared.** `func (s *Server)
Start()` states that `s` is a `Server`, so every `s.foo()` inside resolves
exactly. In Python the same fact has to be inferred from an assignment and is
wrong more often than not.

Vocabulary is deliberately shared rather than extended. A struct and an
interface both become a `class` node; an embedded type becomes `inherits`.
Grouping, naming, ranking and query never learn that Go exists -- which is the
test of whether the model was actually language-neutral.

A doc comment is the run of `//` lines directly above a declaration, which is
Go's docstring. Those become `rationale` nodes, marked as claims like any other.
"""
from __future__ import annotations

from ..ids import mint
from ..parse import DOC_SUFFIX, CallSite, Edge, Node, ParsedFile

EXTENSIONS = {".go"}
WHY = ""

_parser = None


def available() -> bool:
    """True when the grammar is installed. The registry skips us otherwise, so
    a Python-only user never has to carry a Go grammar."""
    global _parser, WHY
    if _parser is not None:
        return True
    try:
        import tree_sitter_go
        from tree_sitter import Language, Parser
        _parser = Parser(Language(tree_sitter_go.language()))
        return True
    except Exception as exc:                      # pragma: no cover
        WHY = f"needs tree-sitter and tree-sitter-go ({exc})"
        return False


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _named(node, *types):
    """First direct child of any of these types."""
    for child in node.children:
        if child.type in types:
            return child
    return None


def _type_name(node, src: bytes) -> str | None:
    """The bare name of a type, through pointers, slices and generics.

    `*Server`, `[]Server` and `Server[T]` are all the Server type for the
    purpose of knowing what a receiver is.
    """
    if node is None:
        return None
    if node.type == "type_identifier":
        return _text(node, src)
    if node.type == "qualified_type":
        ident = _named(node, "type_identifier")
        return _text(ident, src) if ident else None
    for child in node.children:
        found = _type_name(child, src)
        if found:
            return found
    return None


def _doc_above(node, src: bytes) -> str | None:
    """The run of `//` comment lines directly above a declaration.

    Go has no docstring; this is its equivalent, and skipping it would leave
    every Go node without the "why" that Python nodes carry.
    """
    lines: list[str] = []
    prev = node.prev_sibling
    while prev is not None and prev.type == "comment":
        # Only a comment on the line immediately above is documentation; one
        # separated by a blank line is a note about something else. The check
        # has to apply to the first comment too -- guarding it on "we already
        # have lines" let any single detached comment through.
        if prev.end_point[0] + 1 < node.start_point[0]:
            break
        lines.insert(0, _text(prev, src).lstrip("/").lstrip("*").strip())
        node, prev = prev, prev.prev_sibling
    joined = " ".join(l for l in lines if l).strip()
    return joined or None


class _Reader:
    def __init__(self, parsed: ParsedFile, src: bytes):
        self.p = parsed
        self.src = src

    def emit(self, name: str, kind: str, node, scope=None, bases=None) -> str:
        nid = mint(self.p.prefix, name, scope or [])
        line = node.start_point[0] + 1
        self.p.nodes.append(Node(id=nid, label=name, kind=kind, file=self.p.path,
                                 line=line, bases=bases))
        container = mint(self.p.prefix, scope[0]) if scope else self.p.prefix
        self.p.edges.append(Edge(source=container, target=nid, relation="contains",
                                 file=self.p.path, line=line))
        doc = _doc_above(node, self.src)
        if doc:
            doc_id = f"{nid}{DOC_SUFFIX}"
            self.p.nodes.append(Node(id=doc_id, label=f"docstring of {nid}",
                                     kind="rationale", file=self.p.path, line=line,
                                     text=doc[:600]))
            self.p.edges.append(Edge(source=nid, target=doc_id,
                                     relation="rationale_for",
                                     file=self.p.path, line=line))
        return nid

    def local_types(self, node, scope: str, known: dict[str, str]) -> None:
        """Record what every local variable in a body IS.

        Go states this outright, in three forms, and all three are declarations
        rather than guesses:

            func f(srv *Server)   a typed parameter
            var buf Builder       an explicit declaration
            c := Client{}         a composite literal, with or without &

        This is the whole reason Go resolves better than Python here. Python's
        equivalent -- `db = Database()` -- has to be inferred and is wrong
        whenever a factory function is involved.
        """
        stack = list(node.children)
        while stack:
            n = stack.pop()
            if n.type == "parameter_declaration":
                name = _named(n, "identifier")
                found = _type_name(_last_type(n), self.src)
                if name is not None and found:
                    known[_text(name, self.src)] = found
            elif n.type == "var_spec":
                name = _named(n, "identifier")
                found = _type_name(_last_type(n), self.src)
                if name is not None and found:
                    known[_text(name, self.src)] = found
            elif n.type == "short_var_declaration":
                lhs = n.children[0] if n.children else None
                rhs = n.children[-1] if n.children else None
                name = _named(lhs, "identifier") if lhs is not None else None
                lit = _first_deep(rhs, "composite_literal") if rhs is not None else None
                found = _type_name(lit, self.src) if lit is not None else None
                if name is not None and found:
                    known[_text(name, self.src)] = found
            stack.extend(n.children)
        for var, cls in known.items():
            self.p.var_types[f"{scope}::{var}"] = cls

    def calls_in(self, body, caller: str, receiver_names: set[str]) -> None:
        """Every call inside a body, with what it was called on.

        `receiver_names` is ONLY the method's own receiver -- the `s` in
        `func (s *Server)`. Passing every typed local made `srv.Ping()` inside a
        plain function look like a call on self, which then failed because the
        function has no class.
        """
        stack = [body]
        while stack:
            node = stack.pop()
            if node.type == "call_expression":
                fn = node.children[0] if node.children else None
                if fn is not None and fn.type == "identifier":
                    self.p.calls.append(CallSite(
                        caller=caller, file=self.p.path,
                        line=node.start_point[0] + 1, name=_text(fn, self.src)))
                elif fn is not None and fn.type == "selector_expression":
                    who = _named(fn, "identifier")
                    what = _named(fn, "field_identifier")
                    if what is not None:
                        recv = _text(who, self.src) if who is not None else None
                        # A call on the method's own receiver is Go's `self.x()`.
                        on_self = recv is not None and recv in receiver_names
                        self.p.calls.append(CallSite(
                            caller=caller, file=self.p.path,
                            line=node.start_point[0] + 1,
                            name=_text(what, self.src),
                            receiver=recv, on_self=on_self))
            stack.extend(node.children)


def parse(source: str, parsed: ParsedFile) -> bool:
    if not available():                            # pragma: no cover
        return False
    src = source.encode("utf-8")
    tree = _parser.parse(src)
    root = tree.root_node
    if root.has_error and not root.children:
        return False
    reader = _Reader(parsed, src)

    for node in root.children:
        if node.type == "import_declaration":
            _imports(node, src, parsed)
        elif node.type == "type_declaration":
            _types(node, src, reader)
        elif node.type == "function_declaration":
            name = _named(node, "identifier")
            if name is None:
                continue
            nid = reader.emit(_text(name, src), "function", node)
            body = _named(node, "block")
            reader.local_types(node, _text(name, src), {})
            if body is not None:
                reader.calls_in(body, nid, set())
        elif node.type == "method_declaration":
            _method(node, src, reader, parsed)
    return True


def _imports(node, src: bytes, parsed: ParsedFile) -> None:
    for spec in _walk_type(node, "import_spec"):
        path_node = _named(spec, "interpreted_string_literal", "raw_string_literal")
        if path_node is None:
            continue
        path = _text(path_node, src).strip('"`')
        alias = _named(spec, "package_identifier")
        # `import "net/http"` binds the name `http`; the path's last segment.
        local = _text(alias, src) if alias is not None else path.rsplit("/", 1)[-1]
        parsed.imports[local] = path.replace("/", ".")
        parsed.import_sites.append((path.replace("/", "."), spec.start_point[0] + 1))


def _types(node, src: bytes, reader: _Reader) -> None:
    specs = _walk_type(node, "type_spec")
    # `type Foo struct{}` carries its doc comment above the declaration, not
    # above the spec inside it -- the spec's previous sibling is the `type`
    # keyword. A grouped `type ( A ...; B ... )` is the other way round: each
    # spec has its own comment.
    grouped = len(specs) > 1
    for spec in specs:
        name_node = _named(spec, "type_identifier")
        if name_node is None:
            continue
        name = _text(name_node, src)
        body = _named(spec, "struct_type", "interface_type")
        # An embedded type is Go's inheritance: the outer type gains its
        # methods, which is what `inherits` means everywhere else here.
        bases = _embedded(body, src) if body is not None else []
        reader.emit(name, "class", spec if grouped else node, bases=bases or None)
        reader.p.defined_classes.add(name)


def _embedded(body, src: bytes) -> list[str]:
    out: list[str] = []
    for field in _walk_type(body, "field_declaration"):
        # An embedded field has a type and no name of its own.
        if _named(field, "field_identifier") is None:
            found = _type_name(field, src)
            if found:
                out.append(found)
    return out


def _method(node, src: bytes, reader: _Reader, parsed: ParsedFile) -> None:
    receiver = _named(node, "parameter_list")
    name_node = _named(node, "field_identifier")
    if name_node is None or receiver is None:
        return
    owner = _type_name(receiver, src)
    name = _text(name_node, src)
    nid = reader.emit(name, "method", node, scope=[owner] if owner else None)
    if owner:
        parsed.owner_of[nid] = owner
        # The receiver's name is a variable of the owner's type, stated in the
        # source. No inference, unlike Python's `db = Database()`.
        var = _named(_named(receiver, "parameter_declaration") or receiver, "identifier")
        own = set()
        if var is not None:
            local = _text(var, src)
            own.add(local)
            parsed.var_types[f"{owner}.{name}::{local}"] = owner
    else:
        own = set()
    body = _named(node, "block")
    reader.local_types(node, f"{owner}.{name}" if owner else name, {})
    if body is not None:
        reader.calls_in(body, nid, own)


def _last_type(node):
    """The type part of a declaration: the last child that names a type."""
    for child in reversed(node.children):
        if child.type in ("type_identifier", "pointer_type", "qualified_type",
                          "slice_type", "array_type", "generic_type"):
            return child
    return None


def _first_deep(node, wanted: str):
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type == wanted:
            return n
        stack.extend(n.children)
    return None


def _walk_type(node, wanted: str):
    stack, out = list(node.children), []
    while stack:
        n = stack.pop()
        if n.type == wanted:
            out.append(n)
        stack.extend(n.children)
    return out
