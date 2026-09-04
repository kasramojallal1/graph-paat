"""TypeScript, read with tree-sitter.

Same shapes as everywhere else: a class, an interface and a type alias all
become `class` nodes, `extends` and `implements` both become `inherits`, and a
`/** */` block above a declaration becomes a claim.

TypeScript sits between Python and Go on resolution. Parameter and variable
types are usually annotated -- `function use(s: Server)` -- which is declared
information as good as Go's. But a great deal of real TypeScript is functions
assigned to consts and objects with no annotation at all, and those resolve no
better than Python's.

Two shapes need care.

**Everything can be wrapped in `export`.** `export class Foo` is an
export_statement containing a class_declaration, so declarations are unwrapped
before being read or the file looks empty.

**A function is often a const.** `export const make = (a) => ...` is a
variable_declarator holding an arrow function, and skipping those would miss
most of the functions in a modern codebase.
"""
from __future__ import annotations

from ..ids import mint
from ..parse import DOC_SUFFIX, CallSite, Edge, Node, ParsedFile

# `.d.ts` is excluded in `collect`: a declaration file restates the names of the
# module beside it and mints identical ids, exactly as Python's `.pyi` stubs do.
EXTENSIONS = {".ts", ".tsx", ".mts", ".cts"}
WHY = ""

_parsers: dict[str, object] = {}


def available() -> bool:
    global WHY
    if _parsers:
        return True
    try:
        import tree_sitter_typescript as ts
        from tree_sitter import Language, Parser
        _parsers["ts"] = Parser(Language(ts.language_typescript()))
        # TSX is a different grammar: in a .tsx file `<T>` is JSX, not a cast.
        _parsers["tsx"] = Parser(Language(ts.language_tsx()))
        return True
    except Exception as exc:                       # pragma: no cover
        WHY = f"needs tree-sitter and tree-sitter-typescript ({exc})"
        return False


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _named(node, *types):
    for child in node.children:
        if child.type in types:
            return child
    return None


def _deep(node, wanted: str):
    stack, out = list(node.children), []
    while stack:
        n = stack.pop()
        if n.type == wanted:
            out.append(n)
        stack.extend(n.children)
    return out


def _first_deep(node, *wanted):
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type in wanted:
            return n
        stack.extend(n.children)
    return None


def _annotated_type(node, src: bytes) -> str | None:
    """The class named by a `: Type` annotation, through arrays and generics."""
    ann = _named(node, "type_annotation")
    if ann is None:
        return None
    ident = _first_deep(ann, "type_identifier")
    return _text(ident, src) if ident is not None else None


def _doc_above(node, src: bytes) -> str | None:
    """A `/** ... */` block directly above a declaration.

    JSDoc only. A `//` line note is usually about a line, not the declaration,
    and treating every one as documentation buried the real ones.
    """
    prev = node.prev_sibling
    while prev is not None and prev.type == "comment":
        text = _text(prev, src)
        if text.startswith("/**"):
            if prev.end_point[0] + 1 < node.start_point[0]:
                return None                # separated by a blank line
            body = text.strip("/*").replace("*/", "")
            cleaned = " ".join(l.strip().lstrip("*").strip() for l in body.splitlines())
            return " ".join(cleaned.split()) or None
        prev = prev.prev_sibling
    return None


def _unwrap(node):
    """`export class Foo` and `export default function f` hide the declaration."""
    if node.type in ("export_statement",):
        for child in node.children:
            if child.type.endswith("_declaration") or child.type == "lexical_declaration":
                return child
    return node


class _Reader:
    def __init__(self, parsed: ParsedFile, src: bytes):
        self.p = parsed
        self.src = src

    def emit(self, name, kind, node, doc_node=None, scope=None, bases=None) -> str:
        nid = mint(self.p.prefix, name, scope or [])
        line = node.start_point[0] + 1
        self.p.nodes.append(Node(id=nid, label=name, kind=kind, file=self.p.path,
                                 line=line, bases=bases))
        container = mint(self.p.prefix, scope[0]) if scope else self.p.prefix
        self.p.edges.append(Edge(source=container, target=nid, relation="contains",
                                 file=self.p.path, line=line))
        doc = _doc_above(doc_node if doc_node is not None else node, self.src)
        if doc:
            doc_id = f"{nid}{DOC_SUFFIX}"
            self.p.nodes.append(Node(id=doc_id, label=f"docstring of {nid}",
                                     kind="rationale", file=self.p.path,
                                     line=line, text=doc[:600]))
            self.p.edges.append(Edge(source=nid, target=doc_id,
                                     relation="rationale_for",
                                     file=self.p.path, line=line))
        return nid

    def types_in(self, node, scope: str) -> None:
        """Annotated parameters and variables: declared, not inferred."""
        for kind in ("required_parameter", "optional_parameter", "variable_declarator"):
            for decl in _deep(node, kind):
                name = _named(decl, "identifier")
                found = _annotated_type(decl, self.src)
                if found is None:
                    # `const s = new Server()` states the type just as clearly.
                    new = _named(decl, "new_expression")
                    if new is not None:
                        ident = _named(new, "identifier")
                        found = _text(ident, self.src) if ident is not None else None
                if name is not None and found:
                    self.p.var_types[f"{scope}::{_text(name, self.src)}"] = found

    def calls_in(self, body, caller: str) -> None:
        stack = [body]
        while stack:
            node = stack.pop()
            if node.type == "call_expression":
                fn = node.children[0] if node.children else None
                if fn is not None and fn.type == "identifier":
                    self.p.calls.append(CallSite(
                        caller=caller, file=self.p.path,
                        line=node.start_point[0] + 1, name=_text(fn, self.src)))
                elif fn is not None and fn.type == "member_expression":
                    what = _named(fn, "property_identifier")
                    if what is not None:
                        who = _named(fn, "identifier", "this")
                        on_self = who is not None and who.type == "this"
                        self.p.calls.append(CallSite(
                            caller=caller, file=self.p.path,
                            line=node.start_point[0] + 1,
                            name=_text(what, self.src),
                            receiver=None if on_self else (
                                _text(who, self.src) if who is not None else None),
                            on_self=on_self))
            elif node.type == "new_expression":
                # `new Server()` is a call to the class, and is often the only
                # edge tying a factory to what it builds.
                ident = _named(node, "identifier")
                if ident is not None:
                    self.p.calls.append(CallSite(
                        caller=caller, file=self.p.path,
                        line=node.start_point[0] + 1, name=_text(ident, self.src)))
            stack.extend(node.children)


def parse(source: str, parsed: ParsedFile) -> bool:
    if not available():                            # pragma: no cover
        return False
    src = source.encode("utf-8")
    parser = _parsers["tsx" if parsed.path.endswith(".tsx") else "ts"]
    root = parser.parse(src).root_node
    if root.has_error and not root.children:
        return False
    reader = _Reader(parsed, src)

    for outer in root.children:
        node = _unwrap(outer)
        if node.type == "import_statement":
            _imports(node, src, parsed)
        elif node.type == "class_declaration":
            _class(node, outer, src, reader, parsed)
        elif node.type == "interface_declaration":
            name = _named(node, "type_identifier")
            if name is not None:
                reader.emit(_text(name, src), "class", node, doc_node=outer,
                            bases=_heritage(node, src) or None)
                parsed.defined_classes.add(_text(name, src))
        elif node.type == "type_alias_declaration":
            name = _named(node, "type_identifier")
            if name is not None:
                reader.emit(_text(name, src), "class", node, doc_node=outer)
                parsed.defined_classes.add(_text(name, src))
        elif node.type == "function_declaration":
            name = _named(node, "identifier")
            if name is not None:
                nid = reader.emit(_text(name, src), "function", node, doc_node=outer)
                reader.types_in(node, _text(name, src))
                body = _named(node, "statement_block")
                if body is not None:
                    reader.calls_in(body, nid)
        elif node.type == "lexical_declaration":
            _const(node, outer, src, reader)
    return True


def _heritage(node, src: bytes) -> list[str]:
    """`extends Base` and `implements Handler` are both inheritance here.

    TypeScript separates them; the graph does not, because both mean the type
    gains the other's shape, which is what a reader is asking about.
    """
    out = []
    heritage = _named(node, "class_heritage")
    for clause in ("extends_clause", "implements_clause"):
        for found in _deep(heritage, clause) if heritage is not None else []:
            for ident in found.children:
                if ident.type in ("identifier", "type_identifier"):
                    out.append(_text(ident, src))
    if heritage is None:                    # interfaces put extends inline
        for found in _deep(node, "extends_type_clause"):
            ident = _first_deep(found, "type_identifier")
            if ident is not None:
                out.append(_text(ident, src))
    return out


def _class(node, outer, src: bytes, reader: _Reader, parsed: ParsedFile) -> None:
    name_node = _named(node, "type_identifier")
    if name_node is None:
        return
    name = _text(name_node, src)
    reader.emit(name, "class", node, doc_node=outer, bases=_heritage(node, src) or None)
    parsed.defined_classes.add(name)
    body = _named(node, "class_body")
    if body is None:
        return
    for member in body.children:
        if member.type != "method_definition":
            continue
        method_name = _named(member, "property_identifier")
        if method_name is None:
            continue
        label = _text(method_name, src)
        nid = reader.emit(label, "method", member, scope=[name])
        parsed.owner_of[nid] = name
        reader.types_in(member, f"{name}.{label}")
        block = _named(member, "statement_block")
        if block is not None:
            reader.calls_in(block, nid)


def _const(node, outer, src: bytes, reader: _Reader) -> None:
    """`const make = (a) => ...` -- most functions in modern TypeScript."""
    for decl in _deep(node, "variable_declarator"):
        name = _named(decl, "identifier")
        fn = _named(decl, "arrow_function", "function_expression")
        if name is None or fn is None:
            continue
        label = _text(name, src)
        nid = reader.emit(label, "function", decl, doc_node=outer)
        reader.types_in(fn, label)
        body = _named(fn, "statement_block") or fn
        reader.calls_in(body, nid)


def _imports(node, src: bytes, parsed: ParsedFile) -> None:
    path_node = _named(node, "string")
    if path_node is None:
        return
    frag = _named(path_node, "string_fragment")
    path = _text(frag, src) if frag is not None else _text(path_node, src).strip("'\"")
    # "./foo" and "../bar/baz" become dotted module names the resolver already
    # knows how to match against file prefixes.
    dotted = path.lstrip("./").replace("/", ".")
    clause = _named(node, "import_clause")
    if clause is not None:
        for ident in _deep(clause, "identifier") + list(
                c for c in clause.children if c.type == "identifier"):
            parsed.imports[_text(ident, src)] = f"{dotted}.{_text(ident, src)}"
    if dotted:
        parsed.import_sites.append((dotted, node.start_point[0] + 1))
