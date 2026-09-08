"""Swift, read with tree-sitter.

The vocabulary is the shared one. A class, a struct, an enum, an actor, a
protocol and a typealias all become `class` nodes; a conformance list becomes
`inherits`. Nothing downstream learns that Swift exists.

The grammar is the least conventional of the set, and everything below was
checked against a real parse rather than guessed at. Four of its shapes decide
how this module is written.

**One node type covers five keywords.** `class`, `struct`, `enum`, `actor` and
`extension` all arrive as `class_declaration`, told apart by a
`declaration_kind` child. Only `protocol` and `typealias` have node types of
their own. So the dispatch is on the keyword, not on the node type.

**An extension is a body with no type of its own.** `extension Collection {
func chunked(by:) }` adds a method to a type declared somewhere else -- often
in the standard library, which is not in the corpus at all. Those methods are
scoped to the type they extend, because that is where a caller finds them, but
**no class node is minted for the extended type**. Minting one looked tidier
and is measurably worse: 20 of Alamofire's extensions extend a type that a
different file in the same corpus declares, so a second node of that name would
make the class ambiguous and stop every receiver-typed call to it from
resolving. Instead the extended type's members hang off the file when the type
is not declared here, and off the type's own node when it is. The cost is
stated plainly: a method added to `Request` from `Response.swift` cannot be
found by a caller who only knows the type, because the type's home file is a
different one.

**A nested type is minted flat.** `Session.MutableState` becomes
`session_mutablestate`, contained by `Session`, so the id and the containment
edge deliberately say different things. A type's members are minted
`prefix + type name + member`, and the only way a lookup by type name finds
both the type and its members is for the type's own id to be
`prefix + type name`; qualify it and the index points at nothing. The cost is
real and measured: Swift nests a type inside an extension so routinely that 17
type nodes across the two corpora share an id with another, almost all of them
`Index` and `Iterator` structs written once per collection in a single file.
They are reported as collisions rather than quietly merged.

**`self.` is optional.** Swift lets a method call its own type's members
unqualified, and most Swift does. A bare `finish()` inside a method is
therefore recorded as a call on self when the enclosing type declares a member
of that name *in this file*; otherwise it stays a bare call. Without that rule
the commonest call in the language resolves to nothing, because the bare-call
path only looks at file-level functions.

**An overload is one node, not many**, as everywhere else here. Swift takes
that further than most languages, because argument labels are part of a
method's name: `EventMonitor` declares fifty different `request` callbacks.
They mint one id and the first declaration wins -- 342 of Alamofire's 1,131
callable declarations are dropped that way, along with their doc comments.
Encoding the labels into the id would keep them all and would make every id
change when a signature does, which is worse for a map an agent has to name
things in.

A doc comment is a run of `///` lines or a `/** */` block. A plain `//` run is
not taken: every file in both corpora opens with a licence header of them, and
each one would otherwise become the first declaration's rationale.

What does not fit, stated plainly. **An enum case is not a node** -- `.cancelled`
is a value, and the 182 of them across the two corpora would crowd the map
without answering anything a reader asks. **A stored property is not a node
either**, in Swift as in every other language here; its declared type is kept,
which is what resolution needs. **A closure has no name to mint**, so work
written inside `queue.async { ... }` is attributed to the method the closure
sits in, which is the closest true answer.
"""
from __future__ import annotations

from ..ids import mint
from ..parse import DOC_SUFFIX, CallSite, Edge, Node, ParsedFile

EXTENSIONS = {".swift"}
WHY = ""

_parser = None

# The bodies a type declaration can have. `enum_class_body` is the enum's, and
# it holds `enum_entry` cases alongside ordinary methods; `protocol_body` holds
# requirements that have no implementation.
_BODIES = ("class_body", "enum_class_body", "protocol_body")

# Members that are callable and carry a name of their own.
_CALLABLES = ("function_declaration", "init_declaration", "deinit_declaration",
              "subscript_declaration", "protocol_function_declaration")


def available() -> bool:
    """True when the grammar is installed. The registry skips us otherwise, so
    a Python-only user never has to carry a Swift grammar."""
    global _parser, WHY
    if _parser is not None:
        return True
    try:
        import tree_sitter_swift
        from tree_sitter import Language, Parser
        _parser = Parser(Language(tree_sitter_swift.language()))
        return True
    except Exception as exc:                       # pragma: no cover
        WHY = f"needs tree-sitter and tree-sitter-swift ({exc})"
        return False


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _ident(node, src: bytes) -> str:
    """An identifier, with Swift's keyword escaping removed.

    `public static var \\`default\\`` declares a member called `default`; the
    backticks are there only because the word is a keyword, and a caller writes
    `JSONParameterEncoder.default` with none. Keeping them would mint an id
    nobody could name and would stop those members resolving against their own
    call sites.
    """
    return _text(node, src).strip("`")


def _named(node, *types):
    """First direct child of any of these types."""
    if node is None:
        return None
    for child in node.children:
        if child.type in types:
            return child
    return None


def _after(node, keyword: str):
    """The child that follows a keyword token.

    Used instead of the `name` field because the grammar reuses that field
    name: a `function_declaration` labels both its own name and its return type
    `name`, so asking for the field is ambiguous where asking for "the thing
    after `func`" is not.
    """
    for i, child in enumerate(node.children):
        if child.type == keyword and i + 1 < len(node.children):
            return node.children[i + 1]
    return None


def _first_deep(node, *wanted):
    if node is None:
        return None
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type in wanted:
            return n
        stack.extend(reversed(n.children))
    return None


def _deep(node, *wanted):
    stack, out = list(node.children), []
    while stack:
        n = stack.pop()
        if n.type in wanted:
            out.append(n)
        stack.extend(n.children)
    return out


def _type_name(node, src: bytes) -> str | None:
    """The bare name of a type, through optionals, arrays and generics.

    `Server`, `Server?`, `[Server]` and `Server<T>` all name Server for the
    purpose of knowing what a variable is. `[String: Request]` names String,
    which is wrong and harmless: nothing in a corpus is called String, so the
    only effect is a refusal instead of an edge.
    """
    ident = _first_deep(node, "type_identifier")
    return _text(ident, src) if ident is not None else None


def _extended_name(node, src: bytes) -> str | None:
    """The type an `extension` extends.

    `extension String.Encoding` names the nested type `Encoding`, not `String`,
    so the LAST component wins -- which is also what makes `extension
    Session.RequestSetup` land on the nested type's own flat id. `extension
    [Int]` names a sugar form with no type_identifier of its own and is
    refused, leaving its members as file-level functions.
    """
    if node is None or node.type != "user_type":
        return None
    last = None
    for child in node.children:
        if child.type == "type_identifier":
            last = child
    return _ident(last, src) if last is not None else None


def _conformances(node, src: bytes) -> list[str]:
    """`: Sendable, Equatable` -- everything after the colon.

    Swift separates a superclass from the protocols it conforms to only by
    position, and the graph does not separate them at all: both mean the type
    gains the other's shape, which is what a reader is asking about.
    """
    out = []
    for child in node.children:
        if child.type == "inheritance_specifier":
            found = _type_name(child, src)
            if found:
                out.append(found)
    return out


def _func_name(node, src: bytes) -> str | None:
    """The name after `func`.

    An operator declaration -- `static func == (lhs:, rhs:)` -- puts a bare
    token there instead of an identifier, and it is kept: `==` and `<` are real
    API in Swift, 34 of the 1,045 functions across the two corpora, and a
    symbol that is invisible is worse than one with an awkward name.
    """
    nxt = _after(node, "func")
    if nxt is None or nxt.type == "(":
        return None
    return _ident(nxt, src)


def _bound_name(node, src: bytes) -> str | None:
    """The identifier a `let`/`var` binds.

    A protocol requirement writes `var name: String { get }`, and there the
    binding keyword sits INSIDE the pattern, so the pattern's first identifier
    is the name in both forms.
    """
    pattern = _named(node, "pattern")
    if pattern is None:
        return None
    ident = pattern.child_by_field_name("bound_identifier")
    if ident is None:
        ident = _named(pattern, "simple_identifier")
    return _ident(ident, src) if ident is not None else None


def _init_type(node, src: bytes) -> str | None:
    """`let s = Session()` -- the type stated by an initialiser call.

    Swift writes construction exactly like a function call, so `Session()` and
    `helper()` are the same shape and only the capital letter tells them apart.
    That convention is the language's own (the API design guidelines require
    it) but it is still a convention, so the cost is stated: a factory function
    written with a leading capital would type its variable wrongly. It cannot
    produce a wrong EDGE -- a name that is not a class in the corpus simply
    refuses -- so the exposure is a refusal, not a lie.
    """
    value = None
    for i, child in enumerate(node.children):
        if child.type == "=" and i + 1 < len(node.children):
            value = node.children[i + 1]
    if value is None or value.type != "call_expression":
        return None
    callee = value.children[0] if value.children else None
    if callee is None or callee.type != "simple_identifier":
        return None
    name = _ident(callee, src)
    return name if name[:1].isupper() else None


def _doc_above(node, src: bytes) -> str | None:
    """A run of `///` lines, or a `/** */` block, directly above a declaration.

    A plain `//` run is deliberately NOT documentation. Both corpora open every
    file with a licence header written that way, and taking `//` would make the
    MIT licence the rationale of whatever declaration happened to follow it.
    """
    lines: list[str] = []
    prev = node.prev_sibling
    while prev is not None and prev.type in ("comment", "multiline_comment"):
        # Only a comment on the line immediately above documents this
        # declaration; one separated by a blank line is a note about something
        # else. The check has to apply to the first comment too.
        if prev.end_point[0] + 1 < node.start_point[0]:
            break
        text = _text(prev, src)
        if prev.type == "multiline_comment":
            if not text.startswith("/**"):
                break
            body = text[3:]
            if body.endswith("*/"):
                body = body[:-2]
            lines.insert(0, " ".join(
                l.strip().lstrip("*").strip() for l in body.splitlines()))
        else:
            if not text.startswith("///"):
                break
            lines.insert(0, text[3:].strip())
        node, prev = prev, prev.prev_sibling
    joined = " ".join(l for l in lines if l).strip()
    return " ".join(joined.split()) or None


class _Survey:
    """What the file declares, read before anything is emitted.

    Three answers are needed before the first node can be minted, and each one
    depends on a part of the file that may come later than the part that needs
    it:

      * which type names this file declares -- an extension of one of them
        contains its members, an extension of anything else does not;
      * which members each type has, counting the ones an extension in this
        file adds -- that is what makes an unqualified `finish()` a call on
        self;
      * what a type conforms to retroactively -- `extension Request: Equatable`
        is how Swift states inheritance after the fact, and the class node has
        already been emitted by the time that line is reached.
    """

    def __init__(self, root, src: bytes) -> None:
        self.src = src
        self.declared: set[str] = set()
        self.members: dict[str, set[str]] = {}
        self.extra_bases: dict[str, list[str]] = {}
        self._walk(root.children, None)

    def _walk(self, children, owner: str | None) -> None:
        for node in children:
            if node.type == "class_declaration" and _named(node, "extension"):
                name = _extended_name(_after(node, "extension"), self.src)
                if name:
                    self.extra_bases.setdefault(name, []).extend(
                        _conformances(node, self.src))
                body = _named(node, *_BODIES)
                if body is not None:
                    self._walk(body.children, name)
                continue
            if node.type in ("class_declaration", "protocol_declaration"):
                ident = _named(node, "type_identifier")
                if ident is None:
                    continue
                name = _text(ident, self.src)
                self.declared.add(name)
                body = _named(node, *_BODIES)
                if body is not None:
                    self._walk(body.children, name)
                continue
            if node.type == "typealias_declaration":
                ident = _named(node, "type_identifier")
                if ident is not None:
                    self.declared.add(_text(ident, self.src))
                continue
            if owner is None:
                continue
            label = None
            if node.type == "function_declaration":
                label = _func_name(node, self.src)
            elif node.type == "protocol_function_declaration":
                ident = _named(node, "simple_identifier")
                label = _ident(ident, self.src) if ident is not None else None
            elif node.type in ("property_declaration", "protocol_property_declaration"):
                label = _bound_name(node, self.src)
            elif node.type == "init_declaration":
                label = "init"
            if label:
                self.members.setdefault(owner, set()).add(label)


class _Reader:
    def __init__(self, parsed: ParsedFile, src: bytes, survey: _Survey) -> None:
        self.p = parsed
        self.src = src
        self.survey = survey
        self.minted: set[str] = set()

    # ---- emission ----------------------------------------------------

    def emit(self, name: str, kind: str, node, container: str,
             scope=None, bases=None, dedupe: bool = False) -> str | None:
        """One node, its containment edge and its doc comment. None when dropped.

        `container` is passed in rather than derived from `scope`, because a
        nested type is contained by the type around it while its id is minted
        flat, and a method added by an extension is scoped to a type whose node
        may live in another file -- the two are genuinely different facts here.
        """
        nid = mint(self.p.prefix, name, scope or [])
        if dedupe and nid in self.minted:
            return None
        self.minted.add(nid)
        line = node.start_point[0] + 1
        self.p.nodes.append(Node(id=nid, label=name, kind=kind, file=self.p.path,
                                 line=line, bases=bases))
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

    # ---- the walk ----------------------------------------------------

    def walk(self, children, container: str, owner: str | None) -> None:
        for node in children:
            self.member(node, container, owner)

    def member(self, node, container: str, owner: str | None) -> None:
        kind = node.type
        if kind == "import_declaration":
            self.imports(node)
        elif kind == "class_declaration" and _named(node, "extension"):
            self.extension(node, container)
        elif kind in ("class_declaration", "protocol_declaration"):
            self.type_decl(node, container)
        elif kind == "typealias_declaration":
            ident = _named(node, "type_identifier")
            if ident is not None:
                name = _text(ident, self.src)
                self.emit(name, "class", node, container)
                self.p.defined_classes.add(name)
        elif kind in _CALLABLES:
            self.callable(node, container, owner)
        elif kind == "property_declaration":
            self.property(node, container, owner)
        elif kind == "protocol_property_declaration":
            # `var name: String { get }` is a requirement, not storage: every
            # conforming type must provide it, and in both corpora each one
            # carries a `///` comment that is the only description of it. So a
            # protocol property is a node even though a stored one is not.
            self.requirement(node, container, owner)
        # `associatedtype Element` is deliberately not a node. It names a
        # placeholder that has no definition anywhere, and emitting it would
        # put `Element` and `Index` into the corpus-wide class index from every
        # protocol that declares one, making both names ambiguous for the types
        # that really are called that.

    def type_decl(self, node, container: str) -> None:
        ident = _named(node, "type_identifier")
        if ident is None:
            return
        name = _text(ident, self.src)
        # A retroactive conformance -- `extension Request: Equatable` further
        # down the file -- is part of what the type IS, and the survey has
        # already read it.
        bases = _conformances(node, self.src) + self.survey.extra_bases.get(name, [])
        nid = self.emit(name, "class", node, container, bases=bases or None)
        self.p.defined_classes.add(name)
        body = _named(node, *_BODIES)
        if body is not None and nid is not None:
            self.walk(body.children, nid, name)

    def extension(self, node, container: str) -> None:
        """`extension Foo { ... }` -- members of a type declared elsewhere.

        No node is minted for `Foo`. When `Foo` is declared in this file its
        own node holds the members; otherwise the file does, which is the
        honest answer -- the file really does contain them, and claiming the
        type is defined here would make it ambiguous with its real home.
        """
        name = _extended_name(_after(node, "extension"), self.src)
        inner = container
        if name and name in self.survey.declared:
            inner = mint(self.p.prefix, name)
        body = _named(node, *_BODIES)
        if body is not None:
            self.walk(body.children, inner, name)

    def callable(self, node, container: str, owner: str | None) -> None:
        if node.type == "init_declaration":
            label = "init"
        elif node.type == "deinit_declaration":
            label = "deinit"
        elif node.type == "subscript_declaration":
            # A subscript has no name of its own; `subscript` is what Swift
            # calls it and what a reader searching for it would type.
            label = "subscript"
        elif node.type == "protocol_function_declaration":
            ident = _named(node, "simple_identifier")
            label = _ident(ident, self.src) if ident is not None else None
        else:
            label = _func_name(node, self.src)
        if not label:
            return
        # Swift overloads by argument label and by type, so `request(_ url:)`
        # and `request(_ convertible:)` are two declarations of one entry
        # point. They mint one id; the first wins and the rest are dropped
        # rather than reported as lost symbols. Alamofire's Session declares
        # `request` three times and `upload` twelve.
        if owner:
            nid = self.emit(label, "method", node, container,
                            scope=[owner], dedupe=True)
        else:
            nid = self.emit(label, "function", node, container, dedupe=True)
        if nid is None:
            return
        if owner:
            self.p.owner_of[nid] = owner
        scope = f"{owner}.{label}" if owner else label
        self.types_in(node, scope)
        body = _named(node, "function_body", "computed_property")
        if body is not None:
            self.calls_in(body, nid, owner)

    def property(self, node, container: str, owner: str | None) -> None:
        """A property: its declared type always, and a node when it computes.

        A computed property -- `var isSuccess: Bool { ... }` -- is a function
        with no argument list, and 258 of the 1,425 properties across the two
        corpora are one. Skipping them costs whole files most of what they
        declare: four of the seven members of `Result+Alamofire.swift` are
        computed properties. A stored property is NOT a node; its declared
        type is kept, which is what resolution needs.
        """
        name = _bound_name(node, self.src)
        declared = _type_name(_named(node, "type_annotation"), self.src)
        if declared is None:
            declared = _init_type(node, self.src)
        if name and declared:
            self.p.var_types[f"{owner or ''}::{name}"] = declared
            if owner:
                # `self.session.finish()` needs the attribute's type, and a
                # bare `session.finish()` -- which Swift writes far more often
                # -- needs the same fact under the variable lookup.
                self.p.attr_types[f"{owner}::{name}"] = declared
        computed = _named(node, "computed_property")
        if computed is None:
            # A stored property still runs its initialiser, and at type level
            # `let shared = Session()` is often the only edge tying a type to
            # the one it builds. The enclosing symbol is the caller.
            self.calls_in(node, container, owner)
            return
        if not name:
            return
        if owner:
            nid = self.emit(name, "method", node, container,
                            scope=[owner], dedupe=True)
        else:
            nid = self.emit(name, "function", node, container, dedupe=True)
        if nid is None:
            return
        if owner:
            self.p.owner_of[nid] = owner
        self.types_in(computed, f"{owner}.{name}" if owner else name)
        self.calls_in(computed, nid, owner)

    def requirement(self, node, container: str, owner: str | None) -> None:
        name = _bound_name(node, self.src)
        if not name:
            return
        declared = _type_name(_named(node, "type_annotation"), self.src)
        if declared and owner:
            self.p.attr_types[f"{owner}::{name}"] = declared
            self.p.var_types[f"{owner}::{name}"] = declared
        if owner:
            nid = self.emit(name, "method", node, container,
                            scope=[owner], dedupe=True)
            if nid is not None:
                self.p.owner_of[nid] = owner

    # ---- evidence for resolution -------------------------------------

    def types_in(self, node, scope: str) -> None:
        """Declared types of parameters and locals. Stated, never inferred.

        Swift annotates a parameter always and a local often, so this is
        declared information as good as Go's receiver -- with the one exception
        of `let s = Session()`, whose reasoning is in `_init_type`.
        """
        for param in _deep(node, "parameter"):
            ident = _named(param, "simple_identifier")
            found = _type_name(_last_type(param), self.src)
            if ident is not None and found:
                self.p.var_types[f"{scope}::{_ident(ident, self.src)}"] = found
        for decl in _deep(node, "property_declaration"):
            name = _bound_name(decl, self.src)
            found = _type_name(_named(decl, "type_annotation"), self.src)
            if found is None:
                found = _init_type(decl, self.src)
            if name and found:
                self.p.var_types[f"{scope}::{name}"] = found

    def calls_in(self, body, caller: str, owner: str | None) -> None:
        own = self.survey.members.get(owner, set()) if owner else set()
        stack = [body]
        while stack:
            node = stack.pop()
            if node.type == "call_expression":
                self.call(node, caller, own)
            stack.extend(node.children)

    def call(self, node, caller: str, own: set[str]) -> None:
        fn = node.children[0] if node.children else None
        if fn is None:
            return
        line = node.start_point[0] + 1
        if fn.type == "simple_identifier":
            name = _ident(fn, self.src)
            # `Session()` is a call to the type. Nothing special is needed for
            # it -- construction and a plain call are the same shape in Swift,
            # and the bare-call path already looks at class nodes.
            self.p.calls.append(CallSite(
                caller=caller, file=self.p.path, line=line, name=name,
                # `self.` is optional in Swift. An unqualified name that the
                # enclosing type declares is a call on self; anything else
                # stays a bare call and is resolved against the file.
                on_self=name in own))
            return
        if fn.type != "navigation_expression":
            return
        suffix = fn.child_by_field_name("suffix")
        what = _named(suffix, "simple_identifier") if suffix is not None else None
        if what is None:
            return                       # `tuple.0` -- a position, not a name
        name = _ident(what, self.src)
        target = fn.child_by_field_name("target")
        if target is None:
            return
        if target.type == "self_expression":
            self.p.calls.append(CallSite(caller=caller, file=self.p.path,
                                         line=line, name=name, on_self=True))
        elif target.type == "simple_identifier":
            self.p.calls.append(CallSite(caller=caller, file=self.p.path,
                                         line=line, name=name,
                                         receiver=_ident(target, self.src)))
        elif target.type == "navigation_expression":
            # `self.session.finish()` and `Session.default.request()` are the
            # same shape. The receiver is the LAST link before the call, and
            # whether it hangs off `self` decides which evidence can type it.
            inner = target.child_by_field_name("suffix")
            ident = _named(inner, "simple_identifier") if inner is not None else None
            if ident is None:
                return
            base = target.child_by_field_name("target")
            self.p.calls.append(CallSite(
                caller=caller, file=self.p.path, line=line, name=name,
                receiver=_ident(ident, self.src),
                receiver_is_self=base is not None and base.type == "self_expression"))

    def imports(self, node) -> None:
        """`import Foundation` names a MODULE, not a file.

        Swift has no file-level imports at all: files in one module see each
        other with nothing written down. So almost every import here points
        outside the corpus and stays unresolved, which is the correct and
        useful answer -- it says which frameworks a file depends on. The
        submodule form, `import class Foundation.Thread`, binds the last
        segment as the local name, as Go's path imports do.
        """
        ident = _named(node, "identifier")
        if ident is None:
            return
        dotted = _text(ident, self.src)
        local = dotted.rsplit(".", 1)[-1]
        self.p.imports[local] = dotted
        self.p.import_sites.append((dotted, node.start_point[0] + 1))


def _last_type(node):
    """The type part of a parameter: the last child that names one."""
    for child in reversed(node.children):
        if child.type in ("user_type", "optional_type", "array_type",
                          "dictionary_type", "function_type", "tuple_type",
                          "metatype", "opaque_type", "protocol_composition_type"):
            return child
    return None


def parse(source: str, parsed: ParsedFile) -> bool:
    if not available():                            # pragma: no cover
        return False
    src = source.encode("utf-8")
    root = _parser.parse(src).root_node
    # A file with a syntax error in one function still has every other
    # declaration in it; only a file the grammar could make nothing of at all
    # is a failure. Eight of the 71 files across the two corpora carry an error
    # node -- newer syntax the grammar has not caught up with -- and losing
    # them whole would cost far more than the statement each error sits in.
    if root.has_error and not root.children:
        return False
    reader = _Reader(parsed, src, _Survey(root, src))
    reader.walk(root.children, parsed.prefix, None)
    return True
