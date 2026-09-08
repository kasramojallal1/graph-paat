"""C#, read with tree-sitter.

The vocabulary is the shared one. A class, an interface, a struct, a record, an
enum and a delegate all become `class` nodes; `: Base, IThing` becomes
`inherits` without separating the base class from the interfaces, because both
mean the type gains the other's shape, which is what a reader is asking about.

Four things about C# need a decision, and each of them costs something.

**A type is usually not at the top level of the file.** The older braced form,
`namespace N { ... }`, wraps every declaration in it -- that is 239 of
Newtonsoft.Json's 240 files -- so a reader that walks only the file's own
children finds nothing at all. The walk is recursive for that reason, which
also reaches declarations inside a `#if`, since the grammar keeps those in a
`preproc_if` wrapper. The newer file-scoped form, `namespace N;`, is the
opposite shape: it wraps nothing, and the types that follow it are siblings of
it. Both are handled by walking rather than by knowing which form is in use.

**A property is a method.** In C# a property is a pair of accessor methods even
when the compiler writes their bodies, so `public int Count { get; set; }` is
emitted like any other member of the type. Taking only the ones with a written
body would have been defensible -- those are the only ones that can call
anything -- but it deletes the configuration surface of a library along with
its documentation: 173 of Newtonsoft.Json's 285 bodyless properties carry a
`///` comment, and BatchingOptions is nothing but bodyless documented
properties. The cost is that a bodyless property is a `method` node with no
outgoing calls, so it looks like a leaf. A field is NOT emitted; in idiomatic
C# a field is private state (572 of Newtonsoft.Json's 681 are non-public) and
a property is the exposed member, so the line falls in about the right place.

**A namespace is not part of an id.** `Newtonsoft.Json.Linq.JToken` mints
`linq_jtoken_jtoken` from its path, not from its namespace. The file path
already separates two types of the same name, and putting the namespace in as
well would make every id twice as long for no separation gained.

**An overload is not a separate symbol here.** `ForContext` appears five times
in Serilog's Log.cs and mints one id, so four of them are reported as a
collision. Encoding the parameter list into the id would fix it and would also
make every id unstable under a signature change, which is worse for a map an
agent has to name things in.

Doc comments are runs of `///` lines, which is C#'s documentation form; a `//`
note is not taken, because every file in both corpora opens with a licence
header of them and each one would otherwise become the first declaration's
rationale. The XML inside is flattened to prose -- a `<see cref="Log"/>` keeps
the name it points at, an `<example>` block is dropped because a code sample
crowds the sentence out of the 600-character budget.
"""
from __future__ import annotations

import re

from ..ids import mint
from ..parse import DOC_SUFFIX, CallSite, Edge, Node, ParsedFile

EXTENSIONS = {".cs"}
WHY = ""

_parser = None


def available() -> bool:
    """True when the grammar is installed. The registry skips us otherwise, so
    a Python-only user never has to carry a C# grammar."""
    global _parser, WHY
    if _parser is not None:
        return True
    try:
        import tree_sitter_c_sharp
        from tree_sitter import Language, Parser
        _parser = Parser(Language(tree_sitter_c_sharp.language()))
        return True
    except Exception as exc:                      # pragma: no cover
        WHY = f"needs tree-sitter and tree-sitter-c-sharp ({exc})"
        return False


# Every declaration that names a type. `record_declaration` covers `record
# struct` too -- the grammar uses one node type with an extra `struct` token --
# and a delegate is a named type like the rest, so it is a class node as well.
_TYPES = {"class_declaration", "interface_declaration", "struct_declaration",
          "record_declaration", "enum_declaration", "delegate_declaration"}

# Declarations that own a body. An indexer and an operator are deliberately
# absent: neither has a name a caller could write, so there would be no label
# for a query to match. That loses Newtonsoft.Json's 72 conversion operators
# (`explicit operator bool` on JToken and JValue) and its 19 indexers, which is
# a real gap, but inventing a name for them would be a worse one.
_CALLABLES = {"method_declaration", "constructor_declaration",
              "property_declaration", "local_function_statement"}

# The keyword that introduces a type, used to find the name when the grammar
# could not. `record struct` has two of them, so the search does not stop at
# the first.
_TYPE_KEYWORDS = {"class", "interface", "struct", "record", "enum", "delegate"}
# The `@` of a verbatim identifier is an escape, not part of the name.
_LEADING_NAME = re.compile(r"\s*@?([A-Za-z_][A-Za-z0-9_]*)")

# A type that never names a class in the corpus. `int` and `string` are the
# language's own; `var` states nothing at all and the initialiser is read
# instead.
_NOT_A_CLASS = {"predefined_type", "implicit_type"}

# A type expression that wraps another one. `Server?`, `Server[]` and `ref
# Server` are all the Server type for the purpose of knowing what a variable is.
_TYPE_WRAPPERS = {"nullable_type", "array_type", "pointer_type", "ref_type",
                  "scoped_type"}

# `<example>` holds source, not prose. Left in, a sample crowds the actual
# sentence out of the 600-character budget -- ILogger's summary is one line and
# its example is eight.
_DOC_EXAMPLE = re.compile(r"<example\b[^>]*>.*?</example>", re.I | re.S)
# `<code>` is both: a block inside an example, and a one-word value inline.
# Dropping it outright turned "when value is <code>null</code>" into "when
# value is", which reads as damage rather than as brevity, so only the form
# that runs across lines is treated as a sample.
_DOC_CODE = re.compile(r"<code\b[^>]*>(.*?)</code>", re.I | re.S)
# `<see cref="Log"/>` points at a real symbol and `<see langword="true"/>` at a
# keyword, so both keep their value rather than being dropped with the tag:
# "the methods on  and its sibling " and "returns  if enabled, otherwise ."
# both read as damage. Serilog alone writes 41 langword references.
_DOC_REF = re.compile(
    r"<(?:see|seealso)\s+(?:cref|langword)\s*=\s*\"([^\"]+)\"\s*/?>", re.I)
_DOC_NAMEREF = re.compile(
    r"<(?:paramref|typeparamref)\s+name\s*=\s*\"([^\"]+)\"\s*/?>", re.I)
_DOC_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_DOC_SPACED_PUNCT = re.compile(r"\s+([.,;:!?])")
_DOC_ENTITIES = (("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'),
                 ("&apos;", "'"), ("&amp;", "&"))


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _compact(node, src: bytes) -> str:
    return " ".join(_text(node, src).split())


def _named(node, *types):
    """First direct child of any of these types."""
    for child in node.children:
        if child.type in types:
            return child
    return None


def _type_name(node, src: bytes) -> str | None:
    """The bare name of a type expression, or None when it names no class.

    Each branch exists because the obvious recursive descent gets it wrong:
    `List<Server>` would come back as Server (the element, not the variable's
    type) and `int` would come back as a class the corpus does not contain.
    """
    if node is None:
        return None
    kind = node.type
    if kind == "identifier":
        return _text(node, src)
    if kind in _NOT_A_CLASS:
        return None
    if kind == "generic_name":
        ident = _named(node, "identifier")
        return _text(ident, src) if ident is not None else None
    if kind in ("qualified_name", "alias_qualified_name"):
        # `System.Text.StringBuilder` is a StringBuilder. The namespace is not
        # part of what a type is called anywhere else in the graph.
        last = None
        for child in node.children:
            if child.type in ("identifier", "generic_name"):
                last = child
        return _type_name(last, src) if last is not None else None
    if kind in _TYPE_WRAPPERS:
        inner = node.child_by_field_name("type")
        if inner is None:
            inner = next((c for c in node.children if c.is_named), None)
        return _type_name(inner, src)
    return None


def _call_name(node, src: bytes) -> str | None:
    """The name being invoked. `Foo<int>()` is a call to Foo."""
    if node is None:
        return None
    if node.type == "identifier":
        return _text(node, src)
    if node.type == "generic_name":
        ident = _named(node, "identifier")
        return _text(ident, src) if ident is not None else None
    return None


def _declared_name(node, src: bytes) -> str | None:
    """A type's name, including when the grammar could not find it.

    A `#if` between the keyword and the base list is not something the grammar
    can parse. It recovers by leaving the real name as raw text inside an ERROR
    node and promoting the *preprocessor symbol* to the `name` field, so

        public abstract partial class JsonWriter
        #if HAVE_ASYNC_DISPOSABLE
            : IAsyncDisposable
        #endif

    declares, as far as the tree is concerned, a type called
    HAVE_ASYNC_DISPOSABLE. Two of Newtonsoft.Json's files are written that way,
    and taking the `name` field at its word filed every async method of
    JsonReader and JsonWriter under a name that is not a type at all. The ERROR
    sits directly after the keyword, so the identifier it opens with is the
    name the author wrote.
    """
    kids = node.children
    for i, child in enumerate(kids):
        if child.type in _TYPE_KEYWORDS and i + 1 < len(kids) \
                and kids[i + 1].type == "ERROR":
            found = _LEADING_NAME.match(_text(kids[i + 1], src))
            if found:
                return found.group(1)
    name = node.child_by_field_name("name")
    return _text(name, src) if name is not None else None


def _clean_doc(raw: str) -> str:
    """Flatten one doc comment's XML to prose.

    `raw` still has its line breaks: whether a `<code>` element spans lines is
    the only way to tell a sample from an inline value, and collapsing the
    whitespace first would throw that away.
    """
    text = _DOC_EXAMPLE.sub(" ", raw)
    text = _DOC_CODE.sub(lambda m: " " if "\n" in m.group(1) else m.group(1), text)
    text = _DOC_REF.sub(lambda m: m.group(1).split(":")[-1], text)
    text = _DOC_NAMEREF.sub(lambda m: m.group(1), text)
    text = _DOC_TAG.sub(" ", text)
    for entity, char in _DOC_ENTITIES:
        text = text.replace(entity, char)
    # A removed inline tag leaves its space behind: `is <c>true</c>.` became
    # "is true ." Punctuation is where a sentence ends, and a doc node is read
    # by a person as often as by anything else.
    return _DOC_SPACED_PUNCT.sub(r"\1", " ".join(text.split()))


def _doc_above(node, src: bytes) -> str | None:
    """The run of `///` lines directly above a declaration.

    Two guards, both of which were needed:

    Only `///` counts. Both corpora open every file with a `//` licence header
    twenty lines long, and taking `//` runs made that header the documentation
    of whatever declaration happened to follow it.

    Only a comment on the line immediately above is documentation; one
    separated by a blank line is a note about something else. The check applies
    to the first comment too -- guarding it on "we already have lines" lets any
    single detached comment through.

    A declaration's attributes are children of the declaration, not siblings of
    it, so `node.start_point` is the `[Obsolete]` line rather than the
    signature. Measuring adjacency from there is what keeps the doc attached.
    """
    lines: list[str] = []
    prev = node.prev_sibling
    while prev is not None and prev.type == "comment":
        text = _text(prev, src)
        if not text.startswith("///"):
            break
        if prev.end_point[0] + 1 < node.start_point[0]:
            break
        lines.insert(0, text[3:].strip())
        node, prev = prev, prev.prev_sibling
    cleaned = _clean_doc("\n".join(lines))
    # `<inheritdoc/>` is a whole doc comment that becomes empty here. It says
    # "read the base class", which is not a claim about this symbol.
    return cleaned or None


class _Reader:
    """Walks one file, emitting a node per declaration.

    Scope and container are two different stacks on purpose. `scope` is the
    naming chain that goes into an id; `container` is the id of the thing that
    physically encloses this one, so `contains` points at the real parent even
    for a type nested three deep.
    """

    def __init__(self, parsed: ParsedFile, src: bytes) -> None:
        self.p = parsed
        self.src = src
        self.scope: list[str] = []
        self.container: list[str] = [parsed.prefix]
        self.types: list[str] = []          # enclosing TYPE names only

    # ---- emission ----------------------------------------------------

    def emit(self, name: str, kind: str, node, bases=None) -> str:
        nid = mint(self.p.prefix, name, self.scope)
        line = node.start_point[0] + 1
        self.p.nodes.append(Node(id=nid, label=name, kind=kind, file=self.p.path,
                                 line=line, bases=bases))
        self.p.edges.append(Edge(source=self.container[-1], target=nid,
                                 relation="contains", file=self.p.path, line=line))
        doc = _doc_above(node, self.src)
        if doc:
            doc_id = f"{nid}{DOC_SUFFIX}"
            self.p.nodes.append(Node(id=doc_id, label=f"docstring of {nid}",
                                     kind="rationale", file=self.p.path,
                                     line=line, text=doc[:600]))
            self.p.edges.append(Edge(source=nid, target=doc_id,
                                     relation="rationale_for",
                                     file=self.p.path, line=line))
        return nid

    # ---- the walk ----------------------------------------------------

    def visit(self, node) -> None:
        kind = node.type
        if kind == "using_directive":
            self._using(node)
            return
        if kind in _TYPES:
            self._type(node)
            return
        if kind in _CALLABLES:
            self._callable(node)
            return
        if kind == "parameter":
            self._parameter(node)
        elif kind == "variable_declaration":
            self._variable(node)
        elif kind == "foreach_statement":
            self._foreach(node)
        elif kind == "invocation_expression":
            self._invocation(node)
        elif kind == "object_creation_expression":
            self._creation(node)
        for child in node.children:
            self.visit(child)

    def _type(self, node) -> None:
        name = _declared_name(node, self.src)
        if name is None:                           # pragma: no cover - grammar
            return
        self.p.defined_classes.add(name)
        nid = self.emit(name, "class", node, bases=self._bases(node) or None)
        self.scope.append(name)
        self.container.append(nid)
        self.types.append(name)
        for child in node.children:
            self.visit(child)
        self.types.pop()
        self.container.pop()
        self.scope.pop()

    def _callable(self, node) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            # Error recovery left a declaration with no name. Both corpora hit
            # this: `#if` inside a signature is not something the grammar can
            # parse, so it produces a partial node. The body is still walked,
            # because the calls inside it are still real.
            for child in node.children:
                self.visit(child)
            return
        name = _text(name_node, self.src)
        # A local function lives inside a body, so it is a function; everything
        # else in this set can only be declared inside a type.
        kind = "function" if node.type == "local_function_statement" else "method"
        nid = self.emit(name, kind, node)
        owner = self.types[-1] if self.types else None
        if owner:
            # A local function can use `this` too, so it gets an owner as well.
            self.p.owner_of[nid] = owner
        self.scope.append(name)
        self.container.append(nid)
        for child in node.children:
            self.visit(child)
        self.container.pop()
        self.scope.pop()

    def _bases(self, node) -> list[str]:
        """`: Base, IThing` -- the base class and the interfaces together.

        C# separates them; the graph does not. An enum's base list is its
        underlying type (`: byte`), which names no class and drops out here.
        """
        base_list = _named(node, "base_list")
        if base_list is None:
            return []
        out = []
        for child in base_list.children:
            if not child.is_named:
                continue
            found = _type_name(child, self.src)
            if found:
                out.append(found)
        return out

    # ---- imports -----------------------------------------------------

    def _using(self, node) -> None:
        """`using System.Text;` names a namespace, not a file.

        There is no file for it to point at, so nearly every one of these stays
        unresolved -- which is the honest answer and still tells an agent what
        a file depends on.

        Only the alias form binds a local name to a symbol. A plain `using`
        opens a namespace: it makes every type in it visible without naming any
        one of them, so recording `Text` as a local name for `System.Text`
        would be stating something the source does not.
        """
        path = None
        for child in node.children:
            if child.type in ("identifier", "qualified_name",
                              "alias_qualified_name", "generic_name"):
                path = child
        if path is None:
            return
        dotted = _compact(path, self.src)
        self.p.import_sites.append((dotted, node.start_point[0] + 1))
        alias = node.child_by_field_name("name")
        if alias is not None and alias is not path:
            self.p.imports[_text(alias, self.src)] = dotted

    # ---- declared types, never inferred ------------------------------

    def _scope_key(self) -> str:
        return ".".join(self.scope)

    def _record_type(self, name: str, cls: str) -> None:
        self.p.var_types[f"{self._scope_key()}::{name}"] = cls

    def _parameter(self, node) -> None:
        cls = _type_name(node.child_by_field_name("type"), self.src)
        name = node.child_by_field_name("name")
        if cls and name is not None:
            self._record_type(_text(name, self.src), cls)

    def _variable(self, node) -> None:
        """`Server s = ...`, `var s = new Server()`, and a field of either form.

        A field matters more here than a local does. C# writes a field's type
        out and then reads the field with no qualifier at all -- `_converter`,
        not `this._converter` -- so without this a call on a private field of a
        known type looks like a call on something unnamed.
        """
        declared = _type_name(node.child_by_field_name("type"), self.src)
        is_field = node.parent is not None and node.parent.type == "field_declaration"
        owner = self.types[-1] if self.types else None
        for decl in node.children:
            if decl.type != "variable_declarator":
                continue
            name = decl.child_by_field_name("name")
            if name is None:
                continue
            cls = declared
            if cls is None:
                # `var s = new Server()` states the type just as plainly as
                # `Server s` does; it states it on the right of the `=`.
                made = _named(decl, "object_creation_expression")
                if made is not None:
                    cls = _type_name(made.child_by_field_name("type"), self.src)
            if not cls:
                continue
            label = _text(name, self.src)
            self._record_type(label, cls)
            if is_field and owner:
                # The other half: `this._converter.Foo()` is resolved from the
                # class rather than from a scope, so it needs the class key.
                self.p.attr_types[f"{owner}::{label}"] = cls

    def _foreach(self, node) -> None:
        cls = _type_name(node.child_by_field_name("type"), self.src)
        left = node.child_by_field_name("left")
        if cls and left is not None and left.type == "identifier":
            self._record_type(_text(left, self.src), cls)

    # ---- calls -------------------------------------------------------

    def _receiver(self, expr) -> tuple[str | None, bool, bool]:
        """Who a call was made on: (name, is-an-attribute-of-self, is-self).

        The last branch is the one that matters. `GetLogger().Ping()` and
        `a.b.Ping()` have a receiver we cannot name, and returning None for it
        would make the call look BARE -- which resolution answers by finding
        any same-file `Ping`, a wrong edge rather than an honest gap. Handing
        back the receiver's own source text keeps it a receiver call that no
        variable will ever match, so it is refused.

        `base.Ping()` lands there too, on purpose. It is not `this.Ping()`: it
        names the parent's method, and the class we know about is usually the
        one that overrode it, so calling it self-typed would resolve it to the
        wrong end of an override.
        """
        if expr is None:
            return (None, False, False)
        if expr.type == "this":
            return (None, False, True)
        if expr.type == "identifier":
            return (_text(expr, self.src), False, False)
        if expr.type == "member_access_expression":
            inner = expr.child_by_field_name("expression")
            name = expr.child_by_field_name("name")
            if inner is not None and inner.type == "this" and name is not None:
                return (_text(name, self.src), True, False)
        return (_compact(expr, self.src)[:80], False, False)

    def _invocation(self, node) -> None:
        fn = node.child_by_field_name("function")
        if fn is None:                             # pragma: no cover - grammar
            return
        line = node.start_point[0] + 1
        caller = self.container[-1]

        if fn.type in ("identifier", "generic_name"):
            name = _call_name(fn, self.src)
            if name:
                self.p.calls.append(CallSite(caller=caller, file=self.p.path,
                                             line=line, name=name))
            return

        if fn.type == "member_access_expression":
            name = _call_name(fn.child_by_field_name("name"), self.src)
            expr = fn.child_by_field_name("expression")
        elif fn.type == "conditional_access_expression":
            # `s?.Ping()` -- the method sits in a member_binding_expression and
            # the receiver is the `condition` field.
            binding = _named(fn, "member_binding_expression")
            name = _call_name(binding.child_by_field_name("name"), self.src) \
                if binding is not None else None
            expr = fn.child_by_field_name("condition")
        else:
            return
        if not name:
            return
        receiver, on_attr, on_self = self._receiver(expr)
        self.p.calls.append(CallSite(
            caller=caller, file=self.p.path, line=line, name=name,
            receiver=receiver, receiver_is_self=on_attr, on_self=on_self))

    def _creation(self, node) -> None:
        """`new Server()` is a call to the class.

        It is often the only edge tying a factory to what it builds. The
        target-typed form, `Server s = new()`, names nothing at the call site
        and is not recorded -- the declaration's own type is still read, so the
        variable is still typed.
        """
        name = _type_name(node.child_by_field_name("type"), self.src)
        if name:
            self.p.calls.append(CallSite(
                caller=self.container[-1], file=self.p.path,
                line=node.start_point[0] + 1, name=name))


def parse(source: str, parsed: ParsedFile) -> bool:
    if not available():                            # pragma: no cover
        return False
    src = source.encode("utf-8")
    root = _parser.parse(src).root_node
    if root.has_error and not root.children:
        return False
    reader = _Reader(parsed, src)
    for child in root.children:
        reader.visit(child)
    return True
