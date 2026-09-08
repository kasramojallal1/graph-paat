"""C++, read with tree-sitter.

Same five shapes as everywhere else. A class, a struct, a union, an enum and a
`using X = Y` alias all become `class` nodes; a base class becomes `inherits`;
the comment run above a declaration becomes a claim. Nothing downstream learns
that C++ exists.

Three things about C++ make it the hardest language here, and each one is a
decision with a cost.

**There is no preprocessor.** A grammar reads the source as written, so a macro
that expands to real syntax is read as whatever it happens to look like.
`FMT_BEGIN_NAMESPACE` -- which expands to two `namespace` openings -- is read as
the return type of a function whose body is the rest of the file. Bailing out
there would lose almost every symbol in a macro-heavy header, so a
`function_definition` with no parameter list is treated as the macro it is and
we walk into its body looking for the declarations that were really there.

**A method can be defined a long way from its class.** `void Server::Start() {}`
names its owner in a qualified declarator, so the owner is read from the
declarator rather than from where the definition sits. The class itself is
usually in another file, so the containment edge falls back to the file: a
`contains` edge pointing at a class node this file never created is a link an
agent follows to nothing.

**Headers and sources share a name.** `url.c` and `url.h` are two files, so
`KEEP_EXTENSION` puts the extension into the id prefix. This module owns `.h`,
which means it also reads the headers of plain C projects -- the shapes it
produces for a C header have to be the shapes C produces for the `.c` beside it.
"""
from __future__ import annotations

from ..ids import mint
from ..parse import DOC_SUFFIX, CallSite, Edge, Node, ParsedFile

EXTENSIONS = {".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hh", ".hxx", ".h++", ".h"}

# `url.c` and `url.h` are two files with one name. Dropping the extension from
# the id prefix would merge them, and the merge is not small: measured over
# curl's `lib/`, 159 of 385 files share a name with another.
KEEP_EXTENSION = True

WHY = ""

_parser = None


def available() -> bool:
    """True when the grammar is installed. The registry skips us otherwise, so
    a Python-only user never has to carry a C++ grammar."""
    global _parser, WHY
    if _parser is not None:
        return True
    try:
        import tree_sitter_cpp
        from tree_sitter import Language, Parser
        _parser = Parser(Language(tree_sitter_cpp.language()))
        return True
    except Exception as exc:                       # pragma: no cover
        WHY = f"needs tree-sitter and tree-sitter-cpp ({exc})"
        return False


# Nodes that hold declarations without being one. Preprocessor conditionals are
# in this list because a header wraps its whole contents in an include guard --
# stopping at `#ifndef FMT_ARGS_H_` would read every header as empty.
_TRANSPARENT = {
    "translation_unit", "declaration_list", "namespace_definition",
    "linkage_specification", "field_declaration_list",
    "preproc_if", "preproc_ifdef", "preproc_else", "preproc_elif",
    "preproc_elifdef", "preproc_elifndef",
    # An access specifier in a body the grammar mistook for a statement block
    # reads as `public:` labelling everything after it. The members are inside
    # the label, so not looking through it loses the whole public section.
    "labeled_statement",
    # A recovery node still contains real code. Skipping it loses whatever
    # followed the first thing the grammar could not read, which in a
    # macro-heavy file is most of the file.
    "ERROR",
}

# Declarator layers between a declaration and the name it declares.
_WRAPPERS = {"pointer_declarator", "reference_declarator", "array_declarator",
             "init_declarator", "attributed_declarator", "parenthesized_declarator"}

_DECLARATORS = _WRAPPERS | {
    "function_declarator", "identifier", "field_identifier", "type_identifier",
    "qualified_identifier", "operator_name", "destructor_name",
    "template_function", "structured_binding_declarator"}

_RECORDS = {"class_specifier", "struct_specifier", "union_specifier"}


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _named(node, *types):
    if node is None:
        return None
    for child in node.children:
        if child.type in types:
            return child
    return None


def _field(node, name: str):
    if node is None:
        return None
    try:
        return node.child_by_field_name(name)
    except Exception:                              # pragma: no cover
        return None


def _scope_name(node, src: bytes) -> str | None:
    """The bare name of the `Foo` in `Foo::bar`, through templates."""
    if node is None:
        return None
    if node.type in ("namespace_identifier", "type_identifier", "identifier"):
        return _text(node, src)
    if node.type == "template_type":
        return _scope_name(_field(node, "name") or _named(node, "type_identifier"), src)
    return None


def _split_qualified(node, src: bytes) -> tuple[str | None, str | None]:
    """`(owner, name)` for a declarator name that may carry `::`.

    `testing::internal::UnitTestImpl::Foo` nests to the right, so the LAST
    scope walked is the owning class and the rest are namespaces. Namespaces
    are deliberately dropped: a method id has to be `prefix_class_method` for
    the resolver to find it, and a namespace in the middle would make every
    call on that class unresolvable.
    """
    owner = None
    invented = False
    while node is not None and node.type == "qualified_identifier":
        if _written(node):
            got = _scope_name(node.children[0] if node.children else None, src)
            if got:
                owner = got
        else:
            invented = True
        node = _field(node, "name")
    # One invented `::` anywhere makes the whole chain a guess.
    # `FMT_API std::system_error vwindows_error(...)` splits into a real `::`
    # after `std` and an invented one before the name, and reading only the
    # real one made a free function into a method of `std`.
    return (None if invented else owner), _leaf_name(node, src)


def _outer_scope(node, src: bytes) -> str | None:
    """The leftmost namespace of a qualified name: the `std` in `std::string`."""
    if node is None or node.type != "qualified_identifier" or not _written(node):
        return None
    return _scope_name(node.children[0] if node.children else None, src)


def _from_std(node, src: bytes) -> bool:
    """True for a name that lives in the standard library.

    `std` is the one namespace a corpus can never define -- the language
    reserves it -- so a `std::` name is outside the map by construction.
    Dropping the qualifier without this check turned `std::string(str, comma)`
    into a call to a corpus method that happened to be called `string`, which
    is the C++ shape of the same wrong link the Python builtin filter exists
    to stop.

    Only `std` is treated this way. Every other qualifier -- `detail::`,
    `testing::`, `internal::` -- is usually a namespace INSIDE the corpus, and
    those calls do resolve once the qualifier is dropped: 767 of the 1,991
    qualified calls in these two corpora are namespace-qualified that way.
    """
    return _outer_scope(node, src) == "std"


def _written(qualified) -> bool:
    """True when the `::` in a qualified name is actually in the source.

    Error recovery invents a zero-width `::` to make sense of two type names in
    a row, so `CURL_EXTERN curl_socket_t curl_dbg_socket(...)` -- a macro, a
    return type and a name -- arrives as `curl_socket_t::curl_dbg_socket`.
    Believing it gave curl three methods on a class that is a socket handle.
    """
    for child in qualified.children:
        if child.type == "::":
            return child.end_byte > child.start_byte
    return False


def _leaf_name(node, src: bytes) -> str | None:
    if node is None:
        return None
    t = node.type
    if t in ("identifier", "field_identifier", "type_identifier",
             "namespace_identifier"):
        return _text(node, src)
    if t in ("destructor_name", "operator_name"):
        # `~Server` and `operator ==` are real names and two of them in one
        # class are two different methods; collapsing whitespace keeps
        # `operator ==` and `operator==` minting one id.
        return "".join(_text(node, src).split())
    if t == "template_function":
        return _leaf_name(_field(node, "name") or (node.children[0] if node.children
                                                   else None), src)
    if t == "qualified_identifier":
        return _split_qualified(node, src)[1]
    return None


def _peel(decl) -> tuple[object, bool, bool]:
    """Walk a declarator down to the name it declares.

    Returns `(leaf, is_function, through_parens)`. `through_parens` matters:
    `void (*cb)(void*)` has a parameter list and is NOT a function -- it is a
    function POINTER, and the parentheses are the only thing that says so.
    Without the check, every callback field in a C struct became a method.
    """
    is_function = False
    through_parens = False
    for _ in range(24):                            # a declarator chain is short
        if decl is None:
            break
        t = decl.type
        if t == "function_declarator":
            is_function = True
            decl = _field(decl, "declarator")
        elif t == "parenthesized_declarator":
            through_parens = True
            decl = _named(decl, *_DECLARATORS)
        elif t in _WRAPPERS:
            decl = _field(decl, "declarator") or _named(decl, *_DECLARATORS)
        else:
            break
    return decl, is_function, through_parens


def _record_spec(type_node):
    """The class, struct, union or enum a `type` field holds, if any.

    `placeholder_type_specifier` has to be looked through. Recovering from a
    class whose body ends in a way it did not expect, the grammar files the
    whole `class basic_format_args { ... }` under `auto`, and testing the type
    field directly lost fmt's `basic_format_args`, `basic_memory_buffer` and
    every other class written the same way.
    """
    node = type_node
    for _ in range(4):
        if node is None:
            return None
        if node.type in _RECORDS or node.type == "enum_specifier":
            return node
        if node.type == "placeholder_type_specifier":
            node = node.children[0] if node.children else None
            continue
        return None
    return None


def _is_macro_call(type_node, leaf, home: str | None, name: str | None) -> bool:
    """True when a "declaration" is really a macro invocation.

    The test is a language rule, not a naming convention. A declaration with no
    return type can only be a constructor, a destructor or a conversion
    operator, and a constructor's name is its class's name. So:

        BIT(secure);                    inside struct Cookie   -> macro
        GTEST_DISALLOW_COPY_(Message);  inside class Message   -> macro
        Cookie();                       inside struct Cookie   -> constructor
        TEST(HeapTest, Grows) { ... }   at file scope          -> macro

    Both were live defects. curl's `BIT(x)` bitfield macro put 274 methods
    called `BIT` into the graph, and googletest's `TEST` macro put 294
    different test bodies onto one node.
    """
    if type_node is not None or leaf is None:
        return False
    if leaf.type not in ("identifier", "field_identifier"):
        return False                               # ~Foo, Foo::Foo, operator==
    return name != home


def _declared(node):
    """The declarator children of a declaration, and only those.

    The type is skipped explicitly rather than by node type. `Point p;` and
    `typedef int Handle;` both put a bare `type_identifier` where a declarator
    goes, so a type-name test cannot tell them apart -- and reading the type as
    a declarator recorded `Point` as a variable of type `Point`.
    """
    skip = _field(node, "type")
    # Compared by extent, not by identity: the bindings hand back a fresh Node
    # object on every field access, so `is` was never true and the type came
    # through as a declarator anyway.
    span = (skip.start_byte, skip.end_byte) if skip is not None else None
    for child in node.children:
        if child.type in _DECLARATORS and (child.start_byte, child.end_byte) != span:
            yield child


def _type_name(node, src: bytes) -> str | None:
    """The class named by a type, or None when the type is not a class.

    `int`, `auto` and `void` name no class, and returning a name for them would
    let the resolver "prove" a call on a variable whose type it does not know.
    """
    if node is None:
        return None
    t = node.type
    if t == "type_identifier":
        return _text(node, src)
    if t == "template_type":
        return _type_name(_field(node, "name") or _named(node, "type_identifier"), src)
    if t == "qualified_identifier":
        # `detail::buffer` is a corpus class and resolves once the namespace is
        # dropped. `std::string` is not, and typing a variable with it made
        # every `s.size()` resolve to whatever corpus method shared the name.
        if _from_std(node, src):
            return None
        inner = _field(node, "name")
        return _type_name(inner, src) if inner is not None else None
    if t in _RECORDS or t == "enum_specifier":
        return _leaf_name(_field(node, "name"), src)
    return None


def _bases(node, src: bytes) -> list[str]:
    """What a class extends. `public`, `private` and `virtual` are access, not
    identity, so only the type names are kept."""
    clause = _named(node, "base_class_clause")
    if clause is None:
        return []
    out = []
    for child in clause.children:
        found = _scope_name(child, src) if child.type != "qualified_identifier" \
            else _split_qualified(child, src)[1]
        if found:
            out.append(found)
    return out


def _doc_above(node, src: bytes) -> str | None:
    """The comment run directly above a declaration.

    C++ takes every comment form, unlike TypeScript where only `/** */` counts.
    The reason is measured in the corpora: googletest documents its entire
    public API with plain `//` runs and never writes a `/** */` block, so a
    JSDoc-style rule would leave that codebase with no "why" at all.

    Only a comment on the line immediately above is documentation. The check
    has to apply to the first comment too -- guarding it on "we already have
    lines" lets any single detached comment through.
    """
    lines: list[str] = []
    prev = node.prev_sibling
    while prev is not None and prev.type == "comment":
        if prev.end_point[0] + 1 < node.start_point[0]:
            break
        lines.insert(0, _clean_comment(_text(prev, src)))
        node, prev = prev, prev.prev_sibling
    joined = " ".join(l for l in lines if l).strip()
    return " ".join(joined.split()) or None


def _clean_comment(text: str) -> str:
    if text.startswith("/*"):
        body = text[2:]
        if body.endswith("*/"):
            body = body[:-2]
        return " ".join(l.strip().lstrip("*!").strip() for l in body.splitlines())
    return text.lstrip("/").lstrip("!<").strip()


class _Reader:
    """One file's worth of extraction.

    Two passes, and the second is the reason for the split. Pass one emits every
    declaration and remembers which names each class owns; pass two walks the
    bodies. An unqualified `Foo()` inside a member function is C++'s `self.foo()`
    -- but only when the class really has a `Foo`, and that is not known until
    the whole class has been read.
    """

    def __init__(self, parsed: ParsedFile, src: bytes):
        self.p = parsed
        self.src = src
        self.members: dict[str, set[str]] = {}
        # Classes THIS file defines, by name -> node id. `members` is not the
        # same thing: an out-of-line definition adds its owner there without
        # the class being here, and hanging a `contains` edge on that name
        # would point at a node nothing created.
        self.class_ids: dict[str, str] = {}
        # (body, caller node id, owner class or None), read in pass two
        self.bodies: list[tuple] = []

    # ---- emitting -------------------------------------------------------

    def emit(self, name: str, kind: str, node, container: str,
             doc_node=None, scope=None, bases=None) -> str:
        nid = mint(self.p.prefix, name, scope or [])
        line = node.start_point[0] + 1
        self.p.nodes.append(Node(id=nid, label=name, kind=kind, file=self.p.path,
                                 line=line, bases=bases))
        self.p.edges.append(Edge(source=container, target=nid, relation="contains",
                                 file=self.p.path, line=line))
        doc = _doc_above(doc_node if doc_node is not None else node, self.src)
        if doc:
            doc_id = f"{nid}{DOC_SUFFIX}"
            self.p.nodes.append(Node(id=doc_id, label=f"docstring of {nid}",
                                     kind="rationale", file=self.p.path, line=line,
                                     text=doc[:600]))
            self.p.edges.append(Edge(source=nid, target=doc_id,
                                     relation="rationale_for",
                                     file=self.p.path, line=line))
        return nid

    # ---- pass one: declarations ----------------------------------------

    def walk(self, children, container: str, owner: str | None) -> None:
        """Read a run of siblings at declaration position.

        `container` is what a `contains` edge should come from; `owner` is the
        class whose body we are in, or None at file scope.
        """
        for outer in children:
            node = outer
            if node.type == "template_declaration":
                # fmt is almost entirely templates. Without unwrapping, the file
                # yields nothing at all.
                node = _template_body(node)
                if node is None:
                    continue
            t = node.type
            if t in _TRANSPARENT:
                self.walk(node.children, container, owner)
            elif t == "preproc_include":
                self._include(node)
            elif t in _RECORDS:
                self._record(node, outer, container, owner)
            elif t == "enum_specifier":
                self._enum(node, outer, container, owner)
            elif t == "alias_declaration":
                self._alias(node, outer, container, owner)
            elif t == "type_definition":
                self._typedef(node, outer, container, owner)
            elif t == "function_definition":
                self._function(node, outer, container, owner)
            elif t in ("declaration", "field_declaration"):
                self._declaration(node, outer, container, owner)

    def _record(self, node, outer, container: str, owner: str | None) -> None:
        name = _leaf_name(_field(node, "name"), self.src)
        body = _field(node, "body")
        if body is None:
            # `class Foo;` is a promise, not a definition. Emitting it would
            # mint the id the real definition needs and claim the line of the
            # forward declaration instead.
            return
        if not name:
            # An anonymous struct has no name to mint from. Its members are
            # reachable only through whatever it was typedef'd or declared as.
            return
        nid = self.emit(name, "class", node, container, doc_node=outer,
                        bases=_bases(node, self.src) or None)
        self.p.defined_classes.add(name)
        self.members.setdefault(name, set())
        self.class_ids.setdefault(name, nid)
        # Nested classes keep flat ids -- `prefix_inner`, not `prefix_outer_inner`
        # -- because the resolver reconstructs a class id as prefix plus label.
        # Containment still records the nesting.
        self.walk(body.children, nid, name)

    def _enum(self, node, outer, container: str, owner: str | None) -> None:
        name = _leaf_name(_field(node, "name"), self.src)
        if not name or _field(node, "body") is None:
            return
        self.emit(name, "class", node, container, doc_node=outer)
        self.p.defined_classes.add(name)

    def _alias(self, node, outer, container: str, owner: str | None) -> None:
        """`using Alias = Server;` -- a named type, so a class node."""
        name = _leaf_name(_named(node, "type_identifier"), self.src)
        if name:
            self.emit(name, "class", node, container, doc_node=outer)
            self.p.defined_classes.add(name)

    def _typedef(self, node, outer, container: str, owner: str | None) -> None:
        """`typedef struct rax { ... } rax;` is C's way of naming a type.

        The struct and the alias usually share a name; emitting both would mint
        one id twice and report a collision against itself.
        """
        inner = _named(node, *_RECORDS, "enum_specifier")
        inner_name = None
        if inner is not None and _field(inner, "body") is not None:
            inner_name = _leaf_name(_field(inner, "name"), self.src)
            if inner_name:
                nid = self.emit(inner_name, "class", inner, container, doc_node=outer,
                                bases=_bases(inner, self.src) or None)
                self.p.defined_classes.add(inner_name)
                self.members.setdefault(inner_name, set())
                self.class_ids.setdefault(inner_name, nid)
                if inner.type in _RECORDS:
                    self.walk(_field(inner, "body").children, nid, inner_name)
        for decl in _declared(node):
            leaf, _, _ = _peel(decl)
            name = _leaf_name(leaf, self.src)
            if name and name != inner_name:
                self.emit(name, "class", node, container, doc_node=outer)
                self.p.defined_classes.add(name)

    def _function(self, node, outer, container: str, owner: str | None) -> None:
        declarator = _field(node, "declarator")
        body = _field(node, "body")
        type_node = _field(node, "type")

        leaf, is_function, through_parens = _peel(declarator)
        spec = _record_spec(type_node)
        if spec is not None and _field(spec, "body") is not None:
            # A type definition the grammar filed as a function because of what
            # came after it. The type is the real declaration; whatever the
            # declarator says is wreckage.
            if spec.type == "enum_specifier":
                self._enum(spec, outer, container, owner)
            else:
                self._record(spec, outer, container, owner)
            return
        # A macro the preprocessor would have expanded reads as a definition
        # whose "return type" is the macro name and whose body is everything
        # after it -- `FMT_BEGIN_NAMESPACE` opens two namespaces and swallows the
        # rest of the header that way. The tell is exact: a function definition
        # without a parameter list is not a function definition, so the body is
        # walked for the declarations that were really in it.
        if not is_function or through_parens:
            if spec is not None and spec.type in _RECORDS and body is not None \
                    and leaf is not None and leaf.type == "identifier":
                self._macro_record(node, leaf, body, outer, container)
                return
            if body is not None:
                self.walk(body.children, container, owner)
            return

        qualified, name = (_split_qualified(leaf, self.src) if leaf is not None
                           and leaf.type == "qualified_identifier"
                           else (None, _leaf_name(leaf, self.src)))
        if not name:
            return
        if _is_macro_call(type_node, leaf, qualified or owner, name):
            # The block after the macro is still real code, so it is walked for
            # the declarations inside it. The stated cost is one-sided: calls in
            # a googletest `TEST` body lose their caller and are not recorded.
            if body is not None:
                self.walk(body.children, container, owner)
            return
        # `void Server::Start()` states its owner; inside a class body the owner
        # is where we are standing.
        home = qualified or owner
        if home:
            nid = self.emit(name, "method", node, self._home(home, container),
                            doc_node=outer, scope=[home])
            self.p.owner_of[nid] = home
            self.members.setdefault(home, set()).add(name)
        else:
            nid = self.emit(name, "function", node, container, doc_node=outer)
        self._types_in(node, f"{home}.{name}" if home else name)
        if body is not None:
            self.bodies.append((body, nid, home))

    def _macro_record(self, node, leaf, body, outer, container: str) -> None:
        """`class FMT_API file { ... };` -- an export macro before the name.

        The grammar reads the macro as the class's name and the real name as a
        variable after it, so the class disappears and every member leaks out to
        file scope as a free function. Measured on these corpora, 48 classes are
        written this way -- `class GTEST_API_ UnitTest`, `class FMT_API file`.

        The body arrives as a statement block rather than a member list, which
        is why `public:` shows up as a label; `walk` looks through those.
        """
        name = _text(leaf, self.src)
        nid = self.emit(name, "class", node, container, doc_node=outer,
                        bases=self._wrecked_bases(node) or None)
        self.p.defined_classes.add(name)
        self.members.setdefault(name, set())
        self.class_ids.setdefault(name, nid)
        self.walk(body.children, nid, name)

    def _wrecked_bases(self, node) -> list[str]:
        """`: public Base` after a name the grammar has already given up on.

        It lands in a recovery node instead of a base_class_clause. Reading it
        is worth the fragility: what a class extends is part of what it is, and
        these are real classes.
        """
        out = []
        for child in node.children:
            if child.type != "ERROR" or not child.children:
                continue
            if child.children[0].type != ":":
                continue
            for token in child.children[1:]:
                if token.type in ("identifier", "type_identifier"):
                    out.append(_text(token, self.src))
        return out

    def _home(self, cls: str, container: str) -> str:
        """Where a method's `contains` edge starts.

        An out-of-line definition names a class this file may not define. Hanging
        the edge on a class node that was never created is a link an agent
        follows to nothing, so it falls back to the file.
        """
        return self.class_ids.get(cls, container)

    def _declaration(self, node, outer, container: str, owner: str | None) -> None:
        """A declaration with no body: a prototype, a member, or a variable.

        Prototypes carry the whole public surface of a C header and most of the
        method list of a C++ class, so they are nodes. A variable is not one of
        the five kinds and is recorded only as a type, for resolution.
        """
        type_node = _field(node, "type")
        if type_node is not None and type_node.type in _RECORDS \
                and _field(type_node, "body") is not None:
            # `struct S { ... } instance;` defines a type and a variable at once.
            self._record(type_node, outer, container, owner)
        for decl in _declared(node):
            leaf, is_function, through_parens = _peel(decl)
            name = _leaf_name(leaf, self.src)
            if not name:
                continue
            if is_function and not through_parens:
                qualified = (_split_qualified(leaf, self.src)[0]
                             if leaf.type == "qualified_identifier" else None)
                home = qualified or owner
                if _is_macro_call(type_node, leaf, home, name):
                    continue
                if not home:
                    # A free prototype -- `void listRelease(list*);` in a header
                    # -- names a function defined in another file. Emitting it
                    # would put two nodes with one name in the corpus, and the
                    # resolver refuses a name it finds in two places: every
                    # cross-file call to it would turn from an edge into a gap.
                    # A member declaration is different and is kept, because a
                    # C++ class lists its methods in the header and defines them
                    # in the source -- without it the class arrives empty.
                    continue
                nid = self.emit(name, "method", node, self._home(home, container),
                                doc_node=outer, scope=[home])
                self.p.owner_of[nid] = home
                self.members.setdefault(home, set()).add(name)
                continue
            found = _type_name(type_node, self.src)
            if found and owner:
                # A member's type is declared, not inferred. It is written to
                # both maps because `this->buf_.append()` and a bare
                # `buf_.append()` are the same call and the resolver reads them
                # through different keys.
                self.p.attr_types[f"{owner}::{name}"] = found
                self.p.var_types[f"{owner}::{name}"] = found

    def _include(self, node) -> None:
        """`#include "x.h"` becomes an edge to the file it names.

        A quoted include is resolved against the including file's own directory,
        which is what the compiler does first: `lib/url.c` including
        `"curl_setup.h"` means `lib/curl_setup.h`. The extension stays as its own
        segment because it is part of the file's identity here.

        `parsed.imports` is deliberately left empty. It maps a local NAME onto a
        module, and an include binds no name -- filling it with header paths
        would make the resolver's "imported" rule fire on the letter `h`.
        """
        quoted = _named(node, "string_literal")
        system = _named(node, "system_lib_string")
        if quoted is not None:
            raw = _text(quoted, self.src).strip('"')
            here = self.p.path.replace("\\", "/").rsplit("/", 1)
            parts = (here[0].split("/") if len(here) > 1 else [])
            for piece in raw.split("/"):
                if piece == "." or not piece:
                    continue
                if piece == "..":
                    if parts:
                        parts.pop()
                    continue
                parts.append(piece)
            dotted = ".".join(parts)
        elif system is not None:
            # `<vector>` leaves the corpus. Recording it unresolved is the point:
            # which libraries a file depends on is a question worth answering.
            dotted = _text(system, self.src).strip("<>").replace("/", ".")
        else:
            return
        if dotted:
            self.p.import_sites.append((dotted, node.start_point[0] + 1))

    # ---- pass two: bodies ----------------------------------------------

    def finish(self) -> None:
        for body, caller, home in self.bodies:
            self._calls_in(body, caller, home)

    def _types_in(self, node, scope: str) -> None:
        """Every local whose type the source states outright.

        Three forms, all declarations rather than guesses:

            void f(Server* s)     a typed parameter
            Server s;             a declaration
            Server* s = new ...   a declaration with an initialiser

        `auto` is skipped on purpose. It states that the compiler knows the
        type, not that we do.
        """
        stack = list(node.children)
        while stack:
            n = stack.pop()
            if n.type in ("parameter_declaration", "declaration",
                          "optional_parameter_declaration"):
                found = _type_name(_field(n, "type"), self.src)
                if found:
                    for decl in _declared(n):
                        leaf, is_function, _ = _peel(decl)
                        name = _leaf_name(leaf, self.src)
                        if name and not is_function:
                            self.p.var_types[f"{scope}::{name}"] = found
            if n.type not in ("function_definition",):
                stack.extend(n.children)

    def _calls_in(self, body, caller: str, home: str | None) -> None:
        stack = [body]
        while stack:
            node = stack.pop()
            if node.type == "call_expression":
                self._call(node, caller, home)
            elif node.type == "new_expression":
                # `new Server()` is often the only edge tying a factory to what
                # it builds, exactly as it is in TypeScript.
                found = _type_name(_field(node, "type"), self.src)
                if found:
                    self.p.calls.append(CallSite(
                        caller=caller, file=self.p.path,
                        line=node.start_point[0] + 1, name=found))
            stack.extend(node.children)

    def _call(self, node, caller: str, home: str | None) -> None:
        fn = node.children[0] if node.children else None
        if fn is None:
            return
        if fn.type in ("identifier", "template_function"):
            name = _leaf_name(fn, self.src)
            if not name:
                return
            # An unqualified call inside a member function is C++'s `self.foo()`
            # -- but only when the class really has that member. Marking every
            # bare call as a self-call refuses each one that is actually a free
            # function, which is most of them.
            on_self = bool(home) and name in self.members.get(home, set())
            self.p.calls.append(CallSite(
                caller=caller, file=self.p.path, line=node.start_point[0] + 1,
                name=name, on_self=on_self))
        elif fn.type == "field_expression":
            what = _field(fn, "field") or _named(fn, "field_identifier")
            name = _leaf_name(what, self.src)
            if not name:
                return
            obj = _field(fn, "argument") or (fn.children[0] if fn.children else None)
            on_self = obj is not None and obj.type == "this"
            receiver = None
            receiver_is_self = False
            if not on_self and obj is not None:
                if obj.type == "identifier":
                    receiver = _text(obj, self.src)
                elif obj.type == "field_expression":
                    inner = _field(obj, "argument") or (obj.children[0] if obj.children
                                                        else None)
                    if inner is not None and inner.type == "this":
                        # `this->buf_.append()`: the receiver is a member whose
                        # type the class declared.
                        receiver = _leaf_name(_field(obj, "field")
                                              or _named(obj, "field_identifier"),
                                              self.src)
                        receiver_is_self = receiver is not None
            self.p.calls.append(CallSite(
                caller=caller, file=self.p.path, line=node.start_point[0] + 1,
                name=name, receiver=receiver, receiver_is_self=receiver_is_self,
                on_self=on_self))
        elif fn.type == "qualified_identifier":
            # `Class::method()` and `ns::free()` are the same shape and the
            # graph cannot tell them apart, so the qualifier is dropped and the
            # call is recorded by name. Stated cost: we know more about a static
            # call than we can express, and it resolves only when the name is
            # unambiguous in the corpus.
            if _from_std(fn, self.src):
                return
            name = _split_qualified(fn, self.src)[1]
            if name:
                self.p.calls.append(CallSite(
                    caller=caller, file=self.p.path,
                    line=node.start_point[0] + 1, name=name))


def _template_body(node):
    """The declaration a `template <...>` wraps.

    Returned without the template header so the rest of the reader never has to
    know about templates -- but the caller keeps the outer node for the doc
    comment, which sits above `template`, not above the declaration.
    """
    for child in node.children:
        if child.type in ("template", "template_parameter_list", "requires_clause",
                          "comment", ";"):
            continue
        if child.type == "template_declaration":
            return _template_body(child)
        return child
    return None


def parse(source: str, parsed: ParsedFile) -> bool:
    if not available():                            # pragma: no cover
        return False
    src = source.encode("utf-8")
    root = _parser.parse(src).root_node
    if root.has_error and not root.children:
        return False
    reader = _Reader(parsed, src)
    reader.walk(root.children, parsed.prefix, None)
    reader.finish()
    return True
