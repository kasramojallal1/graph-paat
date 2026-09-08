"""C, read with tree-sitter.

C has no classes and no methods, so the mapping is the flattest of any
language here: **a struct, union or enum is a `class` node, and every function
is a plain `function` node.** A typedef over one of those is the same single
type, carried under the typedef's name because that is what the code writes.
Nothing becomes a method. A struct full of
function pointers looks like an object with methods and is not one -- the
pointer is a field whose value is chosen at run time, and inventing a method
node for it would put a definition in the map that the source does not contain.

Four decisions cost something and were measured rather than assumed.

**The extension is part of a file's identity** (`KEEP_EXTENSION`). `url.c` and
`url.h` are two files, not one. Dropping the extension made curl's `lib/` lose
159 of 385 files to a shared name and redis `src/` 67 of 218, so ids here are
`url_c` and `url_h`.

**A function prototype is not a node.** A header's `void listRelease(list *);`
declares a function defined elsewhere, and emitting it would put two nodes with
one name in the corpus -- which turns every cross-file call to that name from a
resolved edge into a refusal. Measured on these corpora: 2,788 of redis's 5,708
function names carry a prototype somewhere, and 1,146 of curl's 3,559. A file's
own forward declarations are worse still, because they break resolution *inside*
the file: redis has 380 names declared and then defined in the same `.c`. The
cost is real and stated -- a header that only declares functions contributes its
types and its doc comments, not its function list.

**A function-like macro is not a node either.** `#define listLength(l) ((l)->len)`
is a callable name and it was tempting. But a macro is routinely defined twice
in one file, once per platform, behind `#ifdef`; that pattern alone would mint
81 colliding ids on redis and 102 on curl -- 13% and 20% of all macro
definitions -- and on curl 120 macro names shadow a real function of the same
name. A node that silently merges with another is worse than an absent one.

**A failed parse is read anyway.** C's preprocessor lets a `#ifdef` open a
brace in one arm and close it in another, which no grammar can parse; when that
happens tree-sitter collapses everything after it into one ERROR node. Reading
only well-formed children then reads the whole file as empty. Measured on curl,
five `.c` files -- `cf-haproxy.c`, `curlx/timeval.c`, `http.c`, `vauth/gsasl.c`
and `vtls/apple.c` -- produced zero function nodes for this reason, and `http.c`
alone defines 98 of them; two more, `cf-socket.c` and `vtls/vtls.c`, lost
everything after the break without going empty. `_salvage` reaches into the
wreckage for the declarations that are still intact -- 182 functions and 6 types
on curl, one function on redis. What it cannot recover is the body of the one
function the break lands in, so calls from that function are lost.

What C does give, and no dynamic language does, is **a declared type on every
local variable**: `struct connectdata *conn;` states what `conn` is. That goes
into `var_types`. It resolves no calls on its own here (there are no methods to
land on), but it is the only receiver typing C offers, and it is what makes a
`conn->handler->done()` refusal say "receiver typed, no such method" instead of
"receiver type unknown".

An `#include` is an import. `#include "curl_setup.h"` inside `vtls/openssl.c` is
emitted as `vtls.curl_setup.h`: the including file's own directory first, then
the include path, then the extension as its own segment. Resolution tries the
whole dotted string and then drops leading segments, so the sibling header wins
when there is one and the root-relative header wins otherwise -- which is
exactly C's own search order. `#include <stdio.h>` is emitted as `stdio.h` and
stays unresolved, which is the true answer; the nine cases per corpus where a
system header shares a filename with a corpus one are noted at `_include`.
"""
from __future__ import annotations

from ..ids import mint
from ..parse import DOC_SUFFIX, CallSite, Edge, Node, ParsedFile

EXTENSIONS = {".c", ".h"}
WHY = ""

# `url.c` and `url.h` are two files. See the note above; nothing outside this
# module knows which languages say this about themselves.
KEEP_EXTENSION = True

_parser = None


def available() -> bool:
    """True when the grammar is installed. The registry skips us otherwise, so
    a Python-only user never has to carry a C grammar."""
    global _parser, WHY
    if _parser is not None:
        return True
    try:
        import tree_sitter_c
        from tree_sitter import Language, Parser
        _parser = Parser(Language(tree_sitter_c.language()))
        return True
    except Exception as exc:                       # pragma: no cover
        WHY = f"needs tree-sitter and tree-sitter-c ({exc})"
        return False


# A declaration is at "top level" even when it sits inside `#ifndef GUARD` --
# which, in C, nearly all of them do. Every header opens with an include guard,
# so reading only `root.children` finds the guard and nothing else: adlist.h's
# three structs and its whole prototype list are children of one preproc_ifdef.
_CONDITIONAL = {"preproc_ifdef", "preproc_if", "preproc_else", "preproc_elif",
                "preproc_elifdef", "preproc_elifndef"}

# The declarator of a declaration: the part after the type. `list *listCreate()`
# puts a `type_identifier` for the return type in front of it, so a type name is
# deliberately NOT in this set -- including it made `_spine` return the return
# type and every pointer-returning function in redis was named `list`.
_SPINE = ("function_declarator", "pointer_declarator", "init_declarator",
          "parenthesized_declarator", "array_declarator",
          "attributed_declarator", "identifier")

# Descending INSIDE a declarator is the other way round: `typedef int (*cb)(void
# *)` spells the name it declares as a `type_identifier`, because at that point
# in the grammar `cb` is becoming a type.
_DECLARATORS = _SPINE + ("field_identifier", "type_identifier")

_TAGGED = ("struct_specifier", "union_specifier", "enum_specifier")

# Everything `parse` knows how to read. Named once because recovery has to look
# for the same things; see `_salvage`.
_WANTED = ("preproc_include", "function_definition", "type_definition",
           "declaration") + _TAGGED


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _named(node, *types):
    """First direct child of any of these types."""
    if node is None:
        return None
    for child in node.children:
        if child.type in types:
            return child
    return None


def _declarations(node):
    """Every declaration at this level, looking through `#if` / `#ifdef` and
    through the wreckage of a failed parse.

    Both branches of an `#ifdef` are read. The preprocessor picks one; we are
    building a map of what the source contains, and a Windows-only function is
    still a function someone will ask about.
    """
    out = []
    for child in node.children:
        if child.type in _CONDITIONAL:
            out.extend(_declarations(child))
        elif child.type == "ERROR":
            out.extend(_salvage(child))
        else:
            out.append(child)
    return out


def _salvage(node):
    """The declarations still standing inside a recovery node.

    A `#ifdef` may open a brace in one arm and close it in another. It is legal
    C -- the preprocessor runs first and only one arm is ever compiled -- and
    it is not parseable syntax, so the grammar gives up and flattens everything
    after it into one ERROR. curl's `cf-haproxy.c:78` is the shape: `else {`
    sits inside the `#ifdef USE_UNIX_SOCKETS` opened at line 74, and its `}` at
    line 100 sits inside a second one.

    The damage is not local. That one brace collapsed lines 26-241 into a
    single ERROR child of the root, and since an ERROR is not a declaration,
    the file arrived in the graph as nothing but its own file node. Measured on
    curl, five `.c` files were empty for this reason -- `http.c` among them,
    which defines 98 functions.

    What survives inside the wreckage is still a well-formed subtree, just at
    the wrong depth, so this reaches to any depth for the things `parse` reads.
    It stops as soon as it finds one: descending INTO a recovered
    `function_definition` would hand that function's local variables back as
    file-scope declarations. The cost of reaching this deep is that a `struct`
    or a `declaration` written inside a function body now reads as a file-level
    one -- true of the wreckage only, and a type that exists in the file is a
    smaller error than a file that reads as empty.
    """
    out = []
    for child in node.children:
        if child.type in _WANTED:
            out.append(child)
        elif child.type == "function_declarator":
            # The header of a function the grammar could not assemble; see
            # `_wrecked_function`. Descending into one instead would only reach
            # its parameter list.
            if _opens_a_body(child):
                out.append(child)
        else:
            out.extend(_salvage(child))
    return out


def _opens_a_body(declarator) -> bool:
    """True when a `{` follows a declarator, which makes it a definition.

    This is the whole test that separates a wrecked definition from a
    prototype, and it is the language's own: `void f(void);` ends in a
    semicolon, `void f(void) {` does not.
    """
    sibling = declarator.next_sibling
    while sibling is not None and sibling.type == "comment":
        sibling = sibling.next_sibling
    return sibling is not None and sibling.type == "{"


def _declarator_name(node, src: bytes) -> str | None:
    """The name a declarator declares, down through pointers and parentheses.

    Only ever follows the FIRST declarator-shaped child, which is the spine.
    Following any matching descendant would read `listInsertNode(list *list,
    ...)` and return the first parameter's name.
    """
    cur = node
    while cur is not None:
        if cur.type in ("identifier", "field_identifier", "type_identifier"):
            return _text(cur, src)
        cur = _named(cur, *_DECLARATORS)
    return None


def _spine(node):
    """The declarator of a definition, skipping the type in front of it."""
    for child in node.children:
        if child.type in _SPINE:
            return child
    return None


def _type_name(node, src: bytes) -> str | None:
    """The corpus type a declaration states, or None for a builtin one.

    `listNode *n` names listNode; `struct connectdata *conn` names connectdata;
    `unsigned long len` and `int x` name nothing worth recording, because no
    node in the graph can ever be a `long`.
    """
    for child in node.children:
        if child.type == "type_identifier":
            return _text(child, src)
        if child.type in _TAGGED:
            tag = _named(child, "type_identifier")
            return _text(tag, src) if tag is not None else None
        if child.type in ("primitive_type", "sized_type_specifier"):
            return None
    return None


def _clean(text: str) -> str:
    """One comment's prose, with the comment syntax taken off.

    Both forms count as documentation in C, unlike TypeScript where only the
    `/** */` block does. C has no dedicated doc syntax, so a `/* */` block above
    a function and a run of `//` lines above one are the same act; refusing
    either would leave most of a real C file with no "why" at all.
    """
    if text.startswith("/*"):
        body = text[2:]
        if body.endswith("*/"):
            body = body[:-2]
        lines = [line.strip().lstrip("*").strip() for line in body.splitlines()]
    else:
        lines = [line.lstrip("/").strip() for line in text.splitlines()]
    return " ".join(line for line in lines if line)


def _doc_above(node, src: bytes) -> str | None:
    """The run of comments directly above a declaration."""
    parts: list[str] = []
    prev = node.prev_sibling
    while prev is not None and prev.type == "comment":
        # A comment on the line immediately above documents the declaration; one
        # separated by a blank line is a note about something else. adlist.h:14
        # is the case that matters -- "Node, List, and Iterator are the only
        # data structures used currently" sits two lines above `typedef struct
        # listNode`, and it describes the file, not that struct.
        if prev.end_point[0] + 1 < node.start_point[0]:
            break
        cleaned = _clean(_text(prev, src))
        if cleaned:
            parts.insert(0, cleaned)
        node, prev = prev, prev.prev_sibling
    joined = " ".join(parts).strip()
    return joined or None


def _include_path(raw: str, own_dir: list[str]) -> str:
    """A quoted include, made absolute against the including file's directory.

    C looks for `"foo.h"` beside the including file first and on the include
    path second. Emitting the directory-qualified form gets both, because
    resolution tries the whole dotted string and then drops leading segments:
    `#include "urldata.h"` in `vtls/openssl.c` becomes `vtls.urldata.h`, misses
    `vtls_urldata_h`, and lands on `urldata_h`.
    """
    parts = own_dir + [p for p in raw.replace("\\", "/").split("/") if p and p != "."]
    out: list[str] = []
    for part in parts:
        if part == "..":
            # A `..` that walks above the corpus root has nothing to point at;
            # dropping it leaves the tail, which resolution will fail to match
            # and report honestly. redis's `modules/hellotype.c` includes
            # `"../redismodule.h"`, which does resolve, by this path.
            if out:
                out.pop()
        else:
            out.append(part)
    if not out:
        return ""
    name = out[-1]
    # The extension becomes its own dotted segment so it survives into the id:
    # `url.h` must reach `url_h`, not `url`.
    return ".".join(out[:-1] + name.rsplit(".", 1)) if "." in name else ".".join(out)


class _Reader:
    def __init__(self, parsed: ParsedFile, src: bytes):
        self.p = parsed
        self.src = src
        self.seen: set[str] = set()

    def emit(self, name: str, kind: str, node, doc_node=None) -> str:
        nid = mint(self.p.prefix, name)
        # Every arm of an `#if` chain is read, so one name can arrive several
        # times: curl's `curl_ed25519.c` defines `Curl_ed25519_sign` four times,
        # once per TLS backend, and redis's `defrag.c` defines
        # `activeDefragAlloc` twice. That is ONE symbol with several
        # implementations -- the preprocessor keeps exactly one and we cannot
        # know which -- so the first wins and the rest are dropped rather than
        # minting a colliding id. Measured: doing nothing here would claim one
        # id twice or more 69 times on redis and 168 times on curl.
        if nid in self.seen:
            return nid
        self.seen.add(nid)
        line = node.start_point[0] + 1
        self.p.nodes.append(Node(id=nid, label=name, kind=kind, file=self.p.path,
                                 line=line))
        self.p.edges.append(Edge(source=self.p.prefix, target=nid,
                                 relation="contains", file=self.p.path, line=line))
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

    def local_types(self, node, scope: str) -> None:
        """What every local and parameter in a function IS, as C states it.

        This is a declaration, not an inference: `struct connectdata *conn;` at
        the top of a function is the source saying so. Python's equivalent has
        to be guessed from an assignment and is wrong whenever a factory is
        involved.
        """
        stack = list(node.children)
        while stack:
            n = stack.pop()
            # Locals and parameters only. A `field_declaration` inside a struct
            # declared in this body names a member, not a variable, and the
            # resolver matches a receiver against any binding of that name in
            # the file -- so a field called `l` would have typed an unrelated
            # local `l`.
            if n.type in ("declaration", "parameter_declaration"):
                found = _type_name(n, self.src)
                if found:
                    # One declaration can name several variables of one type:
                    # `listNode *current, *next;` is two, and taking only the
                    # first would leave `next` untyped in the file that defines
                    # it.
                    for child in n.children:
                        if child.type not in _SPINE:
                            continue
                        var = _declarator_name(child, self.src)
                        if var:
                            self.p.var_types[f"{scope}::{var}"] = found
            stack.extend(n.children)

    def calls_in(self, body, caller: str) -> None:
        """Every call inside a body, with what it was called on.

        C has two shapes and they mean different things. `foo(x)` names a
        function directly. `conn->handler->done(conn)` calls through a function
        pointer stored in a struct, and the name `done` is a field, not a
        symbol -- so it is recorded WITH its receiver. Recording it bare would
        make it look like a call to any global function called `done`, which is
        the one thing extraction must never do.

        When the receiver is not a plain variable -- `conn->handler` above --
        its whole text is kept as the receiver. No declared variable is ever
        spelled that way, so it cannot accidentally match one, and the refusal
        that follows says the receiver's type is unknown, which is true.
        """
        stack = [body]
        while stack:
            node = stack.pop()
            if node.type == "call_expression" and node.children:
                fn = node.children[0]
                if fn.type == "identifier":
                    self.p.calls.append(CallSite(
                        caller=caller, file=self.p.path,
                        line=node.start_point[0] + 1, name=_text(fn, self.src)))
                elif fn.type == "field_expression":
                    what = _named(fn, "field_identifier")
                    who = fn.children[0] if fn.children else None
                    if what is not None and who is not None:
                        self.p.calls.append(CallSite(
                            caller=caller, file=self.p.path,
                            line=node.start_point[0] + 1,
                            name=_text(what, self.src),
                            receiver=_text(who, self.src)))
            stack.extend(node.children)


def parse(source: str, parsed: ParsedFile) -> bool:
    if not available():                            # pragma: no cover
        return False
    src = source.encode("utf-8")
    root = _parser.parse(src).root_node
    # A C file dense with macros produces ERROR nodes and keeps going; measured
    # over both corpora, 162 of 603 files contain at least one and none is a
    # total loss. Only a file that yielded no tree at all is a failure.
    if root.has_error and not root.children:
        return False
    reader = _Reader(parsed, src)
    own_dir = [p for p in parsed.path.replace("\\", "/").split("/")[:-1] if p]

    for node in _declarations(root):
        if node.type == "preproc_include":
            _include(node, src, parsed, own_dir)
        elif node.type == "function_definition":
            _function(node, src, reader)
        elif node.type == "function_declarator":
            # Only ever reached from `_salvage`; a well-formed tree puts a
            # declarator inside a definition, never at this level.
            _wrecked_function(node, src, reader)
        elif node.type == "type_definition":
            _typedef(node, src, reader, parsed)
        elif node.type in _TAGGED:
            _tagged(node, src, reader, parsed, doc_node=node)
        elif node.type == "declaration":
            # `struct AnonHolder { int a; } instance;` defines a type on its way
            # to declaring a variable. A prototype is also a `declaration` and
            # is deliberately skipped -- see the module note.
            for child in node.children:
                if child.type in _TAGGED:
                    _tagged(child, src, reader, parsed, doc_node=node)
    return True


def _include(node, src: bytes, parsed: ParsedFile, own_dir: list[str]) -> None:
    quoted = _named(node, "string_literal")
    system = _named(node, "system_lib_string")
    if quoted is not None:
        frag = _named(quoted, "string_content")
        raw = _text(frag, src) if frag is not None else _text(quoted, src).strip('"')
        dotted = _include_path(raw, own_dir)
    elif system is not None:
        # `<stdio.h>` is not in the corpus and is not meant to be. Emitting it
        # unresolved is the answer to "what does this file depend on".
        #
        # Not qualified by the including file's directory, because `<>` means
        # "search the include path, never here". Known cost, measured over both
        # corpora: 9 of 1,142 system includes end in a name a corpus file also
        # has -- `<openssl/rand.h>` beside curl's own `lib/rand.h`,
        # `<sys/select.h>` beside `lib/select.h` -- and resolution, which tries
        # the tail of a dotted path, links them to the wrong file.
        raw = _text(system, src).strip("<>")
        dotted = _include_path(raw, [])
    else:
        return
    if dotted:
        parsed.import_sites.append((dotted, node.start_point[0] + 1))
    # `parsed.imports` maps a LOCAL NAME to a module, and C has no such binding:
    # an include brings in every name the header declares, and which ones those
    # are is not knowable while parsing this file alone. Leaving it empty sends
    # every bare call to the corpus-wide unique-name rule, which is the honest
    # amount of evidence a C include actually provides.


def _function(node, src: bytes, reader: _Reader) -> None:
    spine = _spine(node)
    name = _declarator_name(spine, src) if spine is not None else None
    if not name:
        return
    nid = reader.emit(name, "function", node)
    body = _named(node, "compound_statement")
    reader.local_types(node, name)
    if body is not None:
        reader.calls_in(body, nid)


def _wrecked_function(node, src: bytes, reader: _Reader) -> None:
    """A function whose header the grammar could not assemble into a definition.

    `vtls/apple.c:81` arrives as a `CURLcode`, a `function_declarator` and a
    `{`, lying loose in a recovery node because a `#ifdef` further down opens a
    brace in one arm and closes it in another. It is the only function that
    file defines, and without this rule the file has no function nodes at all.

    The body is deliberately NOT read, and the cost is one-sided and real:
    calls made from these functions have no caller and are not recorded. The
    wreckage does not say where the body ends -- in `cf-haproxy.c` the run of
    statements after the brace contains five more function definitions -- so
    guessing a boundary would put other functions' calls on this one, which is
    worse than recording none.
    """
    name = _declarator_name(node, src)
    if name:
        reader.emit(name, "function", node)


def _typedef_names(node):
    """The declarators of a typedef, read from the semicolon backwards.

    A typedef is `typedef <type> <name>[, <name>]* ;` and the hard part is
    telling the type from the name, because both can be a bare
    `type_identifier`. Reading forwards gets it wrong: in `typedef CURLcode
    Curl_cft_connect(struct Curl_cfilter *cf, ...)` -- curl's whole filter
    interface, `cfilters.h:49` -- the first type_identifier is the RETURN type,
    and taking it named eleven different function types `CURLcode`.

    Backwards there is no ambiguity. The last declarator sits just before the
    `;`, and another one only follows a comma. `typedef Foo Bar;` therefore
    stops after `Bar`, while `typedef int a, b;` collects both. Eight of
    `cfilters.h`'s eleven callback types are declared this way.
    """
    out = []
    expect_comma = False
    for child in reversed(node.children):
        if child.type == ";":
            continue
        if child.type == ",":
            expect_comma = False
            continue
        if expect_comma:
            break                                  # reached the type
        if child.type in _DECLARATORS:
            out.append(child)
            expect_comma = True
        else:
            break
    return out


def _typedef(node, src: bytes, reader: _Reader, parsed: ParsedFile) -> None:
    """`typedef struct listNode { ... } listNode;` -- one type, up to two names.

    The typedef name wins over the struct tag, because that is what the rest of
    the code writes. Emitting the tag as well would put two class nodes in the
    map for one type; the cost of not doing it is that `struct _clusterNode` is
    findable only as `clusterNode`, which is 30 types on redis and 20 on curl.
    """
    for declarator in _typedef_names(node):
        label = _declarator_name(declarator, src)
        if not label:
            continue
        reader.emit(label, "class", node)
        parsed.defined_classes.add(label)


def _tagged(spec, src: bytes, reader: _Reader, parsed: ParsedFile, doc_node) -> None:
    """A struct, union or enum written without a typedef.

    Only one with a body counts. `struct connectdata;` and `struct list *p;`
    both contain a struct_specifier and neither defines anything -- the first is
    an opaque forward declaration, the second is a use. Emitting those would
    fill the map with empty types whose real definition is in another file.
    """
    if _named(spec, "field_declaration_list", "enumerator_list") is None:
        return
    tag = _named(spec, "type_identifier")
    if tag is None:
        return                                     # `enum { X = 1 };` has no name
    label = _text(tag, src)
    reader.emit(label, "class", spec, doc_node=doc_node)
    parsed.defined_classes.add(label)
