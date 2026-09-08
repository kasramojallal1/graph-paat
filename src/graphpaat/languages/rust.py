"""Rust, read with tree-sitter.

Rust maps onto the same shapes as everything else, and one construct decides
whether it maps well: **`impl`**. A method is not written inside its type, it is
written in a separate block that names the type:

    impl Searcher { fn new() -> Searcher { .. } }
    impl fmt::Display for Searcher { fn fmt(&self, ..) { .. } }

Both make `Searcher` the owner. Reading the block's target is what keeps `new`
and `fmt` methods of a type instead of a heap of stray free functions -- the same
job a receiver does in Go, done a level up. The `for Trait` form is inheritance
as well: the type gains the trait's shape, which is what `inherits` means here.

A struct, an enum, a union, a trait and a type alias all become `class` nodes.
`&self`/`self` is the receiver, so `self.x()` resolves exactly. `///` above a
declaration is the doc comment; `//!` at the top of a file documents the file
rather than the next item, so it becomes the file's own rationale, exactly like
a Python module docstring.

**What Rust states, and what we therefore never guess.** Struct fields carry
their types, so `self.opts.check()` resolves through a declared field type --
evidence Python and Go do not hand us. Parameters and `let x: T` are declared
too. `let x = T::new()` is NOT: `new` is a convention, not a signature, and a
builder returning something else is ordinary Rust. A struct literal
`let x = T { .. }` is a declaration and is used.

**The one thing that does not resolve, stated with its cost.** A file prefix is
the path with the extension dropped, so `sync/mpsc/mod.rs` is `sync_mpsc_mod`
while the module it defines is written `sync::mpsc`. Nothing in a *use* line
says which of `sync/mpsc.rs` and `sync/mpsc/mod.rs` exists, so a module living
in a `mod.rs` cannot be matched from the file that imports it. Measured over two
repositories: of 458 in-crate use paths in one, 296 resolve and 61 would need
the `mod.rs` form; of 1,463 in the other -- which uses the older layout almost
everywhere -- 594 resolve and 881 would need it. Emitting both spellings would
buy those edges at the price of an equal number of edges claiming a module is
outside the corpus when it is not, and a false statement is worse than a gap.
"""
from __future__ import annotations

from ..ids import mint
from ..parse import DOC_SUFFIX, CallSite, Edge, Node, ParsedFile

EXTENSIONS = {".rs"}
WHY = ""

_parser = None

# Type nodes that can appear as the target of an `impl`, or as a declared type.
_TYPES = ("type_identifier", "scoped_type_identifier", "generic_type",
          "reference_type", "pointer_type", "dynamic_type", "array_type",
          "tuple_type", "bracketed_type", "primitive_type", "never_type",
          "unit_type", "qualified_type", "abstract_type", "removed_trait_bound")

# The three std wrappers that Deref to what they hold, so `Arc<Searcher>` really
# does answer `run()` with `Searcher::run`. This is the language's own rule, not
# a guess -- and it deliberately stops here. `Option<T>` and `Mutex<T>` do NOT
# deref to T (you unwrap or lock first), so unwrapping them would invent a method
# call that the compiler would reject.
_TRANSPARENT = {"Box", "Rc", "Arc"}

# A crate root file. `crate::` names the module these define, and `self`/`super`
# are counted from it rather than from a module of the file's own name.
_ROOT_FILES = ("mod", "lib", "main")

# Rust's own macros. `println!` and `assert_eq!` are the language's punctuation,
# not calls to anything a reader could go and look at, and a corpus that happens
# to define a function called `write` or `panic` should not collect a gap marker
# every time one is used.
#
# The cost of NOT having this list, measured: recording every macro invocation
# won 55 resolved edges across two repositories and drew 442 gap markers that
# pointed at nothing. Macros the corpus really does define are still recorded.
_STD_MACROS = frozenset("""
assert assert_eq assert_ne debug_assert debug_assert_eq debug_assert_ne
panic todo unimplemented unreachable compile_error
print println eprint eprintln write writeln format format_args format_args_nl
dbg vec matches concat stringify
env option_env cfg line file column module_path
include include_str include_bytes
thread_local try is_x86_feature_detected
""".split())


def available() -> bool:
    """True when the grammar is installed. The registry skips us otherwise, so
    a Python-only user never has to carry a Rust grammar."""
    global _parser, WHY
    if _parser is not None:
        return True
    try:
        import tree_sitter_rust
        from tree_sitter import Language, Parser
        _parser = Parser(Language(tree_sitter_rust.language()))
        return True
    except Exception as exc:                      # pragma: no cover
        WHY = f"needs tree-sitter and tree-sitter-rust ({exc})"
        return False


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


def _walk_type(node, wanted: str):
    stack, out = list(node.children), []
    while stack:
        n = stack.pop()
        if n.type == wanted:
            out.append(n)
        stack.extend(n.children)
    return out


def _type_name(node, src: bytes) -> str | None:
    """The bare name of a type, through references, paths and generics.

    `&mut Searcher`, `std::io::Error` and `Searcher<'a>` are all one named type.
    A generic yields its OUTER name -- `Vec<Glob>` is a Vec, not a Glob -- with
    the three transparent wrappers as the stated exception.
    """
    if node is None:
        return None
    if node.type == "type_identifier":
        return _text(node, src)
    if node.type == "scoped_type_identifier":
        ident = _named(node, "type_identifier")
        return _text(ident, src) if ident is not None else None
    if node.type == "generic_type":
        outer = _named(node, "type_identifier", "scoped_type_identifier")
        name = _type_name(outer, src)
        if name in _TRANSPARENT:
            args = _named(node, "type_arguments")
            for child in (args.children if args is not None else []):
                inner = _type_name(child, src)
                if inner:
                    return inner
        return name
    if node.type == "primitive_type":
        # `usize` is a type, but it is not a symbol anyone can look up in the
        # corpus. Returning it would only produce receivers that never resolve.
        return None
    for child in node.children:
        found = _type_name(child, src)
        if found:
            return found
    return None


def _doc_text(comment, src: bytes) -> str | None:
    """The text of a `///` or `/** */` comment, or None if it is a plain note.

    The grammar marks documentation for us: a doc comment carries an
    `outer_doc_comment_marker` child and holds its body in a `doc_comment`
    child. Reading that instead of stripping slashes ourselves is what keeps a
    `//` line note -- usually about one line of code, not about the declaration
    -- out of the map.
    """
    if _named(comment, "outer_doc_comment_marker") is None:
        return None
    body = _named(comment, "doc_comment")
    text = _text(body, src) if body is not None else _text(comment, src)
    return " ".join(text.replace("*/", " ").split()) or ""


def _last_line(node) -> int:
    """The last row that actually holds text.

    A `line_comment` swallows its own newline, so its end lands on column 0 of
    the row BELOW it. Comparing that raw number against the declaration made a
    comment with a blank line under it look adjacent, and every detached note in
    the corpus arrived as documentation of whatever followed it.
    """
    row, col = node.end_point
    return row - 1 if col == 0 and row > node.start_point[0] else row


def _doc_above(node, src: bytes) -> str | None:
    """The run of `///` lines directly above a declaration.

    Two Rust-specific rules, both of which cost real documentation if missed.

    **Attributes sit between the doc and the item.** `/// ...` then
    `#[derive(Debug)]` then `struct Glob` is the ordinary spelling, so an
    `attribute_item` is stepped over rather than treated as the end of the run.

    **A blank line ends the run**, checked against the topmost thing seen so far
    rather than against the declaration -- otherwise a comment separated from an
    attribute by a blank line would still be read as documentation of the item
    below it.
    """
    lines: list[str] = []
    top = node
    prev = node.prev_sibling
    while prev is not None:
        if _last_line(prev) + 1 < top.start_point[0]:
            break                              # blank line: not this item's doc
        if prev.type == "attribute_item":
            top, prev = prev, prev.prev_sibling
            continue
        if prev.type not in ("line_comment", "block_comment"):
            break
        found = _doc_text(prev, src)
        if found is None:
            break                              # a plain `//` note ends the run
        if found:
            lines.insert(0, found)
        top, prev = prev, prev.prev_sibling
    joined = " ".join(lines).strip()
    return joined or None


def _file_doc(root, src: bytes, parsed: ParsedFile) -> None:
    """`//!` at the top of a file documents the file, not the next declaration.

    Rust's inner doc comment is the direct equivalent of a Python module
    docstring, and it is often the only prose saying what a module is for. It
    hangs off the file node, which `parse_file` has already minted.
    """
    lines: list[str] = []
    for child in root.children:
        if child.type == "inner_attribute_item":
            continue                           # `#![allow(..)]` sits in the run
        if child.type != "line_comment" and child.type != "block_comment":
            break
        if _named(child, "inner_doc_comment_marker") is None:
            break
        body = _named(child, "doc_comment")
        text = _text(body, src) if body is not None else ""
        if text.strip():
            lines.append(" ".join(text.split()))
    doc = " ".join(lines).strip()
    if not doc:
        return
    doc_id = f"{parsed.prefix}{DOC_SUFFIX}"
    parsed.nodes.append(Node(id=doc_id, label=f"docstring of {parsed.prefix}",
                             kind="rationale", file=parsed.path, line=1,
                             text=doc[:600]))
    parsed.edges.append(Edge(source=parsed.prefix, target=doc_id,
                             relation="rationale_for", file=parsed.path, line=1))


# --------------------------------------------------------------------------
# Module paths
#
# A file prefix is the path from the corpus root with `/` turned into `_`, and
# the resolver matches a dotted string against it after dropping LEADING
# segments. So a use path has to be turned into the importing file's own
# directory language before it can match anything.
# --------------------------------------------------------------------------

def _parts(path: str) -> list[str]:
    return path.replace("\\", "/").split("/")


def _crate_root(path: str) -> list[str]:
    """The directory `crate::` starts from, guessed from the path alone.

    Cargo puts a crate's modules under `src/`, so the last `src` in the path is
    the crate root when there is one. A corpus rooted INSIDE a crate has no
    `src` left in its relative paths, and there the first directory is used.

    Being too specific is safe and being too vague is not: the resolver drops
    leading segments until something matches, so an extra directory in front
    costs nothing while a missing one loses the edge. Measured on a workspace
    laid out as `crates/<name>/src/`, this rule resolved 296 in-crate use paths
    where starting from the corpus root resolved 182.
    """
    dirs = _parts(path)[:-1]
    if "src" in dirs:
        return dirs[:len(dirs) - dirs[::-1].index("src")]
    return dirs[:1]


def _module_dir(path: str, inner: tuple[str, ...] = ()) -> list[str]:
    """The directory `self::` names, for an item at `inner` nesting of `mod`s.

    `a/b.rs` defines the module `a::b`, whose children live in `a/b/`; a
    `mod.rs` (or `lib.rs`, or `main.rs`) defines the module its own directory
    is named after. Items inside an inline `mod x { }` are one level deeper
    again, which is what `inner` carries -- without it, `use super::*` inside a
    `mod tests` block would climb one module too far.
    """
    dirs = _parts(path)
    stem = dirs[-1][:-3] if dirs[-1].endswith(".rs") else dirs[-1]
    out = dirs[:-1] if stem in _ROOT_FILES else dirs[:-1] + [stem]
    return out + list(inner)


def _rewrite(segs: list[str], path: str, inner: tuple[str, ...]) -> list[str] | None:
    """A use path in directory terms, or None when it leaves the crate.

    `crate::a::b` is absolute, `self::a` is relative to this module and
    `super::a` climbs out of it. Anything else names another crate, and a crate
    is not a directory in this corpus -- `use std::io::Read` genuinely points
    outside and is reported that way.
    """
    if not segs:
        return None
    head = segs[0]
    if head == "crate":
        return _crate_root(path) + segs[1:]
    if head == "self":
        return _module_dir(path, inner) + segs[1:]
    if head == "super":
        rest = list(segs)
        up = 0
        while rest and rest[0] == "super":
            up += 1
            rest = rest[1:]
        here = _module_dir(path, inner)
        return here[:max(0, len(here) - up)] + rest
    return None


def _use_paths(node, src: bytes, prefix: tuple[str, ...] = ()):
    """Flatten one `use` line into (segments, names an item) pairs.

    `use a::{b::C, d::*}` is one statement and two paths, and a nested brace
    list can nest again. `*` and the `self` inside a brace list name the module
    itself rather than an item in it, which the second half of the pair records
    -- the caller has to drop a trailing item name to get the module and must
    not drop a real module segment.
    """
    kind = node.type
    if kind == "use_declaration":
        for child in node.children:
            if child.type not in ("use", ";", "visibility_modifier"):
                yield from _use_paths(child, src, prefix)
    elif kind == "scoped_identifier":
        yield tuple(prefix) + tuple(_flatten_path(node, src)), True
    elif kind == "scoped_use_list":
        segs: list[str] = []
        for child in node.children:
            if child.type in ("identifier", "crate", "super", "self", "metavariable"):
                segs.append(_text(child, src))
            elif child.type == "scoped_identifier":
                segs.extend(_flatten_path(child, src))
            elif child.type == "use_list":
                yield from _use_paths(child, src, tuple(prefix) + tuple(segs))
    elif kind == "use_list":
        for child in node.children:
            if child.type in ("{", "}", ","):
                continue
            yield from _use_paths(child, src, prefix)
    elif kind == "use_as_clause":
        # `use a::B as C` binds C to a symbol named B. The local name and the
        # real name differ, and `imports` can only express a path whose last
        # segment IS the local name -- so the file-to-file edge is kept and the
        # name binding is dropped rather than recorded wrong.
        if node.children:
            for segs, _ in _use_paths(node.children[0], src, prefix):
                yield segs, False
    elif kind == "use_wildcard":
        for child in node.children:
            if child.type == "scoped_identifier":
                yield tuple(prefix) + tuple(_flatten_path(child, src)), False
            elif child.type in ("identifier", "crate", "super", "self"):
                yield tuple(prefix) + (_text(child, src),), False
    elif kind == "self":
        # `use a::{self, B}` and `use a::b::{self as c}` -- a bare `self` in a
        # brace list IS the module in front of it, not a name inside it. Reading
        # it as a name appended a literal "self" segment to the module path and
        # the import resolved to nothing.
        yield tuple(prefix), False
    elif kind in ("identifier", "crate", "super", "metavariable"):
        yield tuple(prefix) + (_text(node, src),), True


def _flatten_path(node, src: bytes) -> list[str]:
    segs: list[str] = []
    for child in node.children:
        if child.type == "::":
            continue
        if child.type == "scoped_identifier":
            segs.extend(_flatten_path(child, src))
        elif child.type in ("identifier", "crate", "super", "self",
                            "type_identifier", "metavariable"):
            segs.append(_text(child, src))
    return segs


class _Reader:
    def __init__(self, parsed: ParsedFile, src: bytes):
        self.p = parsed
        self.src = src

    def emit(self, name: str, kind: str, node, scope=None, bases=None,
             container=None) -> str:
        nid = mint(self.p.prefix, name, scope or [])
        line = node.start_point[0] + 1
        self.p.nodes.append(Node(id=nid, label=name, kind=kind, file=self.p.path,
                                 line=line, bases=bases))
        if container is None:
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

    def local_types(self, node, scope: str, generics: set[str] = frozenset()) -> None:
        """Record what every local name in a body IS, where Rust says so.

            fn f(s: &Searcher)      a declared parameter
            let s: Searcher = ..    a declared binding
            let s = Searcher { .. } a struct literal, which names its type

        `let s = Searcher::new()` is deliberately absent. `new` returning Self
        is a convention, not a signature -- a builder that returns something
        else is ordinary Rust -- and the whole point of typing a receiver is
        that it is not a guess.
        """
        stack = list(node.children)
        while stack:
            n = stack.pop()
            if n.type == "parameter":
                name = _named(n, "identifier")
                found = _type_name(_declared_type(n), self.src)
                if found in generics:
                    found = None
                if name is not None and found:
                    self.p.var_types[f"{scope}::{_text(name, self.src)}"] = found
            elif n.type == "let_declaration":
                name = _named(n, "identifier")
                if name is None:
                    stack.extend(n.children)
                    continue
                found = _type_name(_declared_type(n), self.src)
                if not found:
                    lit = _named(n, "struct_expression")
                    found = _type_name(_named(lit, *_TYPES), self.src) if lit else None
                if found in generics:
                    found = None
                if found:
                    self.p.var_types[f"{scope}::{_text(name, self.src)}"] = found
            stack.extend(n.children)

    def calls_in(self, body, caller: str, owner: str | None) -> None:
        """Every call inside a body, with what it was called on.

        `owner` is the type of the enclosing `impl`, and it is the reason `Self`
        needs no inference at all: `Self::new()` is rewritten to the real type
        name here, where it is known exactly, instead of being left for the
        resolver to guess in a file that may hold a dozen `impl` blocks.
        """
        stack = [body]
        while stack:
            node = stack.pop()
            if node.type == "call_expression":
                self._call(node, node.children[0] if node.children else None,
                           caller, owner)
            elif node.type == "macro_invocation":
                # `write!(..)` and `try_join!(..)` are calls in every sense that
                # matters to a reader: a name, defined somewhere, doing work. A
                # macro that is not in the corpus refuses like any other call.
                name = _named(node, "identifier")
                if name is not None and _text(name, self.src) not in _STD_MACROS:
                    self.p.calls.append(CallSite(
                        caller=caller, file=self.p.path,
                        line=node.start_point[0] + 1, name=_text(name, self.src)))
            stack.extend(node.children)

    def _call(self, node, fn, caller: str, owner: str | None) -> None:
        if fn is None:
            return
        line = node.start_point[0] + 1
        if fn.type == "generic_function":
            # `collect::<Vec<_>>()` -- the turbofish wraps the real callee.
            fn = fn.children[0] if fn.children else None
            if fn is None:
                return
        if fn.type == "identifier":
            self.p.calls.append(CallSite(caller=caller, file=self.p.path,
                                         line=line, name=_text(fn, self.src)))
        elif fn.type == "field_expression":
            self._method_call(fn, caller, line)
        elif fn.type == "scoped_identifier":
            self._path_call(fn, caller, line, owner)

    def _method_call(self, fn, caller: str, line: int) -> None:
        what = _named(fn, "field_identifier")
        if what is None:
            return
        value = fn.children[0] if fn.children else None
        name = _text(what, self.src)
        if value is None:
            return
        if value.type == "self":
            self.p.calls.append(CallSite(caller=caller, file=self.p.path, line=line,
                                         name=name, on_self=True))
            return
        if value.type == "field_expression" and _named(value, "self") is not None:
            # `self.opts.check()` -- the field's type is declared on the struct,
            # so this resolves without inference.
            attr = _named(value, "field_identifier")
            if attr is not None:
                self.p.calls.append(CallSite(
                    caller=caller, file=self.p.path, line=line, name=name,
                    receiver=_text(attr, self.src), receiver_is_self=True))
                return
        receiver = _text(value, self.src)
        if value.type != "identifier":
            # A chained or computed receiver -- `a.b().c()`, `v[0].c()`. It is
            # recorded verbatim, which matches no variable and so is refused.
            # Recording None instead would make it look like a bare call to `c`
            # and let it resolve to any free function of that name.
            receiver = receiver[:60]
        self.p.calls.append(CallSite(caller=caller, file=self.p.path, line=line,
                                     name=name, receiver=receiver))

    def _path_call(self, fn, caller: str, line: int, owner: str | None) -> None:
        """`Searcher::new()`, `Self::new()`, `std::mem::replace()`.

        The segment before the name decides what kind of call this is. A type
        (Rust capitalises them, and the compiler enforces it in practice) makes
        this an associated function of that type -- the same shape as calling a
        method on a variable of it. A lowercase segment is a module, so the call
        is a free function and the resolver's usual evidence applies.
        """
        segs = _flatten_path(fn, self.src)
        if len(segs) < 2:
            return
        name, qualifier = segs[-1], segs[-2]
        if qualifier == "Self":
            qualifier = owner or ""
        if name[:1].isupper():
            # `Poll::Ready(x)` and `io::Error(x)` build a value, they do not call
            # a method -- Rust reserves capitals for types and their variants.
            # The edge worth drawing is to the type being built, the same one
            # `new Server()` earns in a language with a `new` keyword. Recording
            # it as a method call instead put a gap marker on every enum variant
            # in the file: 225 of them on one corpus, all pointing at nothing.
            built = qualifier if qualifier[:1].isupper() else name
            self.p.calls.append(CallSite(caller=caller, file=self.p.path,
                                         line=line, name=built))
            return
        if qualifier[:1].isupper():
            self.p.calls.append(CallSite(caller=caller, file=self.p.path, line=line,
                                         name=name, receiver=qualifier))
            # `Searcher::new` says the name `Searcher` denotes the type
            # `Searcher`. That is the language's spelling, not an inference, and
            # without it the receiver above has no type to look up.
            self.p.var_types[f"{self.p.prefix}::{qualifier}"] = qualifier
        else:
            self.p.calls.append(CallSite(caller=caller, file=self.p.path,
                                         line=line, name=name))


def _generics(node) -> set[str]:
    """The type parameters a declaration introduces.

    `fn search<P: AsRef<Path>>(path: P)` says `path` is a `P`, and `P` is not a
    type -- it is a hole the caller fills. Recording it made 203 parameters on
    one corpus look typed, and the map then refused the call with "no such
    method on P" instead of the truth, which is that the type is not known here.
    A corpus that also defines a class of that one letter would have gone
    further and drawn an edge to it.
    """
    out: set[str] = set()
    params = _named(node, "type_parameters")
    for child in (params.children if params is not None else []):
        if child.type == "type_parameter":
            name = _named(child, "type_identifier")
            if name is not None:
                out.add(_text_of(name))
    return out


def _declared_type(node):
    """The type in `x: T` -- the child that follows the colon."""
    seen = False
    for child in node.children:
        if child.type == ":":
            seen = True
            continue
        if seen and child.type in _TYPES:
            return child
        if seen:
            return None
    return None


def _items(container, inner: tuple[str, ...] = ()):
    """Every declaration in a file, descending through inline `mod` blocks.

    A `mod x { }` block is a namespace, not a type, so what is inside it is
    emitted flat and belongs to the file. Calling it a class to get a scope
    would put the wrong word on the map: an agent asking what methods a type has
    would be handed a module.
    """
    for child in container.children:
        yield child, inner
        if child.type == "mod_item":
            body = _named(child, "declaration_list")
            if body is not None:
                name = _named(child, "identifier")
                deeper = inner + ((_text_of(name),) if name is not None else ())
                yield from _items(body, deeper)


def _text_of(node) -> str:
    return node.text.decode("utf-8", "replace") if node is not None else ""


def parse(source: str, parsed: ParsedFile) -> bool:
    if not available():                            # pragma: no cover
        return False
    src = source.encode("utf-8")
    root = _parser.parse(src).root_node
    if root.has_error and not root.children:
        return False
    reader = _Reader(parsed, src)
    _file_doc(root, src, parsed)

    items = list(_items(root))

    # First pass: what `impl` blocks say, before any node is emitted.
    #
    # `impl Display for Glob` is a fact about Glob, and Glob's node is created
    # where the struct is written -- which may be further down the file. So the
    # traits are collected first and handed to the class node as `bases`; a
    # second walk would otherwise have to re-open every node to add them.
    traits: dict[str, list[str]] = {}
    declared: set[str] = set()
    for node, _ in items:
        if node.type == "impl_item":
            trait, target = _impl_target(node, src)
            if trait and target:
                traits.setdefault(target, []).append(trait)
        elif node.type in ("struct_item", "enum_item", "union_item",
                           "trait_item", "type_item"):
            name = _named(node, "type_identifier")
            if name is not None:
                declared.add(_text(name, src))

    for node, inner in items:
        kind = node.type
        if kind == "use_declaration":
            _use(node, src, parsed, inner)
        elif kind == "mod_item":
            _mod(node, src, parsed, inner)
        elif kind in ("struct_item", "enum_item", "union_item"):
            _type_decl(node, src, reader, parsed, traits)
        elif kind == "trait_item":
            _trait(node, src, reader, parsed, traits)
        elif kind == "type_item":
            name = _named(node, "type_identifier")
            if name is not None:
                label = _text(name, src)
                reader.emit(label, "class", node,
                            bases=traits.get(label) or None)
                parsed.defined_classes.add(label)
        elif kind == "impl_item":
            _impl(node, src, reader, parsed, declared)
        elif kind == "function_item":
            _function(node, src, reader)
        elif kind == "macro_definition":
            # `macro_rules! foo` is a named definition that other code calls by
            # name. Leaving it out would make every `foo!(..)` point at nothing.
            name = _named(node, "identifier")
            if name is not None:
                reader.emit(_text(name, src), "function", node)
    return True


def _function(node, src: bytes, reader: _Reader) -> None:
    name = _named(node, "identifier")
    if name is None:
        return
    label = _text(name, src)
    nid = reader.emit(label, "function", node)
    reader.local_types(node, label, _generics(node))
    body = _named(node, "block")
    if body is not None:
        reader.calls_in(body, nid, None)


def _type_decl(node, src: bytes, reader: _Reader, parsed: ParsedFile,
               traits: dict[str, list[str]]) -> None:
    name = _named(node, "type_identifier")
    if name is None:
        return
    label = _text(name, src)
    reader.emit(label, "class", node, bases=traits.get(label) or None)
    parsed.defined_classes.add(label)
    # A struct declares the type of every field, which is the strongest evidence
    # this language hands us: `self.opts.check()` resolves through it with no
    # inference at all.
    for field in _walk_type(node, "field_declaration"):
        fname = _named(field, "field_identifier")
        ftype = _type_name(_declared_type(field), src)
        if fname is not None and ftype:
            parsed.attr_types[f"{label}::{_text(fname, src)}"] = ftype


def _trait(node, src: bytes, reader: _Reader, parsed: ParsedFile,
           traits: dict[str, list[str]]) -> None:
    name = _named(node, "type_identifier")
    if name is None:
        return
    label = _text(name, src)
    # `trait Sink: fmt::Debug` -- a supertrait is inheritance in the same sense
    # everything else here is: the trait gains the other's shape.
    bases = list(traits.get(label, []))
    bound = _named(node, "trait_bounds")
    for child in (bound.children if bound is not None else []):
        found = _type_name(child, src) if child.type in _TYPES else None
        if found:
            bases.append(found)
    reader.emit(label, "class", node, bases=bases or None)
    parsed.defined_classes.add(label)
    _members(_named(node, "declaration_list"), label, src, reader, parsed,
             container=mint(parsed.prefix, label), generics=_generics(node))


def _impl_target(node, src: bytes) -> tuple[str | None, str | None]:
    """What an `impl` block is for: (trait or None, the type).

    `impl Glob { }` has one type. `impl Display for Glob { }` has two, and the
    `for` keyword is the only thing separating them -- both sides are ordinary
    type nodes, so reading positionally is the whole trick.
    """
    kids = list(node.children)
    where = next((i for i, c in enumerate(kids) if c.type == "for"), None)
    if where is None:
        target = None
        for child in kids:
            if child.type in _TYPES:
                target = child
        return None, _type_name(target, src)
    trait = None
    for child in kids[:where]:
        if child.type in _TYPES:
            trait = child
    target = next((c for c in kids[where + 1:] if c.type in _TYPES), None)
    return _type_name(trait, src), _type_name(target, src)


def _impl(node, src: bytes, reader: _Reader, parsed: ParsedFile,
          declared: set[str]) -> None:
    _, owner = _impl_target(node, src)
    if not owner:
        return
    body = _named(node, "declaration_list")
    if body is None:
        return
    # An `impl` for a type declared in another file still qualifies its methods
    # by that type -- that is what stops two types' `new` colliding. But the
    # containment edge has to point at something that exists, so it points at
    # the file when the class node is not here to hold it.
    container = mint(parsed.prefix, owner) if owner in declared else parsed.prefix
    _members(body, owner, src, reader, parsed, container=container,
             generics=_generics(node))


def _members(body, owner: str, src: bytes, reader: _Reader, parsed: ParsedFile,
             container: str, generics: set[str] = frozenset()) -> None:
    """The functions of an `impl` or a `trait` block, as methods of the owner."""
    for member in (body.children if body is not None else []):
        if member.type not in ("function_item", "function_signature_item"):
            continue
        name = _named(member, "identifier")
        if name is None:
            continue
        label = _text(name, src)
        nid = reader.emit(label, "method", member, scope=[owner],
                          container=container)
        parsed.owner_of[nid] = owner
        scope = f"{owner}.{label}"
        reader.local_types(member, scope, generics | _generics(member))
        block = _named(member, "block")
        if block is not None:
            reader.calls_in(block, nid, owner)


def _use(node, src: bytes, parsed: ParsedFile, inner: tuple[str, ...]) -> None:
    line = node.start_point[0] + 1
    for segs, names_item in _use_paths(node, src):
        path = list(segs)
        if not path:
            continue
        inside = _rewrite(path, parsed.path, inner)
        # A path that names another crate is kept as written. `use std::io::Read`
        # really does leave this corpus, and reporting it unresolved with the
        # crate named is the useful answer, not silence.
        module = list(inside) if inside is not None else path
        item = path[-1] if names_item else None
        if item is not None and len(module) > 1:
            module = module[:-1]
        if item is not None:
            parsed.imports[item] = ".".join(module + [item])
        dotted = ".".join(module)
        if dotted and (dotted, line) not in parsed.import_sites:
            parsed.import_sites.append((dotted, line))


def _mod(node, src: bytes, parsed: ParsedFile, inner: tuple[str, ...]) -> None:
    """`mod foo;` names a sibling file; `mod foo { }` does not name a file at all."""
    if _named(node, "declaration_list") is not None:
        return
    name = _named(node, "identifier")
    if name is None:
        return
    dotted = ".".join(_module_dir(parsed.path, inner) + [_text(name, src)])
    line = node.start_point[0] + 1
    if (dotted, line) not in parsed.import_sites:
        parsed.import_sites.append((dotted, line))
