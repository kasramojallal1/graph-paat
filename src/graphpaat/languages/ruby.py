"""Ruby, read with tree-sitter.

Same shapes as everywhere else: a class and a module both become `class` nodes,
a `def` inside either becomes a `method`, a run of `#` lines above a definition
becomes a claim. Nothing downstream learns Ruby exists.

Four things about Ruby need a decision, and all four are made here rather than
downstream.

**Nesting is flat in the id.** Ruby wraps almost everything in a module --
`module Rack; class Request` -- and it reopens the same class from several
files. If the module went into the id, `Request` would be minted as
`<file>_rack_request` while a call site looking for it asks for
`<file>_request`, and nothing would resolve. So a class or module is minted
under its own name alone, and the enclosing module is preserved as a
`contains` edge instead of as part of the name. The file prefix is what
separates two `Request`s, which is what actually distinguishes them.
The cost, stated: `module A; class X` and `module B; class X` in ONE file mint
one id, and the collision report says so.

**`include` is inheritance.** `include Comparable` puts the module into the
class's ancestor chain and its instance methods become the class's methods --
the same thing Go's embedding and TypeScript's `implements` mean, both of which
are already drawn as `inherits`. `prepend` and `extend` do it at the other end
of the chain and on the singleton respectively; the graph does not distinguish
them, exactly as it does not distinguish `extends` from `implements`.

**A require is an ordinary method call.** `require 'rack/utils'` parses as a
`call`, not as a statement type of its own, and it can sit at any depth -- half
of Jekyll's requires are inside `module Jekyll`. So requires are collected on
the walk rather than from the top-level statement list.

**A call can drop its parentheses, and then it looks like a variable.**
`listen` with no receiver and no arguments is a plain `identifier`, the same
node a local variable read produces -- and that is how most calls between the
methods of one class are written. Reading them by Ruby's own rule (a bound name
is the variable, otherwise it is a call on self) is what takes self-calls from
16 to 114 on Rack and 24 to 520 on Jekyll. See `bare_identifier`.
"""
from __future__ import annotations

import posixpath
import re

from ..ids import mint
from ..parse import DOC_SUFFIX, CallSite, Edge, Node, ParsedFile

# `.rake` is Ruby with a different name; rakefiles define real modules and
# methods and leaving them out loses the build surface of a repository.
EXTENSIONS = {".rb", ".rake"}
WHY = ""

_parser = None

# Calls that are consumed as structure rather than drawn as calls. Emitting
# `include Helpers` as a call to a method named `include` would add a refusal
# for every mixin in the corpus and teach an agent nothing -- the fact is
# already recorded, as an `inherits` edge.
_STRUCTURAL = {"require", "require_relative", "include", "extend",
               "prepend", "attr_accessor", "attr_reader", "attr_writer"}

# `require` and `require_relative` only. `load` is Ruby's third loader, but it
# is also an ordinary method name -- Jekyll's own `Cache#load` (cache.rb:163) --
# and neither corpus contains a single receiverless `load "path"`. Claiming the
# name would swallow real calls to buy nothing.
_REQUIRE = ("require", "require_relative")

# Mixin forms. All three put another module's methods on this one.
_MIXINS = {"include", "extend", "prepend"}

# The two halves `attr_*` generates. `attr_accessor :env` defines both `env`
# and `env=`; see `attributes`.
_ATTR_READERS = {"attr_accessor", "attr_reader"}
_ATTR_WRITERS = {"attr_accessor", "attr_writer"}
_ATTR = _ATTR_READERS | _ATTR_WRITERS

# A `#` line that is an instruction to a tool, not documentation. Ruby has no
# `///` to separate the two -- `#` carries the magic comments, the linter
# directives and the actual prose -- so the noise has to be named. Left in,
# `# rubocop:disable Metrics/AbcSize` becomes the stated reason for a method,
# which is worse than that method having no reason at all.
# The colon is required, not optional. Every magic comment and linter pragma
# carries one -- `# rubocop:disable ...`, `# encoding: utf-8` -- while a
# sentence does not, and an optional colon here silently deleted the
# documentation of anything whose first word happened to be "Encoding".
_DIRECTIVE = re.compile(
    r"^(?:!"                                    # shebang
    r"|-\*-"                                    # -*- coding: utf-8 -*-
    r"|(?:frozen_string_literal|encoding|coding|warn_indent"
    r"|shareable_constant_value|typed|rubocop|sorbet)\s*:"
    r"|:(?:nodoc|stopdoc|startdoc|enddoc|notnew|nocov|toc|main|title):?)",
    re.IGNORECASE)


def available() -> bool:
    """True when the grammar is installed. The registry skips us otherwise, so
    a Python-only user never has to carry a Ruby grammar."""
    global _parser, WHY
    if _parser is not None:
        return True
    try:
        import tree_sitter_ruby
        from tree_sitter import Language, Parser
        _parser = Parser(Language(tree_sitter_ruby.language()))
        return True
    except Exception as exc:                      # pragma: no cover
        WHY = f"needs tree-sitter and tree-sitter-ruby ({exc})"
        return False


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _field(node, name: str):
    return node.child_by_field_name(name)


def _named(node, *types):
    for child in node.children:
        if child.type in types:
            return child
    return None


def _body(node):
    """The `body_statement` of a class, module, method or singleton class."""
    return _field(node, "body")


def _statements(node) -> list:
    body = _body(node)
    return list(body.children) if body is not None else []


def _const_name(node, src: bytes) -> str | None:
    """The bare name a constant reference ends in.

    `Rack::Response::Raw` is written three ways in one corpus -- fully
    qualified, half qualified, and bare inside its own module -- and all three
    mean the same class. Keying on the last segment is what lets a subclass in
    one file find a base class in another. The cost is that `A::Foo` and
    `B::Foo` become one name; the resolver already refuses when a name has two
    homes, so an ambiguous case produces an honest gap rather than a wrong edge.
    """
    if node is None:
        return None
    if node.type == "constant":
        return _text(node, src)
    if node.type == "scope_resolution":
        return _const_name(_field(node, "name"), src)
    return None


def _own_line(comment) -> bool:
    """True when nothing but whitespace precedes the comment on its line.

    `X = 1  # trailing note` puts the comment on the same row as the statement
    before it and one row above the next definition, which passed the
    "immediately above" test and made every trailing note the documentation of
    whatever followed it. Ruby uses trailing notes heavily, so this is not a
    corner case.
    """
    prev = comment.prev_sibling
    return prev is None or prev.end_point[0] < comment.start_point[0]


def _prev_statement(node):
    """The node before this one, climbing out of a body when it is the first.

    tree-sitter-ruby attaches the comments that open a `module` or `class` body
    to the module node itself, ahead of the `body_statement`. So the doc comment
    of the first definition inside a module is not that definition's previous
    sibling, and looking only at siblings loses the documentation of the first
    class in every file that wraps its code in a module -- which in Ruby is
    nearly all of them.
    """
    while node is not None:
        prev = node.prev_sibling
        if prev is not None:
            return prev
        parent = node.parent
        if parent is None or parent.type != "body_statement":
            return None
        node = parent
    return None


def _clean_comment(text: str) -> list[str]:
    """One comment node's prose lines, with tool directives dropped."""
    if text.startswith("=begin"):
        # `=begin/=end` is Ruby's block comment and always documentation.
        inner = text.splitlines()[1:]
        if inner and inner[-1].startswith("=end"):
            inner = inner[:-1]
        return [line.strip() for line in inner]
    body = text.lstrip("#")
    stripped = body.strip()
    if _DIRECTIVE.match(stripped):
        return []
    return [stripped]


def _doc_above(node, src: bytes) -> str | None:
    """The run of `#` lines, or the `=begin` block, directly above a definition.

    This is RDoc, which is Ruby's docstring. Only a comment on the line
    immediately above counts: one separated by a blank line is a note about
    something else. The check has to apply to the first comment too -- guarding
    it on "we already have lines" lets any single detached comment through.
    """
    lines: list[str] = []
    prev = _prev_statement(node)
    while prev is not None and prev.type == "comment":
        if prev.end_point[0] + 1 < node.start_point[0] or not _own_line(prev):
            break
        lines[:0] = _clean_comment(_text(prev, src))
        node, prev = prev, _prev_statement(prev)
    joined = " ".join(l for l in lines if l).strip()
    return joined or None


def _camelize(segment: str) -> str:
    """`mock_request` -> `MockRequest`.

    A require names a path; the constant it defines is that path's last segment
    camelized. That is not a guess about a particular project -- it is the
    convention every Ruby autoloader implements, and it is the only thing tying
    `require 'rack/utils'` to the name `Utils` used three lines later. The
    binding is only ever used as evidence: the resolver still checks that the
    named symbol exists in that exact file before drawing an edge.
    """
    return "".join(p[:1].upper() + p[1:] for p in segment.split("_") if p)


def _own_methods(node, src: bytes) -> set[str]:
    """Every method name this class or module body defines directly.

    Used only to decide whether a bare name is a call on self, so it walks the
    body without descending into a nested class -- a nested class's methods are
    not callable by bare name from the outer one.
    """
    out: set[str] = set()
    stack = _statements(node)
    while stack:
        n = stack.pop()
        if n.type in ("class", "module"):
            continue
        if n.type in ("method", "singleton_method"):
            name = _field(n, "name")
            if name is not None:
                out.add(_text(name, src))
            continue
        if n.type == "call":
            method = _field(n, "method")
            if method is not None and _text(method, src) in _ATTR:
                args = _field(n, "arguments")
                for arg in args.children if args is not None else []:
                    if arg.type in ("simple_symbol", "symbol"):
                        out.add(_text(arg, src).lstrip(":").strip("\"'"))
                continue
        stack.extend(n.children)
    return out


# Every way Ruby binds a local name. Over-collecting here is the safe
# direction: a name wrongly believed to be a local loses one call edge, while a
# local wrongly believed to be a method invents one.
_PARAMETER_LISTS = ("method_parameters", "block_parameters", "lambda_parameters",
                    "parameters")
_BINDS_LEFT = ("assignment", "operator_assignment", "for")


def _locals_in(node, src: bytes) -> set[str]:
    """Names bound as local variables anywhere in this method."""
    out: set[str] = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type in _PARAMETER_LISTS or n.type == "exception_variable" \
                or n.type == "left_assignment_list":
            for ident in _all_identifiers(n):
                out.add(_text(ident, src))
        elif n.type in _BINDS_LEFT:
            left = _field(n, "left") or _field(n, "pattern")
            if left is not None:
                for ident in _all_identifiers(left):
                    out.add(_text(ident, src))
        stack.extend(n.children)
    return out


def _all_identifiers(node) -> list:
    stack, out = [node], []
    while stack:
        n = stack.pop()
        if n.type == "identifier":
            out.append(n)
        stack.extend(n.children)
    return out


class _Reader:
    def __init__(self, parsed: ParsedFile, src: bytes):
        self.p = parsed
        self.src = src
        # id -> fully qualified name, for the reopen check in `definition`.
        self.classes: dict[str, str] = {}
        self.by_id: dict[str, Node] = {}
        # The names `bare_identifier` needs; see `_own_methods` and `_locals_in`.
        self.own: set[str] = set()
        self.locals: set[str] = set()
        # Bare-name calls are only read inside a `def`. At class-body level
        # `self` is the class, so nothing there can resolve against a method
        # owner -- and a block parameter there (`lambda { |ip| ... }`) is not
        # covered by `locals`, so every one of them looked like a call.
        self.in_method = False

    # ---- emission ----------------------------------------------------

    def emit(self, name: str, kind: str, node, container: str,
             scope: list[str] | None = None, bases=None, doc_node=None) -> str:
        nid = mint(self.p.prefix, name, scope or [])
        line = node.start_point[0] + 1
        self.last_node = Node(id=nid, label=name, kind=kind, file=self.p.path,
                              line=line, bases=bases)
        self.p.nodes.append(self.last_node)
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

    # ---- the walk ----------------------------------------------------

    def walk(self, statements, container: str, chain: list[str],
             cls: str | None, in_class: bool, qualified: str = "") -> None:
        """Read a list of statements.

        `chain` is what a definition here is minted under, `cls` is the nearest
        enclosing class or module (which owns any method defined here),
        `in_class` says whether a `def` here is a method or a plain function,
        and `qualified` is the Ruby constant path -- `Jekyll::Converters` --
        used only to tell a reopened class from a same-named different one.
        """
        for node in statements:
            self.statement(node, container, chain, cls, in_class, qualified)

    def statement(self, node, container: str, chain: list[str],
                  cls: str | None, in_class: bool, qualified: str) -> None:
        kind = node.type
        if kind == "comment":
            return          # read by `_doc_above`, from the node it documents

        if kind in ("class", "module"):
            self.definition(node, container, qualified)
            return

        if kind == "singleton_class":
            # `class << self` is not a type of its own: it opens the singleton
            # of the class around it, so a `def` inside is one of that class's
            # methods and belongs on that class's node. Emitting a node for the
            # `class << self` itself would put every class method under a
            # nameless owner.
            self.walk(_statements(node), container, chain, cls,
                      cls is not None, qualified)
            return

        if kind in ("method", "singleton_method"):
            self.method(node, container, chain, cls, in_class, qualified)
            return

        if kind == "call":
            name_node = _field(node, "method")
            name = _text(name_node, self.src) if name_node is not None else ""
            if name in _REQUIRE:
                self.require_call(node, name)
                return
            if in_class and cls and name in _MIXINS:
                return          # recorded as `bases` by `mixins`
            if in_class and cls and name in _ATTR:
                self.attributes(node, container, cls)
                return

        # Anything else at this level -- an assignment, a constant, a call that
        # configures the class -- still belongs to the enclosing container, and
        # a `def` can be hidden inside a conditional wrapping the class body.
        self.expression(node, container, chain, cls, in_class, qualified)

    def expression(self, node, container: str, chain: list[str],
                   cls: str | None, in_class: bool, qualified: str) -> None:
        """Everything that is not itself a definition: calls, types, and any
        definition buried inside a conditional.

        Descending through `if`/`case`/`begin` matters: a class body that
        defines a method one way on one Ruby version and another way on the
        next hides both defs inside an `if`, and treating that subtree as
        opaque loses them both. Measured on Rack: `utils.rb:89` guards two
        `clock_time` definitions that way, and neither is a top-level statement.
        """
        stack = [node]
        scope_key = ".".join(chain)
        setters: set[int] = set()
        while stack:
            n = stack.pop()
            if n.type in ("class", "module"):
                self.definition(n, container, qualified)
                continue
            if n.type in ("method", "singleton_method"):
                self.method(n, container, chain, cls, in_class, qualified)
                continue
            if n.type == "singleton_class":
                self.walk(_statements(n), container, chain, cls,
                          cls is not None, qualified)
                continue
            if n.type == "call":
                self.call(n, container, scope_key, cls, setter=n.id in setters)
            elif n.type == "assignment":
                self.assignment(n, scope_key, cls)
                left = _field(n, "left")
                if left is not None and left.type == "call":
                    # `self.docs = result` calls `docs=`, not `docs`. Read as an
                    # ordinary call it pointed the edge at the reader, which is
                    # a different method that happens to share a prefix.
                    setters.add(left.id)
            elif n.type in ("alias", "undef"):
                # `alias include? has_key?` names two methods and calls
                # neither. Read as expressions they became two calls on self.
                continue
            elif n.type == "identifier" and self.in_method:
                self.bare_identifier(n, container)
            stack.extend(n.children)

    def bare_identifier(self, node, caller: str) -> None:
        """`listen` on its own line is a call to `self.listen`.

        Ruby lets a receiverless, argumentless call drop its parentheses, and
        the grammar then gives back a plain `identifier` -- the same node a
        local variable read produces. Ignoring them costs most of the calls
        between the methods of one class, because that is how Ruby is written:
        `@ip ||= super`, `return unless post?`, `params['data']`.

        Ruby's own rule decides it, so this is not a guess. A bare name is a
        local variable if the method binds one of that name, and otherwise it is
        a call on self. So the name is emitted only when the enclosing class
        really defines a method of that name AND nothing in this method binds it
        -- and every binding form is collected, because over-collecting only
        loses an edge while under-collecting invents one.
        """
        name = _text(node, self.src)
        if name in self.locals or name not in self.own:
            return
        parent = node.parent
        if parent is not None and parent.type == "call":
            # Already recorded by `call`. Compared by byte range, not by
            # identity: the bindings hand back a fresh Node object on every
            # field access, so `is` was never true and every parenthesised
            # call was counted a second time as a bare name.
            method = _field(parent, "method")
            if method is not None and method.start_byte == node.start_byte:
                return
        self.p.calls.append(CallSite(caller=caller, file=self.p.path,
                                     line=node.start_point[0] + 1,
                                     name=name, on_self=True))

    def definition(self, node, container: str, qualified: str = "") -> None:
        """A `class` or a `module`. Both become a `class` node.

        Reopening is Ruby's normal way of writing: `module Rack; end` at the top
        of a file and `module Rack` again ten lines down are one module, and
        emitting a node for each puts the same class in the map twice. So a
        definition whose id AND whose fully qualified name are already known
        here is folded into the node already emitted.

        The qualified name is what makes that safe. Two classes that are
        genuinely different -- `Kramdown::Parser::SmartyPants` and
        `Jekyll::Converters::SmartyPants`, both in one Jekyll file -- share an
        id under the flat scheme but not a qualified name, so they are still
        emitted separately and still reported as the collision they are.
        """
        name = _const_name(_field(node, "name"), self.src)
        if name is None:
            return
        full = f"{qualified}::{name}" if qualified else name
        nid = mint(self.p.prefix, name)
        bases = self.bases(node)
        if self.classes.get(nid) == full:
            existing = self.by_id[nid]
            if bases:
                # A reopening can add a mixin the first opening did not have.
                existing.bases = sorted(set((existing.bases or []) + bases))
        else:
            nid = self.emit(name, "class", node, container, bases=bases or None)
            self.classes[nid] = full
            self.by_id[nid] = self.last_node
        self.p.defined_classes.add(name)
        # Reset the chain rather than extending it: a method must be minted as
        # `mint(prefix, name, [ClassName])` or the resolver, which rebuilds
        # exactly that id, will never find it.
        outer, self.own = self.own, _own_methods(node, self.src)
        self.walk(_statements(node), nid, [name], name, True, full)
        self.own = outer

    def bases(self, node) -> list[str]:
        out: list[str] = []
        superclass = _field(node, "superclass")
        if superclass is not None:
            for child in superclass.children:
                found = _const_name(child, self.src)
                if found:
                    out.append(found)
        out.extend(self.mixins(node))
        return out

    def mixins(self, node) -> list[str]:
        """`include`, `extend` and `prepend` written directly in the body."""
        out: list[str] = []
        for stmt in _statements(node):
            if stmt.type != "call":
                continue
            name_node = _field(stmt, "method")
            if name_node is None or _text(name_node, self.src) not in _MIXINS:
                continue
            if _field(stmt, "receiver") is not None:
                continue        # `other.include X` is not this class's mixin
            args = _field(stmt, "arguments")
            for arg in args.children if args is not None else []:
                found = _const_name(arg, self.src)
                if found:
                    out.append(found)
        return out

    def attributes(self, node, container: str, cls: str) -> None:
        """`attr_reader :env` defines a real method named `env`.

        Skipping it makes the map say the class has no `env`, which is false,
        and leaves every `request.env` unresolvable. `attr_accessor` defines
        both halves and both are emitted: the setter `env=` is what
        `self.env = x` actually calls, and dropping it left 26 of Jekyll's
        assignments pointing at a gap or, worse, at the reader.

        The comment above the `attr_reader` line documents what it generates,
        so the doc is read from the call and not from the symbol -- a symbol has
        no comment above it and every generated method would arrive with no
        reason attached.
        """
        keyword_node = _field(node, "method")
        keyword = _text(keyword_node, self.src) if keyword_node is not None else ""
        args = _field(node, "arguments")
        for arg in args.children if args is not None else []:
            if arg.type not in ("simple_symbol", "symbol"):
                continue
            label = _text(arg, self.src).lstrip(":").strip("\"'")
            if not label:
                continue
            names = ([label] if keyword in _ATTR_READERS else []) + \
                    ([f"{label}="] if keyword in _ATTR_WRITERS else [])
            for one in names:
                nid = self.emit(one, "method", arg, container, scope=[cls],
                                doc_node=node)
                self.p.owner_of[nid] = cls

    def method(self, node, container: str, chain: list[str],
               cls: str | None, in_class: bool, qualified: str) -> None:
        name_node = _field(node, "name")
        if name_node is None:
            return
        name = _text(name_node, self.src)
        # A `def` directly in a class or module body is a method of it; one
        # nested inside another `def` is a plain function, the same rule Python
        # uses. `def self.foo` is a class method and still belongs to the class:
        # Ruby keeps class and instance methods in separate namespaces and this
        # graph has one, so `Cache.clear` and `cache.clear` mint one id. That
        # costs 5 of Rack's 611 methods and 5 of Jekyll's 1,033, and the
        # collision report names every one of them.
        kind = "method" if in_class and cls else "function"
        nid = self.emit(name, kind, node, container, scope=list(chain))
        if kind == "method":
            self.p.owner_of[nid] = cls
        body = _body(node)
        if body is not None:
            outer, self.locals = self.locals, _locals_in(node, self.src)
            was_in, self.in_method = self.in_method, True
            self.walk(list(body.children), nid, chain + [name], cls, False,
                      qualified)
            self.in_method, self.locals = was_in, outer

    # ---- evidence for resolution -------------------------------------

    def require_call(self, node, keyword: str) -> None:
        args = _field(node, "arguments")
        if args is None:
            return
        for arg in args.children:
            if arg.type != "string":
                continue        # `require f` names a variable; nothing to record
            content = _named(arg, "string_content")
            raw = _text(content, self.src) if content is not None else ""
            if not raw:
                continue
            path = raw
            if keyword == "require_relative":
                # Relative to the requiring file's own directory, and `..` is
                # collapsed rather than stripped -- `require_relative '../foo'`
                # names a real file and dropping the dots would name the wrong
                # one.
                here = posixpath.dirname(self.p.path.replace("\\", "/"))
                path = posixpath.normpath(posixpath.join(here, raw))
                if path.startswith(".."):
                    # It climbs above the corpus root, so it names a file that
                    # is not in the graph. Emitting it anyway would strip the
                    # dots and match some unrelated file of the same name.
                    continue
            path = path.removesuffix(".rb")
            dotted = path.replace("/", ".").lstrip(".")
            if not dotted:
                continue
            self.p.import_sites.append((dotted, node.start_point[0] + 1))
            constant = _camelize(dotted.rsplit(".", 1)[-1])
            if constant:
                self.p.imports[constant] = f"{dotted}.{constant}"

    def assignment(self, node, scope_key: str, cls: str | None) -> None:
        """`parser = QueryParser.new` states a type; nothing else in Ruby does.

        Ruby annotates nothing, so a construction is the only place the source
        says what a name holds. `@x ||= Foo.new`, Ruby's memoised reader, is
        deliberately not read: it is an `operator_assignment`, and reading it
        was measured to resolve one more call across both corpora -- because a
        memoised value is reached through its reader method, not through the
        instance variable.
        """
        left, right = _field(node, "left"), _field(node, "right")
        if left is None or right is None or right.type != "call":
            return
        method = _field(right, "method")
        if method is None or _text(method, self.src) != "new":
            return
        found = _const_name(_field(right, "receiver"), self.src)
        if not found:
            return
        if left.type == "identifier":
            self.p.var_types[f"{scope_key}::{_text(left, self.src)}"] = found
        elif left.type == "instance_variable" and cls:
            # `@parser` is an attribute of the class, which is what
            # `attr_types` is for: a later `@parser.parse(...)` then resolves.
            self.p.attr_types[f"{cls}::{_text(left, self.src)}"] = found

    def call(self, node, caller: str, scope_key: str, cls: str | None,
             setter: bool = False) -> None:
        name_node = _field(node, "method")
        if name_node is None:
            return
        name = _text(name_node, self.src) + ("=" if setter else "")
        receiver = _field(node, "receiver")
        line = node.start_point[0] + 1

        if receiver is None:
            if name in _STRUCTURAL:
                return
            self.p.calls.append(CallSite(caller=caller, file=self.p.path,
                                         line=line, name=name))
            return

        if receiver.type == "self":
            self.p.calls.append(CallSite(caller=caller, file=self.p.path,
                                         line=line, name=name, on_self=True))
            return

        if receiver.type == "instance_variable":
            self.p.calls.append(CallSite(
                caller=caller, file=self.p.path, line=line, name=name,
                receiver=_text(receiver, self.src), receiver_is_self=True))
            return

        constant = _const_name(receiver, self.src)
        if constant is not None:
            if name == "new":
                # `Foo.new` is construction. Drawn as a call to the class, the
                # way a constructor is everywhere else here -- it is often the
                # only edge tying a factory to what it builds. Drawn as a call
                # to a method named `new` it would resolve to nothing, since
                # almost no class defines one.
                self.p.calls.append(CallSite(caller=caller, file=self.p.path,
                                             line=line, name=constant))
                return
            # A constant receiver states its own type: `Utils.escape` is the
            # `escape` defined on `Utils`, with no inference in between.
            self.p.var_types[f"{scope_key}::{constant}"] = constant
            self.p.calls.append(CallSite(caller=caller, file=self.p.path,
                                         line=line, name=name, receiver=constant))
            return

        if receiver.type == "identifier":
            self.p.calls.append(CallSite(caller=caller, file=self.p.path,
                                         line=line, name=name,
                                         receiver=_text(receiver, self.src)))
            return

        # A chained or literal receiver -- `a.b.c`, `"x".foo`, `foo { }.bar`.
        # Recorded with a receiver that can never be typed, so it is refused
        # rather than resolved. Dropping the receiver instead would make
        # `client.get.parse` look like a bare call to `parse` and link it to
        # whatever `parse` happens to be unique in the corpus.
        self.p.calls.append(CallSite(caller=caller, file=self.p.path, line=line,
                                     name=name, receiver="(expression)"))


def parse(source: str, parsed: ParsedFile) -> bool:
    if not available():                            # pragma: no cover
        return False
    src = source.encode("utf-8")
    root = _parser.parse(src).root_node
    if root.has_error and not root.children:
        return False
    reader = _Reader(parsed, src)
    reader.walk(list(root.children), parsed.prefix, [], None, False)
    return True
