"""PHP, read with tree-sitter.

Same shapes as everywhere else. A class, an interface, a trait and an enum all
become `class` nodes; `extends`, `implements` and a `use SomeTrait;` inside a
class body all become `inherits`, because each one means the type gains the
other's shape. A `/** */` block above a declaration becomes a claim.

PHP resolves unusually well for a dynamic language, and for three reasons that
are all *stated* in the source rather than inferred:

**A file is a class.** PSR-4 puts `Slim\\Routing\\RouteResolver` in
`Routing/RouteResolver.php`, so `use Slim\\Routing\\RouteResolver;` names a file
as well as a symbol -- see `_use_clause` for how that is written out.

**Properties are typed.** `protected LoggerInterface $logger;` types every
`$this->logger->x()` in the class exactly. Python has to infer the same fact
from an assignment and is wrong whenever a factory is involved.

**A static call names its class outright.** `Utils::describeType()` says which
class it means; nothing has to be guessed and nothing has to be looked up.

What is deliberately NOT done: a receiver we cannot name -- `$rows[0]->go()`,
`$x->getBody()->read()` -- is recorded with a receiver that matches no variable
rather than with no receiver at all. Dropping the field would leave the call
looking like a plain `go()`, and the resolver would bind it to any corpus
function of that name. A refusal is the correct answer there.
"""
from __future__ import annotations

from ..ids import mint
from ..parse import DOC_SUFFIX, CallSite, Edge, Node, ParsedFile

EXTENSIONS = {".php"}
WHY = ""

_parser = None

# `self`, `static` and `parent` arrive as `named_type`, exactly like a real
# class name, so the type reader has to reject them by name. Everything else
# built into the language (`int`, `array`, `callable`, `mixed`, ...) arrives as
# `primitive_type` and never reaches this list.
_NOT_A_CLASS = frozenset({"self", "static", "parent"})

# The receiver of a call we could not name. It has to be a string no variable
# can be spelled as: the resolver matches a receiver against the file's
# recorded variable types, and this must never match one.
_UNNAMEABLE = "(expression)"


def available() -> bool:
    """True when the grammar is installed. The registry skips us otherwise, so
    a Python-only user never has to carry a PHP grammar."""
    global _parser, WHY
    if _parser is not None:
        return True
    try:
        import tree_sitter_php
        from tree_sitter import Language, Parser
        # `language_php()`, NOT `language_php_only()`. The "only" grammar is the
        # embedded-PHP dialect and rejects the `<?php` opening tag, which every
        # real file on disk starts with -- picking it fails every file.
        _parser = Parser(Language(tree_sitter_php.language_php()))
        return True
    except Exception as exc:                       # pragma: no cover
        WHY = f"needs tree-sitter and tree-sitter-php ({exc})"
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


def _field(node, name: str):
    return node.child_by_field_name(name) if node is not None else None


def _name_parts(node, src: bytes) -> list[str]:
    """`\\Slim\\Routing\\RouteResolver` -> ["Slim", "Routing", "RouteResolver"].

    The separators are their own nodes and a leading `\\` is common, so the
    parts are read off the `name` leaves rather than by splitting the text.
    """
    if node is None:
        return []
    if node.type == "name":
        return [_text(node, src)]
    out: list[str] = []
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type == "name":
            out.append((n.start_byte, _text(n, src)))
        else:
            stack.extend(n.children)
    return [t for _, t in sorted(out)]


def _sole_name(node, src: bytes) -> str | None:
    """The name a reference points at -- but ONLY when it names one segment.

    `Psr7\\Utils` is not `Utils`. One corpus measured here writes `Psr7\\Utils::`
    eighty times while defining an unrelated `Utils` class of its own, so the
    last segment aims all eighty at the wrong class. None of them produced a
    wrong edge -- the two classes happen to share no method name, and the
    resolver checks that the method exists -- but that is luck, not a rule, and
    one shared method name is all it takes. Nothing downstream knows what a
    namespace is, so a reference that names one is not resolved at all: a
    refusal an agent can see beats an edge it cannot check.

    The cost is real and small: it also gives up the handful of references that
    spell out a namespace inside the corpus. 183 calls in that corpus are
    refused for this reason.

    A leading `\\` is not a segment -- `\\Throwable` is one name, and that is the
    shape of 111 of the 124 qualified type annotations across both corpora.
    """
    parts = _name_parts(node, src)
    return parts[0] if len(parts) == 1 else None


def _spelled(node, src: bytes) -> str:
    """A reference as the source spells it, for one we are not going to resolve.

    Recorded rather than dropped so the refusal is counted: a call we chose not
    to follow is a gap, and a gap nobody counts is one nobody can close.
    """
    return "\\".join(_name_parts(node, src))


def _type_name(node, src: bytes) -> str | None:
    """The class a type annotation names, or None when it names no single class.

    A union (`Foo|Bar`) and an intersection (`Countable&Traversable`) are both
    refused: choosing one would be a guess, and a wrong receiver type produces a
    confident edge to the wrong method.
    """
    if node is None:
        return None
    if node.type == "optional_type":               # `?Foo`
        return _type_name(_named(node, "named_type", "primitive_type"), src)
    if node.type != "named_type":
        return None
    short = _sole_name(node, src)
    return None if short is None or short in _NOT_A_CLASS else short


def _doc_above(node, src: bytes) -> str | None:
    """A `/** ... */` block directly above a declaration -- PHP's docblock.

    Docblocks only. A `//` or `#` line note is a remark about the next line, not
    documentation of the declaration, and taking every one buried the real ones.

    A docblock that is nothing but `@param`/`@return`/`@throws` tags is kept.
    That is 88 of 272 doc nodes in one corpus measured here and 178 of 450 in
    the other, so the cost of keeping them is a third of the "why" layer saying
    nothing about why. They stay because in PHP a tag is often the only place a
    type or a thrown exception is written down at all.
    """
    prev = node.prev_sibling
    while prev is not None and prev.type == "comment":
        text = _text(prev, src)
        if text.startswith("/**"):
            # Only a block on the line immediately above documents this
            # declaration; one separated by a blank line is about something
            # else. `node` here is the whole declaration including any `#[...]`
            # attribute, which is what puts the attribute on the right side of
            # the gap when a docblock sits above it.
            if prev.end_point[0] + 1 < node.start_point[0]:
                return None
            body = text[3:]
            if body.endswith("*/"):
                body = body[:-2]
            lines = (l.strip().lstrip("*").strip() for l in body.splitlines())
            return " ".join(" ".join(lines).split()) or None
        prev = prev.prev_sibling
    return None


def _statements(root):
    """Top-level declarations, looking through a braced namespace.

    `namespace A\\B { class Foo {} }` is the one wrapper PHP puts real
    declarations inside; without this, a file written that way looks empty.
    """
    out = []
    for node in root.children:
        if node.type == "namespace_definition":
            body = _named(node, "compound_statement", "declaration_list")
            if body is not None:
                out.extend(body.children)
            else:
                out.append(node)
        else:
            out.append(node)
    return out


class _Reader:
    def __init__(self, parsed: ParsedFile, src: bytes):
        self.p = parsed
        self.src = src

    def emit(self, name: str, kind: str, node, at=None, scope=None, bases=None) -> str:
        nid = mint(self.p.prefix, name, scope or [])
        # `at` is the declaration's own name, and the line reported is its line
        # rather than the node's. A `#[Attribute]` is part of the declaration
        # node, so without this a decorated class is filed at the attribute's
        # line and a reader following it lands one line short of the class.
        line = (at if at is not None else node).start_point[0] + 1
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

    def types_in(self, node, scope: str) -> None:
        """What every local variable in a body IS, where PHP states it.

        Two declared forms, and both are facts rather than inferences:

            function f(Client $c)          a typed parameter
            $x = new Response();           a constructor call

        A parameter with no type, and an assignment from anything but `new`,
        are left unknown on purpose -- a call on them is refused rather than
        guessed.
        """
        stack = list(node.children)
        while stack:
            n = stack.pop()
            if n.type in ("simple_parameter", "variadic_parameter",
                          "property_promotion_parameter"):
                var = _field(n, "name")
                found = _type_name(_field(n, "type"), self.src)
                if var is not None and found:
                    self.p.var_types[f"{scope}::{_text(var, self.src).lstrip('$')}"] = found
            elif n.type == "assignment_expression":
                lhs, rhs = n.children[0], n.children[-1]
                if (lhs.type == "variable_name"
                        and rhs.type == "object_creation_expression"):
                    made = _sole_name(_named(rhs, "name", "qualified_name"), self.src)
                    if made:
                        self.p.var_types[
                            f"{scope}::{_text(lhs, self.src).lstrip('$')}"] = made
            stack.extend(n.children)

    def calls_in(self, body, caller: str, scope: str, in_method: bool) -> None:
        """Every call inside a body, with what it was called on.

        `in_method` gates `$this`: at file scope there is no `$this`, and
        treating one as a call on self would ask the resolver for the method of
        a class the caller does not belong to.
        """
        stack = [body]
        while stack:
            node = stack.pop()
            kind = node.type
            if kind == "function_call_expression":
                fn = _field(node, "function")
                # `$callback()` and `($obj->fn)()` name nothing we can look up.
                if fn is not None and fn.type in ("name", "qualified_name"):
                    called = _sole_name(fn, self.src) or _spelled(fn, self.src)
                    if called:
                        self._call(node, caller, called)
            elif kind == "member_call_expression":
                self._member_call(node, caller, in_method)
            elif kind == "scoped_call_expression":
                self._scoped_call(node, caller, scope)
            elif kind == "object_creation_expression":
                # `new Response()` is a call to the class, and is often the only
                # edge tying a factory to what it builds. `new class {...}` and
                # `new $name()` name no class and are skipped.
                cls = _named(node, "name", "qualified_name")
                made = (_sole_name(cls, self.src) or _spelled(cls, self.src)
                        ) if cls is not None else None
                if made:
                    self._call(node, caller, made)
            stack.extend(node.children)

    def _call(self, node, caller: str, name: str, **kw) -> None:
        self.p.calls.append(CallSite(caller=caller, file=self.p.path,
                                     line=node.start_point[0] + 1, name=name, **kw))

    def _member_call(self, node, caller: str, in_method: bool) -> None:
        what = _field(node, "name")
        if what is None or what.type != "name":
            return                                 # `$obj->{$dynamic}()`
        name = _text(what, self.src)
        obj = _field(node, "object")
        if obj is not None and obj.type == "variable_name":
            var = _text(obj, self.src).lstrip("$")
            if var == "this" and in_method:
                self._call(node, caller, name, on_self=True)
            else:
                self._call(node, caller, name, receiver=var)
            return
        if obj is not None and obj.type == "member_access_expression" and in_method:
            inner = _field(obj, "object")
            attr = _field(obj, "name")
            if (inner is not None and attr is not None and attr.type == "name"
                    and inner.type == "variable_name"
                    and _text(inner, self.src) == "$this"):
                # `$this->logger->info()`: the property's declared type answers
                # this exactly, which is why property types are recorded.
                self._call(node, caller, name, receiver=_text(attr, self.src),
                           receiver_is_self=True)
                return
        self._call(node, caller, name, receiver=_UNNAMEABLE)

    def _scoped_call(self, node, caller: str, scope: str) -> None:
        what = _field(node, "name")
        if what is None or what.type != "name":
            return                                 # `Foo::$handler()`
        name = _text(what, self.src)
        where = _field(node, "scope")
        if where is None:
            return
        if where.type == "relative_scope":
            word = _text(where, self.src)
            if word in ("self", "static"):
                self._call(node, caller, name, on_self=True)
            elif word == "parent":
                # Typed by the file's `extends` clause; see `_parent_of`. When
                # that could not be pinned down the binding is absent and this
                # is refused rather than aimed at a guess.
                self._call(node, caller, name, receiver="parent")
            return
        cls = _sole_name(where, self.src)
        if cls is None:
            # A namespaced scope such as `Psr7\Utils::`. Recorded as a receiver
            # nothing can be typed as, so it is refused and counted.
            self._call(node, caller, name,
                       receiver=_spelled(where, self.src) or _UNNAMEABLE)
            return
        # `Utils::describeType()` names its class outright. Binding that name to
        # itself lets the ordinary receiver-typing path prove the target, and it
        # still has to prove it: the class must be unambiguous in the corpus and
        # must really have the method.
        self.p.var_types[f"{scope}::{cls}"] = cls
        self._call(node, caller, name, receiver=cls)


def parse(source: str, parsed: ParsedFile) -> bool:
    if not available():                            # pragma: no cover
        return False
    src = source.encode("utf-8")
    root = _parser.parse(src).root_node
    if root.has_error and not root.children:
        return False
    reader = _Reader(parsed, src)
    statements = _statements(root)
    parent = _parent_of(statements, src)

    for node in statements:
        kind = node.type
        if kind == "namespace_use_declaration":
            _imports(node, src, parsed)
        elif kind in ("class_declaration", "interface_declaration",
                      "trait_declaration", "enum_declaration"):
            _type_decl(node, src, reader, parsed, parent)
        elif kind == "function_definition":
            name_node = _field(node, "name")
            if name_node is None:
                continue
            name = _text(name_node, src)
            nid = reader.emit(name, "function", node, at=name_node)
            reader.types_in(node, name)
            body = _field(node, "body")
            if body is not None:
                reader.calls_in(body, nid, name, in_method=False)
    return True


def _parent_of(statements, src: bytes) -> str | None:
    """What `parent::` means in this file, when there is one answer.

    A class states its base outright, but the resolver looks a receiver up
    across the whole file rather than one scope. So the binding is only made
    when every class in the file extends the same thing -- two classes with
    different bases would give `parent::` an answer that is right for one of
    them and confidently wrong for the other.
    """
    bases = set()
    for node in statements:
        if node.type not in ("class_declaration", "interface_declaration"):
            continue
        clause = _named(node, "base_clause")
        if clause is None:
            continue
        for child in clause.children:
            if child.type in ("name", "qualified_name"):
                found = _sole_name(child, src) or _spelled(child, src)
                if found:
                    bases.add(found)
                break
    return bases.pop() if len(bases) == 1 else None


def _bases(node, body, src: bytes) -> list[str]:
    """`extends`, `implements` and a used trait are all inheritance here.

    PHP separates the three; the graph does not, because each one means the
    type gains the other's shape, which is what a reader is asking about. A
    trait in particular is copied into the class wholesale, so leaving it out
    would hide where half the methods of a trait-heavy class come from.
    """
    out: list[str] = []
    for clause_type in ("base_clause", "class_interface_clause"):
        clause = _named(node, clause_type)
        for child in clause.children if clause is not None else []:
            if child.type in ("name", "qualified_name"):
                found = _sole_name(child, src) or _spelled(child, src)
                if found:
                    out.append(found)
    for member in body.children if body is not None else []:
        if member.type != "use_declaration":
            continue
        # Direct children only: `use A { A::x insteadof B; }` puts names inside
        # an adaptation block that renames rather than inherits.
        for child in member.children:
            if child.type in ("name", "qualified_name"):
                found = _sole_name(child, src) or _spelled(child, src)
                if found:
                    out.append(found)
    return out


def _type_decl(node, src: bytes, reader: _Reader, parsed: ParsedFile,
               parent: str | None) -> None:
    """A class, interface, trait or enum, and the methods inside it."""
    name_node = _field(node, "name")
    if name_node is None:
        return                                     # `new class {...}`
    name = _text(name_node, src)
    body = _field(node, "body")
    reader.emit(name, "class", node, at=name_node,
                bases=_bases(node, body, src) or None)
    parsed.defined_classes.add(name)
    if body is None:
        return
    has_base = _named(node, "base_clause") is not None

    for member in body.children:
        if member.type == "property_declaration":
            _property(member, src, parsed, name)
        elif member.type == "method_declaration":
            method_name_node = _field(member, "name")
            if method_name_node is None:
                continue
            label = _text(method_name_node, src)
            nid = reader.emit(label, "method", member, at=method_name_node,
                              scope=[name])
            parsed.owner_of[nid] = name
            scope = f"{name}.{label}"
            reader.types_in(member, scope)
            _promoted(member, src, parsed, name)
            if has_base and parent:
                parsed.var_types[f"{scope}::parent"] = parent
            # An interface or abstract method has no body; it is still a node,
            # because it is what a call on an interface-typed property resolves
            # to and losing it would refuse every one of those.
            block = _field(member, "body")
            if block is not None:
                reader.calls_in(block, nid, scope, in_method=True)


def _property(member, src: bytes, parsed: ParsedFile, owner: str) -> None:
    """`protected LoggerInterface $logger;` types every `$this->logger->x()`.

    This is the single biggest reason PHP resolves better than Python: the fact
    is written down in the class rather than inferred from what was assigned.
    """
    found = _type_name(_field(member, "type"), src)
    if not found:
        return
    for element in member.children:
        if element.type != "property_element":
            continue
        var = _named(element, "variable_name")
        if var is not None:
            parsed.attr_types[f"{owner}::{_text(var, src).lstrip('$')}"] = found


def _promoted(member, src: bytes, parsed: ParsedFile, owner: str) -> None:
    """`__construct(private Client $client)` declares a property too.

    Constructor promotion is how modern PHP writes most of its properties, and
    reading only `property_declaration` would leave those classes looking as if
    they had no state at all.
    """
    params = _field(member, "parameters")
    for param in params.children if params is not None else []:
        if param.type != "property_promotion_parameter":
            continue
        var = _field(param, "name")
        found = _type_name(_field(param, "type"), src)
        if var is not None and found:
            parsed.attr_types[f"{owner}::{_text(var, src).lstrip('$')}"] = found


def _imports(node, src: bytes, parsed: ParsedFile) -> None:
    group = _named(node, "namespace_use_group")
    if group is not None:
        # `use Slim\{App, Logger};` -- the prefix is a sibling of the group.
        prefix = _name_parts(_named(node, "namespace_name"), src)
        for clause in group.children:
            if clause.type == "namespace_use_clause":
                _use_clause(clause, src, parsed, prefix)
        return
    for clause in node.children:
        if clause.type == "namespace_use_clause":
            _use_clause(clause, src, parsed, [])


def _use_clause(clause, src: bytes, parsed: ParsedFile, prefix: list[str]) -> None:
    """One imported name, written so the resolver can find the file it lives in.

    A class lives in the file named after it -- `Slim\\Routing\\RouteResolver` is
    `Routing/RouteResolver.php` -- so the WHOLE path names the module, and the
    class sits inside that module. That is why the local name is appended to a
    path that already ends with it: the resolver reads the last segment as the
    symbol and the rest as the file, and dropping the repeat would point it at
    the directory instead.

    `use function GuzzleHttp\\Psr7\\hash;` is the other case. A function's file is
    not named after the function, so only the namespace can name a module; and
    `use function sprintf;` names a global function with no module at all, which
    is why it records nothing. Emitting `sprintf` as a module would both add a
    gap that can never close and risk matching a same-named file in the corpus.
    """
    parts: list[str] = []
    alias: str | None = None
    flavour = "class"
    after_as = False
    for child in clause.children:
        if child.type in ("function", "const"):
            flavour = child.type
        elif child.type == "as":
            after_as = True
        elif child.type in ("qualified_name", "namespace_name", "name"):
            if after_as:
                alias = _text(child, src)
            elif not parts:
                parts = _name_parts(child, src)
    full = prefix + parts
    if not full:
        return
    local = alias or full[-1]
    line = clause.start_point[0] + 1
    module = ".".join(full) if flavour == "class" else ".".join(full[:-1])
    if not module:
        return
    parsed.imports[local] = f"{module}.{local}"
    parsed.import_sites.append((module, line))
