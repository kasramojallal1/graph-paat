"""JavaScript, read with tree-sitter.

Same shapes as everywhere else: a class becomes a `class` node, a function
becomes a `function` node, a thing that belongs to a class becomes a `method`,
and a `/** */` block above a declaration becomes a claim.

JavaScript is the hardest of these languages to resolve, for one reason: **it
states almost nothing.** There are no parameter types and no variable
annotations, so the only place the source ever names a class is
`const s = new Server()`. Where Go declares a receiver and TypeScript annotates
a parameter, JavaScript leaves us with an identifier and no evidence, and the
honest answer to `x.foo()` is usually "I do not know what x is".

Three shapes carry most of the real code and none of them is a plain
declaration.

**A function is usually a const.** `const isArray = (v) => ...` is a
variable_declarator holding an arrow function. axios's `utils.js` has thirty-six
of those against seven `function` declarations, so reading only declarations
would miss five functions in six.

**Everything can be wrapped in `export`.** `export class Foo` is an
export_statement containing a class_declaration, so declarations are unwrapped
before being read or a module-style file looks empty.

**CommonJS is a second, older module system living in the same language.**
`require('./utils')` is a call, not an import statement, and `exports.foo =
function foo() {}` is an assignment, not a declaration. express's `lib/` is
written entirely that way -- every symbol in its six files is an assignment --
and reading only `import` and declarations gives it six file nodes and nothing
else. The same goes for `View.prototype.render = ...`, which is what a method
was before `class` existed.
"""
from __future__ import annotations

import posixpath

from ..ids import mint
from ..parse import DOC_SUFFIX, CallSite, Edge, Node, ParsedFile

EXTENSIONS = {".js", ".jsx", ".mjs", ".cjs"}
WHY = ""

# Extensions a module specifier may carry and a file prefix does not. `./a.js`
# and `./a` name the same module, so both have to arrive at the prefix `a`.
_MODULE_EXTENSIONS = (".js", ".jsx", ".mjs", ".cjs")

# How far above a declaration a `/** */` block may sit and still be its
# documentation, counted in lines from the end of the comment to the start of
# the declaration: 1 is touching, 2 is one blank line between.
#
# Both are real house styles and the split is close to total. Measured
# 2026-09-08 over two corpora: axios writes 107 of its 111 JSDoc blocks
# touching the declaration, express writes 140 of its 143 with exactly one
# blank line. Requiring adjacency -- the rule the `//` comment languages use --
# would have thrown away express's entire "why" lane and kept axios's.
#
# The cost, stated: a file-level `/** @module x */` header followed by a blank
# line and then the first declaration is now read as that declaration's doc.
# A `//` line note is still never documentation, at any distance.
_MAX_DOC_GAP = 2

# How much of an unnameable receiver expression to keep. Enough to read in a
# refusal report, short enough that a call spread over four lines does not
# become a four-line key.
_RECEIVER_TEXT_LIMIT = 60

_parser = None


def available() -> bool:
    """True when the grammar is installed. The registry skips us otherwise, so
    a Python-only user never has to carry a JavaScript grammar."""
    global _parser, WHY
    if _parser is not None:
        return True
    try:
        import tree_sitter_javascript
        from tree_sitter import Language, Parser
        _parser = Parser(Language(tree_sitter_javascript.language()))
        return True
    except Exception as exc:                       # pragma: no cover
        WHY = f"needs tree-sitter and tree-sitter-javascript ({exc})"
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


def _deep(node, *wanted):
    stack, out = list(node.children), []
    while stack:
        n = stack.pop()
        if n.type in wanted:
            out.append(n)
        stack.extend(n.children)
    return out


def _declarators(node):
    """The bindings of one `const`/`let`/`var` statement, and only those.

    Direct children on purpose. Searching the whole subtree also finds every
    binding inside a function assigned to the const, and those are not
    top-level functions -- emitting them would mint an unscoped id per nested
    helper and collide the moment two functions each have a local `handler`.
    """
    return [c for c in node.children if c.type == "variable_declarator"]


def _doc_above(node, src: bytes) -> str | None:
    """A `/** ... */` block above a declaration.

    JSDoc only. A `//` line note is usually about a line, not the declaration,
    and treating every one as documentation buries the real ones: express sits
    216 line comments directly above a declaration against 143 real JSDoc
    blocks, so taking both would make three notes in five the "why" an agent
    reads first.
    """
    prev = node.prev_sibling
    while prev is not None and prev.type == "comment":
        text = _text(prev, src)
        if text.startswith("/**"):
            if node.start_point[0] - prev.end_point[0] > _MAX_DOC_GAP:
                return None                # too far above to be about this
            body = text.strip("/*").replace("*/", "")
            cleaned = " ".join(l.strip().lstrip("*").strip() for l in body.splitlines())
            return " ".join(cleaned.split()) or None
        prev = prev.prev_sibling
    return None


def _unwrap(node):
    """`export class Foo` and `export default function f` hide the declaration.

    Every JavaScript declaration node type ends in `_declaration`, including
    `lexical_declaration` and `variable_declaration`, so one test covers them
    all. `export default Axios;` names an existing binding rather than
    declaring anything and stays wrapped, which means it is ignored.
    """
    if node.type == "export_statement":
        for child in node.children:
            if child.type.endswith("_declaration"):
                return child
    return node


def _method_name(node, src: bytes) -> str | None:
    """The name of a class member, with `#` stripped off a private one.

    Node ids reserve `#` for the docstring suffix, so a `#request` method would
    put that character into an ordinary id. Stripping it costs the marker that
    the method is private and risks colliding `#foo` with a sibling `foo`;
    neither corpus measured here has such a pair.
    """
    name = _named(node, "property_identifier", "private_property_identifier")
    return _text(name, src).lstrip("#") if name is not None else None


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
        """What a local variable IS, in the one case JavaScript says so.

        `const s = new Server()` names the class outright. There is no second
        source: no annotations, no declared receiver, no parameter types. So
        this is the entire evidence base for resolving `s.foo()`, and every
        other receiver is refused rather than guessed.
        """
        for decl in _deep(node, "variable_declarator"):
            name = _named(decl, "identifier")
            new = _named(decl, "new_expression")
            if name is None or new is None:
                continue
            # `new ns.Server()` names a class we cannot place, so it is skipped
            # rather than recorded as a `Server` we might match to the wrong file.
            ctor = _named(new, "identifier")
            if ctor is not None:
                self.p.var_types[f"{scope}::{_text(name, self.src)}"] = _text(ctor, self.src)

    def self_types_in(self, node, owner: str) -> None:
        """`this.helper = new Helper()` -- an attribute whose class is stated.

        Keyed by class rather than by method, because an attribute set in the
        constructor is used from every other method of the class.
        """
        for assign in _deep(node, "assignment_expression"):
            lhs, rhs = assign.children[0], assign.children[-1]
            if rhs.type != "new_expression" or lhs.type != "member_expression":
                continue
            if _named(lhs, "this") is None:
                continue
            attr = _named(lhs, "property_identifier")
            ctor = _named(rhs, "identifier")
            if attr is not None and ctor is not None:
                self.p.attr_types[f"{owner}::{_text(attr, self.src)}"] = _text(ctor, self.src)

    def calls_in(self, body, caller: str) -> None:
        """Every call inside a body, with what it was called on."""
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
                    self._member_call(node, fn, caller)
            elif node.type == "new_expression":
                # `new Server()` is a call to the class, and is often the only
                # edge tying a factory to what it builds.
                ident = _named(node, "identifier")
                if ident is not None:
                    self.p.calls.append(CallSite(
                        caller=caller, file=self.p.path,
                        line=node.start_point[0] + 1, name=_text(ident, self.src)))
            stack.extend(node.children)

    def _member_call(self, node, fn, caller: str) -> None:
        """`x.foo()` -- who `x` is, as far as the source actually says.

        The last branch is the one that matters. A call written `<something>.foo()`
        is a call ON something, whatever that something is, and it must never
        fall through to being treated as a plain `foo()`. When it did, the
        resolver rule that links a bare name unique in the corpus turned
        `/^([a-z][a-z\\d+\\-.]*:)?\\/\\//i.test(url)` into an edge to an
        unrelated helper that happened to be called `test`.

        Measured 2026-09-08 on axios: 21 of the 27 links that rule made came
        from a call on an expression rather than on a name, and nine of those
        were regex `.test(...)` calls landing on the same wrong helper.

        So the receiver is recorded as the source text of whatever it was
        called on. That text can never match a variable name, which is exactly
        the point -- the call is refused with "receiver type unknown" instead
        of resolved to something plausible-looking and false.
        """
        what = _named(fn, "property_identifier")
        if what is None:                   # `a[b]()` names nothing we can use
            return
        obj = fn.children[0]
        receiver, is_self_attr, on_self = None, False, False
        if obj.type == "this":
            on_self = True
        elif obj.type == "identifier":
            receiver = _text(obj, self.src)
        elif obj.type == "member_expression" and _named(obj, "this") is not None:
            attr = _named(obj, "property_identifier")
            if attr is not None:
                receiver, is_self_attr = _text(attr, self.src), True
        else:
            receiver = " ".join(_text(obj, self.src).split())[:_RECEIVER_TEXT_LIMIT]
        self.p.calls.append(CallSite(
            caller=caller, file=self.p.path, line=node.start_point[0] + 1,
            name=_text(what, self.src), receiver=receiver,
            receiver_is_self=is_self_attr, on_self=on_self))


def parse(source: str, parsed: ParsedFile) -> bool:
    if not available():                            # pragma: no cover
        return False
    src = source.encode("utf-8")
    root = _parser.parse(src).root_node
    if root.has_error and not root.children:
        return False
    reader = _Reader(parsed, src)

    # Which top-level names are classes and which are callable, collected
    # before anything is emitted. `View.prototype.render = ...` usually sits
    # below `function View()`, but `module.exports = View` sits above it, and a
    # single forward pass would judge the same file differently depending on
    # which order its author chose.
    classes, callables = _top_level_names(root, src)

    # `require` is a call, so it can appear anywhere -- inside an `if`, inside a
    # function, halfway down a file. Collected over the whole tree rather than
    # the top level, or a lazily-required module looks like no dependency at all.
    _requires(root, src, parsed)

    for outer in root.children:
        node = _unwrap(outer)
        if node.type == "import_statement":
            _imports(node, src, parsed)
        elif node.type == "export_statement" and _named(node, "from") is not None:
            # `export * from './x'` and `export { y } from './x'` are imports
            # wearing an export's clothes: the file genuinely depends on x.
            # A barrel module is nothing but these, and reading only
            # `import` would make it look like it depends on nothing.
            #
            # The `from` is what makes it one. `export default 'a string'` also
            # has a string child, and taking that as a specifier invented a
            # dependency on the value of the string.
            spec = _specifier(node, src)
            if spec is not None:
                parsed.import_sites.append((_module_path(spec, parsed.path),
                                            node.start_point[0] + 1))
        elif node.type == "class_declaration":
            _class(node, outer, src, reader, parsed)
        elif node.type in ("function_declaration", "generator_function_declaration"):
            name = _named(node, "identifier")
            if name is not None:
                label = _text(name, src)
                nid = reader.emit(label, "function", node, doc_node=outer)
                reader.types_in(node, label)
                body = _named(node, "statement_block")
                if body is not None:
                    reader.calls_in(body, nid)
        elif node.type in ("lexical_declaration", "variable_declaration"):
            _bindings(node, outer, src, reader, parsed)
        elif node.type == "expression_statement":
            _assignment(node, src, reader, parsed, classes, callables)
    return True


def _top_level_names(root, src: bytes) -> tuple[set[str], set[str]]:
    """Top-level bindings, split into the ones that are classes and the rest.

    Only these two facts are needed later: whether `X` in `X.foo = ...` is a
    class (making `foo` a static method of it) and whether `X` in
    `X.prototype.foo = ...` exists at all (making `foo` a method of it). An
    unknown `X` means `foo` is emitted at file level instead, which keeps the
    containment edge pointing at a node that exists.
    """
    classes: set[str] = set()
    callables: set[str] = set()
    for outer in root.children:
        node = _unwrap(outer)
        name = _named(node, "identifier")
        if node.type == "class_declaration":
            if name is not None:
                classes.add(_text(name, src))
        elif node.type in ("function_declaration", "generator_function_declaration"):
            if name is not None:
                callables.add(_text(name, src))
        elif node.type in ("lexical_declaration", "variable_declaration"):
            for decl in _declarators(node):
                bound = _named(decl, "identifier")
                if bound is None:
                    continue
                if _named(decl, "class") is not None:
                    classes.add(_text(bound, src))
                elif _named(decl, "arrow_function", "function_expression",
                           "generator_function") is not None:
                    callables.add(_text(bound, src))
    return classes, callables


def _heritage(node, src: bytes) -> list[str]:
    """`class Axios extends Base` -- JavaScript's only form of inheritance.

    There is no `implements` here, so unlike TypeScript there is nothing to
    fold together: a single `extends` clause naming a single expression.

    A namespaced base -- `extends React.Component` -- is recorded by its whole
    dotted text rather than its last segment. It will not resolve, which is
    right, and the unresolved edge then says what the class actually extends;
    recording `Component` alone would instead match any corpus class that
    happens to share the name.

    `extends someMixin(Base)` names no type at all and is left out, because
    there is nothing honest to write down.
    """
    heritage = _named(node, "class_heritage")
    if heritage is None:
        return []
    base = _named(heritage, "identifier", "member_expression")
    if base is None:
        return []
    if base.type == "member_expression" and _deep(base, "call_expression",
                                                  "subscript_expression"):
        return []
    return [" ".join(_text(base, src).split())]


def _class(node, outer, src: bytes, reader: _Reader, parsed: ParsedFile,
           name: str | None = None) -> None:
    name_node = _named(node, "identifier")
    label = name if name is not None else (
        _text(name_node, src) if name_node is not None else None)
    if label is None:
        return
    reader.emit(label, "class", node, doc_node=outer,
                bases=_heritage(node, src) or None)
    parsed.defined_classes.add(label)
    body = _named(node, "class_body")
    if body is None:
        return
    # `get uri()` and `set uri(v)` are two members with one name, and minting
    # both makes an id that two symbols claim. The first one wins and the
    # collision is avoided rather than reported, because the pair describes one
    # property and an agent asking about `uri` wants the line either way.
    seen: set[str] = set()
    for member in body.children:
        if member.type == "method_definition":
            fn = member
        elif member.type == "field_definition" and _named(
                member, "arrow_function", "function_expression") is not None:
            # `handleClick = () => {}` is a method written as a class field.
            # Common enough in React-era code that skipping it loses the whole
            # behaviour of a component.
            fn = member
        else:
            continue
        member_name = _method_name(member, src)
        if member_name is None or member_name in seen:
            continue
        seen.add(member_name)
        nid = reader.emit(member_name, "method", member, scope=[label])
        parsed.owner_of[nid] = label
        reader.types_in(fn, f"{label}.{member_name}")
        reader.self_types_in(fn, label)
        block = _named(fn, "statement_block") or _named(
            fn, "arrow_function", "function_expression")
        if block is not None:
            reader.calls_in(block, nid)


def _bindings(node, outer, src: bytes, reader: _Reader, parsed: ParsedFile) -> None:
    """`const make = (a) => ...` and `const Foo = class {}`.

    Most functions in modern JavaScript arrive this way, and a class expression
    bound to a const is a class by any reading -- axios declares `CancelToken`
    exactly like that.
    """
    for decl in _declarators(node):
        name = _named(decl, "identifier")
        if name is None:
            continue
        label = _text(name, src)
        cls = _named(decl, "class")
        if cls is not None:
            # The binding's name wins over the class expression's own name:
            # `const Named = class Inner {}` is imported and called `Named`
            # everywhere, and `Inner` is visible only inside its own body.
            _class(cls, outer, src, reader, parsed, name=label)
            continue
        fn = _named(decl, "arrow_function", "function_expression", "generator_function")
        if fn is None:
            continue
        nid = reader.emit(label, "function", decl, doc_node=outer)
        reader.types_in(fn, label)
        body = _named(fn, "statement_block") or fn
        reader.calls_in(body, nid)


def _assignment(stmt, src: bytes, reader: _Reader, parsed: ParsedFile,
                classes: set[str], callables: set[str]) -> None:
    """The three ways JavaScript defines something without declaring it.

        exports.normalizeType = function (type) {...}   a CommonJS export
        module.exports = function View(name) {...}      the whole module
        View.prototype.render = function () {...}       a method, pre-`class`

    Without these, express's `lib/` yields six file nodes and nothing else --
    every symbol it has is an assignment. The right-hand side must actually be
    a function or a class; `AxiosError.ERR_BAD_REQUEST = 'ERR_BAD_REQUEST'` is
    a constant and gets no node.
    """
    expr = _named(stmt, "assignment_expression")
    if expr is None:
        return
    lhs, rhs = expr.children[0], expr.children[-1]
    if rhs.type not in ("function_expression", "arrow_function",
                        "generator_function", "class"):
        return
    if lhs.type != "member_expression":
        return
    prop = _named(lhs, "property_identifier")
    if prop is None:
        return
    name = _text(prop, src)
    obj = lhs.children[0]
    obj_text = _text(obj, src)
    kind = "class" if rhs.type == "class" else "function"

    # `module.exports = function View() {}` -- the module IS the function, so
    # its name has to come from the function expression. An anonymous one has
    # no name to give and gets no node.
    if obj_text == "module" and name == "exports":
        inner = _named(rhs, "identifier")
        if inner is None:
            return
        _emit_assigned(stmt, rhs, _text(inner, src), kind, src, reader, parsed)
        return

    owner = None
    if obj_text.endswith(".prototype"):
        base = obj_text[:-len(".prototype")]
        # Only when the constructor is declared in this file. Otherwise the
        # containment edge would point at an id nobody owns.
        owner = base if base in classes or base in callables else None
    elif obj.type == "identifier" and obj_text in classes:
        owner = obj_text            # `AxiosError.from = ...` is a static method
    elif obj.type != "identifier" and obj_text != "module.exports":
        # `a.b.c = function () {}` -- too deep to say what it belongs to.
        return

    _emit_assigned(stmt, rhs, name, kind, src, reader, parsed, owner)


def _emit_assigned(stmt, rhs, label: str, kind: str, src: bytes, reader: _Reader,
                   parsed: ParsedFile, owner: str | None = None) -> None:
    if rhs.type == "class":
        _class(rhs, stmt, src, reader, parsed, name=label)
        return
    scope = [owner] if owner else None
    nid = reader.emit(label, "method" if owner else kind, stmt,
                      doc_node=stmt, scope=scope)
    if owner:
        parsed.owner_of[nid] = owner
        reader.self_types_in(rhs, owner)
    reader.types_in(rhs, f"{owner}.{label}" if owner else label)
    body = _named(rhs, "statement_block") or rhs
    reader.calls_in(body, nid)


def _module_path(spec: str, from_path: str) -> str:
    """A module specifier as a dotted name the resolver can match to a prefix.

        core/Axios.js  +  '../helpers/buildURL.js'  ->  helpers.buildURL
        core/Axios.js  +  './InterceptorManager.js' ->  core.InterceptorManager

    A relative specifier is made absolute against the importing file's own
    directory first. Stripping the leading `./` and stopping there -- which is
    all a flat rewrite can do -- turns `../helpers/x.js` into `helpers.x.js`
    from every directory alike, so a file two levels down resolves to nothing.

    Anything that is not a relative path inside the corpus is returned exactly
    as written: a package name (`node:http`, `mime-types`) and a path that
    climbs out of the root both keep their slashes, and a prefix never contains
    a slash, so neither can match a corpus file by accident. They are still
    recorded -- which third-party modules a file pulls in is worth knowing.
    """
    if not spec.startswith("."):
        return spec
    here = posixpath.dirname(from_path.replace("\\", "/"))
    target = posixpath.normpath(posixpath.join(here, spec))
    if not target or target.startswith((".", "/")):
        return spec
    for ext in _MODULE_EXTENSIONS:
        if target.endswith(ext):
            target = target[:-len(ext)]
            break
    return target.replace("/", ".")


def _specifier(node, src: bytes) -> str | None:
    string = _named(node, "string")
    if string is None:
        return None
    frag = _named(string, "string_fragment")
    return _text(frag, src) if frag is not None else _text(string, src).strip("'\"`")


def _imports(node, src: bytes, parsed: ParsedFile) -> None:
    spec = _specifier(node, src)
    if spec is None:
        return
    dotted = _module_path(spec, parsed.path)
    parsed.import_sites.append((dotted, node.start_point[0] + 1))
    clause = _named(node, "import_clause")
    if clause is None:
        return                             # `import './polyfill.js'` binds nothing
    for child in clause.children:
        if child.type == "identifier":     # import Default from '...'
            _bind(parsed, dotted, _text(child, src), _text(child, src))
        elif child.type == "namespace_import":
            ident = _named(child, "identifier")
            if ident is not None:
                _bind(parsed, dotted, _text(ident, src), _text(ident, src))
        elif child.type == "named_imports":
            for spec_node in child.children:
                if spec_node.type != "import_specifier":
                    continue
                names = [c for c in spec_node.children if c.type == "identifier"]
                if not names:
                    continue
                # `import { a as b }` binds b here and names a over there.
                original = _text(names[0], src)
                local = _text(names[-1], src)
                _bind(parsed, dotted, local, original)


def _bind(parsed: ParsedFile, dotted: str, local: str, original: str) -> None:
    """Record what a local name refers to, in the `module.symbol` form the
    resolver splits on."""
    parsed.imports[local] = f"{dotted}.{original}"


def _requires(root, src: bytes, parsed: ParsedFile) -> None:
    """CommonJS: `const utils = require('./utils')`.

    Three binding shapes, all of them ordinary in the same file:

        var Router = require('./router')                 the module object
        const { join } = require('path')                 destructured
        var setCharset = require('./utils').setCharset   one symbol off it
    """
    for call in _deep(root, "call_expression"):
        callee = call.children[0] if call.children else None
        if callee is None or callee.type != "identifier":
            continue
        if _text(callee, src) != "require":
            continue
        args = _named(call, "arguments")
        spec = _specifier(args, src) if args is not None else None
        if spec is None:
            continue
        dotted = _module_path(spec, parsed.path)
        parsed.import_sites.append((dotted, call.start_point[0] + 1))

        # The binding, when there is one. `require` used as a bare expression
        # -- `require('./side-effect')` -- names nothing and only counts as a
        # dependency.
        parent = call.parent
        picked = None
        if parent is not None and parent.type == "member_expression":
            picked = _named(parent, "property_identifier")
            parent = parent.parent
        if parent is None or parent.type != "variable_declarator":
            continue
        bound = _named(parent, "identifier")
        if bound is not None:
            local = _text(bound, src)
            _bind(parsed, dotted, local,
                  _text(picked, src) if picked is not None else local)
            continue
        pattern = _named(parent, "object_pattern")
        if pattern is not None:
            for part in pattern.children:
                if part.type == "shorthand_property_identifier_pattern":
                    _bind(parsed, dotted, _text(part, src), _text(part, src))
