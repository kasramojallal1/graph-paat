"""Kotlin, read with tree-sitter.

Kotlin maps onto the same five shapes as everything else, and four of its own
habits needed a decision that costs something. All four are stated here.

**A companion object is not a node; its members belong to the class around it.**
`companion object { fun create() }` is where Kotlin keeps what other languages
call statics, and the call site writes `Server.create()`. Folding those members
into `Server` is what makes that call resolve: the resolver looks for a method
of `Server` in the file where `Server` is defined, and that is exactly where a
companion lives. Emitting a separate `Companion` class instead would put a node
of that name in a hundred different files and resolve nothing.

**An extension function is a method of the type it extends.** `fun
Response.commonHeaders()` cannot be called without a `Response`, so it is a
`method` scoped by `Response` rather than a free function. The alternative -- a
flat `function` node -- loses symbols outright, because Kotlin overloads an
extension by changing the receiver and not the name: `Job.kt` writes six
different `cancelChildren`, `RequestBody.kt` four `toRequestBody`, and flat ids
mint one node for each set. Measured over the 9,161 `fun` declarations in the
two corpora, a flat id loses 362 of them and the receiver-scoped id loses 284.
The cost is that an extension is contained by its FILE, not by the class -- it
genuinely is not inside the class -- and it only becomes a reachable call
target when it happens to sit in the same file as the type it extends.

**A property is not a node.** In Kotlin every field is a property, so the
member list of a class is mostly private state -- the thing no language module
here emits. The cost is measured and real: 340 property declarations across the
two corpora carry a KDoc block that does not become a `rationale` node, and a
computed property with a `get()` body is a function that the map does not name.
Its declared type is still recorded, which is what resolution needs, and the
calls inside its accessors are attributed to the class.

**A source set is not a package.** okhttp keeps `okhttp3/Dns.kt` under
`commonJvmAndroid/kotlin/`, so the file prefix carries two directory segments
that the import `okhttp3.internal.concurrent.TaskRunner` never mentions and the
import resolves to nothing. The file states both halves of the fix: its
`package` line and its own path. The tail they share is the mirrored part, and
what is left over on each side is the source root and the package root -- so an
import inside the same package root is rewritten through that mapping before it
is recorded. It cannot invent a link, because the resolver still has to find a
real file at the resulting prefix. Without the rewrite, **not one import in
either corpus** names a file -- 0 of 4,399 and 0 of 2,419. With it, 509 and 15.

That second number is the honest ceiling of this language, and it is worth
saying why rather than treating it as a defect to fix. Kotlin does not require
a file to be named after the type it holds, and it lets a file export top-level
functions under the package name alone, so an import very often names something
no file is called. Of coroutines' 2,419 imports, 2,106 name a package (`import
kotlinx.coroutines.internal.*`) rather than a file at all. Of okhttp's, 953 do,
2,549 leave the corpus entirely, and 388 name a type whose file exists under a
DIFFERENT source set -- the one class of miss a smarter rewrite could reach,
and it would need to know the whole corpus at parse time, which this parser
deliberately does not.

One clash is left visible rather than smoothed over, because it is the shared
id scheme's cost and not this module's to hide. Kotlin writes a
constructor-shaped factory beside the type it builds -- `interface Job` and
`fun Job(parent)` in one file, `fun CoroutineExceptionHandler(...)` beside the
interface of that name -- and the two mint one id. That is 24 pairs on
coroutines and one on okhttp, and every one of them is reported.

What does not fit, stated plainly. **An anonymous object has no name to mint** --
`object : Interceptor { override fun intercept(chain) = ... }` declares a type
nobody can refer to, and the 535 methods inside the two corpora's anonymous
objects get no nodes. **A local function is not emitted either**, 83 of them
across both corpora. In both cases the calls are not lost: they are attributed
to the symbol the code was written inside, which is the closest true answer.
"""
from __future__ import annotations

from ..ids import mint
from ..parse import DOC_SUFFIX, CallSite, Edge, Node, ParsedFile

# `.kts` is a Kotlin script -- a Gradle build file is the common case. It parses
# with the same grammar and its top-level declarations are real symbols.
EXTENSIONS = {".kt", ".kts"}
WHY = ""

_parser = None

# Everything Kotlin calls a type. `class_declaration` already covers `class`,
# `interface`, `fun interface`, `data class`, `sealed class`, `enum class` and
# `annotation class` -- the grammar puts the distinguishing keyword in a
# `modifiers` child rather than in the node type. All of them become `class`
# nodes: a reader asking "what is a Dispatcher" does not care which keyword
# declared it, and giving each its own word would push Kotlin's vocabulary into
# grouping, ranking and query.
_TYPES = ("class_declaration", "object_declaration", "type_alias")

# The declaration list of a type. An enum keeps its members in `enum_class_body`
# alongside its entries.
_BODIES = ("class_body", "enum_class_body")

# Where a call can be written without belonging to a named symbol: a property
# initializer, a `by lazy { }` delegate, an `init { }` block, a `get()` body.
# Kotlin puts a great deal of real work in these, so their calls are attributed
# to the enclosing type rather than dropped.
_LOOSE = ("property_declaration", "anonymous_initializer")


def available() -> bool:
    """True when the grammar is installed. The registry skips us otherwise, so
    a Python-only user never has to carry a Kotlin grammar."""
    global _parser, WHY
    if _parser is not None:
        return True
    try:
        import tree_sitter_kotlin
        from tree_sitter import Language, Parser
        _parser = Parser(Language(tree_sitter_kotlin.language()))
        return True
    except Exception as exc:                      # pragma: no cover
        WHY = f"needs tree-sitter and tree-sitter-kotlin ({exc})"
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


def _type_name(node, src: bytes) -> str | None:
    """The bare class named by a type, or None if it does not name one.

    Written branch by branch rather than as a search for the first `identifier`
    below, because a search finds the wrong one twice over: `List<Server>` would
    answer `List` only by luck and `Map.Entry<A, B>` would answer `Map`. The
    outer, rightmost name is the one a call goes to, so `okhttp3.Request` is
    `Request` and `Map.Entry` is `Entry`.

    A function type -- `s: () -> Unit` -- deliberately answers None. It names no
    class, and typing a variable with one would only produce refusals carrying a
    misleading reason.
    """
    if node is None:
        return None
    if node.type == "user_type":
        # Direct `identifier` children only: the ones inside `type_arguments`
        # belong to the parameters, not to the type being named.
        last = None
        for child in node.children:
            if child.type == "identifier":
                last = child
        return _text(last, src) if last is not None else None
    if node.type in ("nullable_type", "type_projection", "parenthesized_type",
                     "definitely_non_nullable_type"):
        for child in node.children:
            found = _type_name(child, src)
            if found:
                return found
    return None


def _doc_above(node, src: bytes) -> str | None:
    """The KDoc block directly above a declaration.

    `/** ... */` only. Kotlin has a documentation form the language itself
    defines, and a `//` line above a declaration is a note about the line below
    it far more often than it is documentation -- taking those too buries the
    real ones. It also gets the licence header for free: every file in both
    corpora opens with `/*`, which is not KDoc and is not read.

    A blank line anywhere in the run breaks it, and that check has to apply to
    the first comment as well -- guarding it on "we have walked past something
    already" lets a detached note through as documentation.

    Other comments in the run are walked past rather than treated as a wall, so
    a `// TODO` written between the KDoc and the declaration does not cost the
    declaration its description.
    """
    top = node
    prev = node.prev_sibling
    text = ""
    while prev is not None and prev.type in ("line_comment", "block_comment"):
        if prev.end_point[0] + 1 < top.start_point[0]:
            return None                         # separated by a blank line
        text = _text(prev, src)
        if text.startswith("/**"):
            break
        top, prev, text = prev, prev.prev_sibling, ""
    if not text.startswith("/**"):
        return None
    body = text[3:]
    if body.endswith("*/"):
        body = body[:-2]
    cleaned = " ".join(line.strip().lstrip("*").strip() for line in body.splitlines())
    return " ".join(cleaned.split()) or None


def _receiver_type(node, src: bytes) -> str | None:
    """The `String` in `fun String.toSlug()`, or None for an ordinary function.

    The receiver is the `user_type` that sits immediately before the `.` that
    precedes the name. Reading "the first user_type child" instead would answer
    `T` for `fun <T> plain(x: T): T`, where the only user_type is the return
    type, and would call every generic function an extension.
    """
    seen = None
    for child in node.children:
        if child.type == "user_type":
            seen = child
        elif child.type == ".":
            return _type_name(seen, src)
        elif child.type in ("identifier", "function_value_parameters"):
            return None
    return None


def _bases(node, src: bytes) -> list[str]:
    """Every name in the supertype list.

    Kotlin writes three different things there and the graph draws all three as
    `inherits`: `: Base()` is a superclass, `: Handler` is an interface, and
    `: Handler by impl` is delegation. Delegation is included deliberately --
    the delegating class gains the interface's whole shape, which is what a
    reader asking "what is a Handler here" wants to see, and leaving it out
    would make the most idiomatic Kotlin composition invisible.
    """
    out: list[str] = []
    for holder in node.children:
        if holder.type != "delegation_specifiers":
            continue
        for spec in holder.children:
            if spec.type != "delegation_specifier":
                continue
            # `Base()` wraps its name in a constructor_invocation and
            # `Handler by impl` in an explicit_delegation; a bare `Handler`
            # carries the user_type itself.
            for part in (spec, _named(spec, "constructor_invocation"),
                         _named(spec, "explicit_delegation")):
                if part is None:
                    continue
                found = _type_name(_named(part, "user_type"), src)
                if found:
                    out.append(found)
                    break
    return out


def _super_class(node, src: bytes) -> str | None:
    """The superclass, told apart from the interfaces by the grammar itself.

    Kotlin only lets the SUPERCLASS be constructed in the supertype list --
    `class Cache : Closeable, Flushable` names two interfaces, `class Http2
    : Http1()` names a parent. So the one specifier holding a
    `constructor_invocation` is the parent, with no naming convention involved.

    Worth 5 resolved edges on okhttp and 22 on coroutines -- small, because 90
    of the two corpora's 111 `super.` calls go to a platform class the corpus
    does not contain. The rest of its value is in the refusal: without it those
    90 are recorded as "receiver type unknown", which is not true. We do know
    the type; the corpus simply has no such method, and a reason that is wrong
    is worse than no rule at all.
    """
    for holder in node.children:
        if holder.type != "delegation_specifiers":
            continue
        for spec in holder.children:
            invocation = _named(spec, "constructor_invocation")
            found = _type_name(_named(invocation, "user_type"), src) \
                if invocation is not None else None
            if found:
                return found
    return None


def _members(body):
    """The declarations inside a type body, in order.

    Two flattenings happen here, and both are about where Kotlin hides code.

    A `companion object` holds the type's statics and is not a node of its own,
    so its members are yielded as members of the type around it.

    An enum ENTRY is a value, not a class -- but when it carries a body, the
    methods in that body are what the enum actually does. They are yielded LAST
    so a method the enum declares in its own body wins the id and an entry's
    override only fills a gap. What is lost is which entry an override came
    from: several implementations of one name become one node pointing at the
    first.
    """
    overrides = []
    for child in body.children:
        if child.type == "companion_object":
            inner = _named(child, *_BODIES)
            if inner is not None:
                yield from inner.children
        elif child.type == "enum_entry":
            entry_body = _named(child, *_BODIES)
            for member in entry_body.children if entry_body is not None else []:
                if member.type == "function_declaration":
                    overrides.append(member)
        else:
            yield child
    yield from overrides


class _Reader:
    def __init__(self, parsed: ParsedFile, src: bytes):
        self.p = parsed
        self.src = src
        self.minted: set[str] = set()

    # ---- nodes -------------------------------------------------------

    def emit(self, name: str, kind: str, node, container: str,
             scope=None, bases=None, dedupe: bool = False) -> str | None:
        """One node, its containment edge and its KDoc. None when dropped.

        `container` is passed in rather than derived from `scope`, because the
        two are genuinely different facts here: an extension function is scoped
        by the type it extends and contained by the file it was written in, and
        a nested type is scoped flat while being contained by the type around
        it.
        """
        nid = mint(self.p.prefix, name, scope or [])
        if dedupe:
            # Only a callable can suppress another callable. The set is filled
            # ONLY here, and the reason is a real loss found by hand: Kotlin
            # writes a factory function with its type's name -- `interface Job`
            # at Job.kt:120 and `fun Job(parent)` at Job.kt:391 -- and letting
            # the type claim the id made the function vanish with no record of
            # it. The same pair in Await.kt, where the function is written
            # first, was being reported as a collision. Two shapes of the same
            # clash have to be reported the same way, and a loss that is
            # reported beats one that is silent.
            if nid in self.minted:
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

    # ---- evidence for resolution -------------------------------------

    def declare(self, scope: str, name: str, cls: str | None) -> None:
        if cls:
            self.p.var_types[f"{scope}::{name}"] = cls

    def property_type(self, declaration) -> str | None:
        """What a `val`/`var` holds, when the source says so.

        Two forms, and only the first is a declaration:

            val pool: ConnectionPool = ...     stated outright
            val pool = ConnectionPool()        stated by the constructor call

        The second reads the initializer, and it is a convention rather than a
        rule: Kotlin capitalises types and not functions, so a call to a
        capitalised bare name is a construction. It is safe because the resolver
        still has to find a class of that name AND the method being called on it
        before it will draw an edge -- a capitalised factory function produces a
        refusal, not a wrong link. Without it every inferred `val` in the corpus
        would be untyped, and inferred `val` is how Kotlin is written: dropping
        it costs 547 resolved call edges on okhttp and 306 on coroutines, more
        than a tenth of both.
        """
        declared = _named(declaration, "variable_declaration")
        found = _type_name(_named(declared, "user_type", "nullable_type"), self.src) \
            if declared is not None else None
        if found:
            return found
        initializer = _named(declaration, "call_expression")
        callee = initializer.children[0] if initializer is not None and \
            initializer.children else None
        if callee is not None and callee.type == "identifier":
            name = _text(callee, self.src)
            if name[:1].isupper():
                return name
        return None

    def properties_of(self, declaration, owner: str) -> None:
        """A class-level `val`/`var`, recorded twice on purpose.

        `attr_types` answers `this.pool.close()`, and `var_types` answers the
        bare `pool.close()` that Kotlin writes far more often -- an unqualified
        property read looks exactly like a local read at the call site, and the
        resolver has one lookup for both.
        """
        declared = _named(declaration, "variable_declaration")
        name_node = _named(declared, "identifier") if declared is not None else None
        cls = self.property_type(declaration)
        if name_node is None or not cls:
            return
        name = _text(name_node, self.src)
        self.p.attr_types[f"{owner}::{name}"] = cls
        self.declare(owner, name, cls)

    def types_in(self, node, scope: str) -> None:
        """Every declared type inside a body. Three forms, all declarations.

            fun use(srv: Server)          a parameter
            class Server(val addr: Url)   a primary-constructor property
            val srv: Server = ...         a local, or its initializer

        A destructuring `val (a, b) = pair` names no type on either side and is
        skipped rather than guessed at.
        """
        stack = list(node.children)
        while stack:
            n = stack.pop()
            if n.type in ("parameter", "class_parameter"):
                name = _named(n, "identifier")
                found = _type_name(_named(n, "user_type", "nullable_type"), self.src)
                if name is not None:
                    self.declare(scope, _text(name, self.src), found)
            elif n.type == "property_declaration":
                declared = _named(n, "variable_declaration")
                name = _named(declared, "identifier") if declared is not None else None
                if name is not None:
                    self.declare(scope, _text(name, self.src), self.property_type(n))
            stack.extend(n.children)

    # ---- calls -------------------------------------------------------

    def calls_in(self, body, caller: str, own: frozenset, parent: str | None) -> None:
        """Every call inside a body, with what it was called on.

        `own` is the set of function names the enclosing type declares in this
        file, and it is what an unqualified `listen()` is checked against. Kotlin
        is the awkward case here: an unqualified call is a member call exactly
        like Java's, EXCEPT that Kotlin also has top-level functions and imports
        them by name, and those are written the same way. Treating every
        unqualified call as a member -- Java's rule -- sends each top-level call
        to the resolver's self-method path, which refuses it and never falls
        through to the same-file and import rules. That is not a small loss in a
        language where a top-level function is the idiom.

        So the parser asks a question it can answer from this file alone: does
        the type around this call declare that name? If it does, the call is on
        self. If it does not, it is left as a plain name and the resolver's own
        rules decide. What is still lost is a call to a method the type inherits
        rather than declares, which looks like a plain name and is refused.

        The three rules were measured against each other, in resolved call
        edges:

                                     okhttp   coroutines
            declared here (shipped)    5167         4524
            always on self              4007         1818
            never on self               4174         4265

        Always-on-self is the rule Java uses and it is the worst of the three
        here, by a factor of two and a half on coroutines -- which is exactly
        where the top-level function is most used.
        """
        stack = [body]
        while stack:
            node = stack.pop()
            if node.type == "call_expression":
                self._call(node, caller, own, parent)
            stack.extend(node.children)

    def _call(self, node, caller: str, own: frozenset, parent: str | None) -> None:
        callee = node.children[0] if node.children else None
        if callee is None:                             # pragma: no cover - grammar
            return
        line = node.start_point[0] + 1

        def record(name, receiver=None, receiver_is_self=False, on_self=False):
            self.p.calls.append(CallSite(
                caller=caller, file=self.p.path, line=line, name=name,
                receiver=receiver, receiver_is_self=receiver_is_self,
                on_self=on_self))

        if callee.type == "identifier":
            # A bare name: a member of the enclosing type, a top-level function,
            # or a constructor -- `Server(addr)` is spelled exactly like a call.
            name = _text(callee, self.src)
            record(name, on_self=name in own)
            return
        if callee.type != "navigation_expression":
            # A call on the result of something -- `f()()`, `(a ?: b).run()`.
            # There is no name to record and nothing to refuse.
            return
        what = None
        for child in callee.children:
            if child.type == "identifier":
                what = child
        if what is None:
            return
        name = _text(what, self.src)
        target = callee.children[0] if callee.children else None
        if target is None or target is what:           # pragma: no cover - grammar
            return
        if target.type == "this_expression":
            record(name, on_self=True)
            return
        if target.type == "super_expression":
            # `super.foo()` is the parent's foo, and the parent is named by the
            # one supertype the class constructs. The receiver has to be
            # DECLARED as well as named -- the resolver types a receiver by
            # looking the name up as a variable, so recording `Base` without
            # also saying `Base` is a `Base` resolved exactly nothing.
            if parent:
                self.declare(parent, parent, parent)
            record(name, receiver=parent or "super")
            return
        if target.type == "identifier":
            who = _text(target, self.src)
            # `Dispatcher.INSTANCE` and `Server.create()` name the type outright,
            # which is how a companion object's members are reached. Kotlin's
            # convention -- types capitalised, values not -- is the only thing
            # separating that from a local, so this is a convention and not a
            # rule. It is safe because the resolver still has to find that class
            # AND that method on it before it draws an edge; a capitalised local
            # produces a refusal, not a wrong link. It is worth 146 resolved
            # edges on okhttp and 87 on coroutines.
            if who[:1].isupper():
                self.declare(who, who, who)
            record(name, receiver=who)
            return
        if target.type == "navigation_expression":
            inner = target.children[0] if target.children else None
            member = None
            for child in target.children:
                if child.type == "identifier":
                    member = child
            on_this = inner is not None and inner.type == "this_expression"
            if on_this and member is not None:
                record(name, receiver=_text(member, self.src),
                       receiver_is_self=True)
                return
        # A chain, an index, a cast, a literal. There is no name to type here,
        # so the receiver is recorded as the expression itself: it can never
        # match a declared variable, which means the call is counted as a
        # refusal rather than quietly disappearing from the coverage report.
        record(name, receiver=" ".join(_text(target, self.src).split())[:60])


def parse(source: str, parsed: ParsedFile) -> bool:
    if not available():                            # pragma: no cover
        return False
    src = source.encode("utf-8")
    root = _parser.parse(src).root_node
    if root.has_error and not root.children:
        return False
    reader = _Reader(parsed, src)

    package = ""
    for node in root.children:
        if node.type == "package_header":
            named = _named(node, "qualified_identifier")
            package = _text(named, src) if named is not None else ""
            _file_doc(node, src, parsed)
    to_module = _module_map(parsed.path, package)

    for node in root.children:
        if node.type == "import":
            _import(node, src, parsed, to_module)
        elif node.type in _TYPES:
            _type(node, src, reader, parsed, parsed.prefix)
        elif node.type == "function_declaration":
            _function(node, src, reader, parsed, parsed.prefix, None, frozenset(), None)
        elif node.type in _LOOSE:
            # A top-level `val client = OkHttpClient()` runs real code and the
            # file is the only honest caller for it.
            reader.types_in(node, parsed.prefix)
            reader.calls_in(node, parsed.prefix, frozenset(), None)
    return True


def _module_map(path: str, package: str):
    """How this file's imports have to be spelled to name a file prefix.

    A prefix is the path relative to the corpus root, so `okhttp3/Dns.kt` under
    `commonJvmAndroid/kotlin/` is `commonjvmandroid_kotlin_okhttp3_dns` while
    the import that reaches it is written `okhttp3.Dns`. The two disagree by the
    source-set directories, and Kotlin lets nothing else in the file tell you
    what those are.

    The file itself does, though, in two halves it always carries: its `package`
    line and its own path. Line them up from the right and the part they share
    is the mirrored directory structure; what is left over on each side is the
    source root and the package root.

        commonJvmAndroid/kotlin/okhttp3/Dns.kt   package okhttp3
            shared okhttp3 -> source root commonJvmAndroid/kotlin, package root -
        common/src/flow/Flow.kt                  package kotlinx.coroutines.flow
            shared flow     -> source root common/src, package root kotlinx.coroutines

    An import that begins with the package root is rewritten through it, so
    `kotlinx.coroutines.channels.Channel` becomes `common.src.channels.Channel`
    and lands on a real file. Anything else -- `java.io.IOException`,
    `kotlin.jvm.JvmStatic` -- is left exactly as written, so an import that
    leaves the corpus still reads as what the source says.

    A package root that is empty makes the first test vacuous, which is why the
    package's own first segment is checked as well: without it every
    `java.io.IOException` in okhttp would be recorded as
    `commonJvmAndroid.kotlin.java.io.IOException`, which is not a module anyone
    wrote.
    """
    dirs = [p for p in path.replace("\\", "/").split("/")[:-1] if p]
    pkg = [p for p in package.split(".") if p]
    if not pkg:
        return lambda dotted: dotted
    shared = 0
    while shared < min(len(dirs), len(pkg)) and \
            dirs[len(dirs) - shared - 1].lower() == pkg[len(pkg) - shared - 1].lower():
        shared += 1
    source_root = dirs[:len(dirs) - shared]
    package_root = [p.lower() for p in pkg[:len(pkg) - shared]]

    def to_module(dotted: str) -> str:
        parts = [p for p in dotted.split(".") if p]
        if not parts or parts[0].lower() != pkg[0].lower():
            return dotted
        if [p.lower() for p in parts[:len(package_root)]] != package_root:
            return dotted
        return ".".join(source_root + parts[len(package_root):])

    return to_module


def _file_doc(node, src: bytes, parsed: ParsedFile) -> None:
    """A KDoc block above the `package` line documents the file itself.

    It is the one KDoc in Kotlin that describes a file rather than a
    declaration, so it becomes the file node's rationale -- the same shape a
    Python module docstring gets. The licence header every file in both corpora
    opens with is skipped for free: it starts `/*`, not `/**`.
    """
    doc = _doc_above(node, src)
    if not doc:
        return
    doc_id = f"{parsed.prefix}{DOC_SUFFIX}"
    parsed.nodes.append(Node(id=doc_id, label=f"docstring of {parsed.prefix}",
                             kind="rationale", file=parsed.path, line=1,
                             text=doc[:600]))
    parsed.edges.append(Edge(source=parsed.prefix, target=doc_id,
                             relation="rationale_for", file=parsed.path, line=1))


def _import(node, src: bytes, parsed: ParsedFile, to_module) -> None:
    """Which file an import names, and what local name it binds.

    Kotlin writes the whole path, so this is exact -- after one step that is
    not obvious. The file is the prefix up to and including the first
    CAPITALISED segment, because Kotlin packages are lowercase and types are
    not:

        okhttp3.internal.concurrent.TaskRunner        -> file .../TaskRunner.kt
        okhttp3.internal.http2.Http2Connection.Listener -> file .../Http2Connection.kt
        okhttp3.internal.toCanonicalHost              -> a top-level function,
                                                         whose file is unknowable
        okhttp3.*                                     -> a package, no one file

    Without the cut, a nested-type import points at a directory that has no file
    node. The third line is the case Java does not have: Kotlin lets a file
    export top-level functions under the package name, and nothing in the import
    says which file holds them, so those resolve to the package and stay
    unresolved. That is an honest gap rather than a guess.

    The local name is bound as `<file>.<name>` rather than the raw path, because
    the resolver locates an imported symbol by dropping the last segment. An
    aliased import binds the alias, which then finds nothing -- correctly, since
    the corpus contains no symbol under the new name.
    """
    ident = _named(node, "qualified_identifier")
    if ident is None:                              # pragma: no cover - grammar
        return
    dotted = _text(ident, src)
    parts = [p for p in dotted.split(".") if p]
    if not parts:                                  # pragma: no cover - grammar
        return
    cut = next((i for i, p in enumerate(parts) if p[:1].isupper()), len(parts) - 1)
    file_dotted = to_module(".".join(parts[:cut + 1]))
    parsed.import_sites.append((file_dotted, node.start_point[0] + 1))
    # `import a.b.*` binds no single name; there is nothing to record.
    if _named(node, "*") is not None:
        return
    alias = None
    after_as = False
    for child in node.children:
        if child.type == "as":
            after_as = True
        elif after_as and child.type == "identifier":
            alias = _text(child, src)
    local = alias or parts[-1]
    parsed.imports[local] = f"{file_dotted}.{local}"


def _declared_names(body, src: bytes) -> frozenset:
    """The function names a type declares in its own body, companion included.

    This is what an unqualified call inside the type is checked against, and it
    is deliberately only what THIS file states -- an inherited method is not in
    it, and a call to one is refused rather than assumed.
    """
    out = set()
    for member in _members(body):
        if member.type == "function_declaration":
            name = _field(member, "name")
            if name is not None:
                out.add(_text(name, src))
    return frozenset(out)


def _type(node, src: bytes, reader: _Reader, parsed: ParsedFile,
          container: str) -> None:
    """A class, interface, object or type alias, and everything inside it.

    A nested type is minted FLAT inside its file -- `class Http2Connection {
    interface Listener }` becomes `..._listener`, not `..._http2connection_
    listener` -- while being contained by the type around it. Qualifying it
    would be tidier and would break resolution outright, because `import
    ...Http2Connection.Listener` and every receiver-typed call on a `Listener`
    go looking for `mint(file prefix, "Listener")`. The cost is that two nested
    types of one name in one file mint one id, and that collision is left
    visible in the report rather than papered over.
    """
    name_node = _field(node, "name") or _named(node, "identifier")
    if name_node is None:                          # pragma: no cover - grammar
        return
    name = _text(name_node, src)
    nid = reader.emit(name, "class", node, container, bases=_bases(node, src) or None)
    if nid is None:                                # pragma: no cover
        return
    parsed.defined_classes.add(name)

    # `class Server(val addr: HttpUrl)` declares a property in its parameter
    # list, and it is the same fact as a `val` written in the body. Read from
    # this type's OWN primary constructor rather than by searching the subtree,
    # which would hand a nested class's parameters to the type around it.
    listing = _named(_named(node, "primary_constructor"), "class_parameters")
    for param in listing.children if listing is not None else []:
        if param.type != "class_parameter":
            continue
        label_node = _named(param, "identifier")
        found = _type_name(_named(param, "user_type", "nullable_type"), src)
        if label_node is not None and found:
            label = _text(label_node, src)
            parsed.attr_types[f"{name}::{label}"] = found
            reader.declare(name, label, found)

    body = _named(node, *_BODIES)
    if body is None:
        return
    own = _declared_names(body, src)
    parent = _super_class(node, src)
    for member in _members(body):
        if member.type == "function_declaration":
            _function(member, src, reader, parsed, nid, name, own, parent)
        elif member.type == "secondary_constructor":
            _constructor(member, src, reader, parsed, nid, name, own, parent)
        elif member.type in _TYPES:
            _type(member, src, reader, parsed, nid)
        elif member.type in _LOOSE:
            # An `init { }` block, a property initializer and a `get()` body all
            # run real code and call real methods. The type itself is the caller,
            # which is the same answer Python gives for a call written in a class
            # body.
            if member.type == "property_declaration":
                reader.properties_of(member, name)
            reader.types_in(member, name)
            reader.calls_in(member, nid, own, parent)


def _function(node, src: bytes, reader: _Reader, parsed: ParsedFile,
              container: str, owner: str | None, own: frozenset,
              parent: str | None) -> None:
    """A `fun`, in any of the three places Kotlin puts one.

    Top level it is a `function`. Inside a type it is a `method` scoped by that
    type. A top-level EXTENSION -- `fun Response.commonHeaders()` -- is a
    `method` scoped by the type it extends; see the module docstring for why,
    and for what that costs.

    A MEMBER extension -- `fun URL.toHttpUrlOrNull()` written inside
    `HttpUrl`'s companion -- carries both names, because both are needed to
    tell it from its siblings and dropping either one loses symbols. The
    counts, over the 9,161 `fun` declarations in the two corpora: scoping by
    the enclosing type alone loses 327 of them, by the receiver alone 348, by
    neither 362, by both 284. okhttp writes three different
    `URL/URI/String.toHttpUrlOrNull` in one companion, and one of its test
    files writes `fun X.value()` inside a dozen different classes; only the
    pair of names separates all of them.

    What that costs is stated: a two-part scope is an id no resolution rule
    reconstructs, so a member extension is never found as a call target. It
    could not be found anyway -- reaching one needs a receiver of BOTH types at
    the call site, which is not evidence this parser has.
    """
    name_node = _field(node, "name")
    if name_node is None:                          # pragma: no cover - grammar
        return
    name = _text(name_node, src)
    receiver = _receiver_type(node, src)
    scope = [n for n in (owner, receiver) if n]
    kind = "method" if scope else "function"
    nid = reader.emit(name, kind, node, container, scope=scope or None, dedupe=True)
    if nid is None:
        # An overload of one already seen. `readPriority(a)` and
        # `readPriority(a, b)` are one entry point written twice, not two
        # symbols, and a caller writes one name for both.
        return
    holder = owner or receiver
    if holder:
        parsed.owner_of[nid] = holder
    scope = f"{holder}.{name}" if holder else name
    reader.types_in(node, scope)
    body = _named(node, "function_body")
    if body is not None:
        # A top-level extension is not inside the type it extends, so an
        # unqualified call in its body is not a call on that type. A member
        # extension IS inside its class and keeps that class's names.
        inside = owner is not None
        reader.calls_in(body, nid, own if inside else frozenset(),
                        parent if inside else None)


def _constructor(node, src: bytes, reader: _Reader, parsed: ParsedFile,
                 container: str, owner: str, own: frozenset,
                 parent: str | None) -> None:
    """A secondary constructor, emitted as a method labelled `constructor`.

    Kotlin spells construction `Server(addr)`, which is indistinguishable from a
    call to the class -- and the class node is the right target for it, so the
    constructor is deliberately NOT given the class's own name. Naming it
    `Server` would put two symbols called `Server` in the corpus and the
    resolver, faced with two candidates, would refuse every construction of it.

    The cost is that several secondary constructors on one type mint one id: 42
    of them across both corpora, so a handful of second declarations point at
    the first. Their calls are still attributed to a real symbol of the type.
    """
    nid = reader.emit("constructor", "method", node, container,
                      scope=[owner], dedupe=True)
    if nid is None:
        return
    parsed.owner_of[nid] = owner
    scope = f"{owner}.constructor"
    reader.types_in(node, scope)
    body = _named(node, "block")
    if body is not None:
        reader.calls_in(body, nid, own, parent)
