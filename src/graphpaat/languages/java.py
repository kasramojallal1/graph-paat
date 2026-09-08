"""Java, read with tree-sitter.

Java maps onto the same five shapes as everything else, and two of its habits
make it resolve unusually well.

**Every type is written down.** A field, a parameter, a local and a loop
variable all state their class in the source -- `private final Excluder
excluder;` is the author telling us outright. Nothing here is inferred the way
Python's `db = Database()` has to be.

**A file is a type.** `com/google/gson/Gson.java` holds `class Gson`, so an
import like `com.google.gson.internal.Streams` lands exactly on the file prefix
`com_google_gson_internal_streams` with no guessing at all.

Three decisions are worth stating up front because they cost something.

**Java has no free functions**, so this module never emits a `function` node --
only `class` and `method`. A constructor is a method whose name is the class
name, which is what Java itself calls it.

**A nested type is minted flat inside its file**, not qualified by the type that
encloses it: `class LinkedTreeMap { static class Node {...} }` becomes
`..._linkedtreemap_node`, contained by `LinkedTreeMap`. Qualifying it would be
tidier and would break resolution outright, because `import
com.google.gson.internal.LinkedTreeMap.Node` and every receiver-typed call go
looking for `mint(file prefix, "Node")`. The cost is real and was measured: of
the 164 nested types across the two corpora, exactly one file (commons-lang's
`LockingVisitors.java`) declares three called `Builder`, and two of those lose
their id. That is the price of making the other 161 resolvable.

**Overloads are one node, not many.** `substring(String, int)` and
`substring(String, int, int)` mint the same id, and a caller writes one name for
both -- they are one entry point, not two symbols. Emitting each signature put
1,504 duplicate method nodes into commons-lang alone, a third of its methods,
and every one of them would have been reported as a lost symbol. The first
declaration wins and the rest are dropped. Types are NOT deduplicated the same
way: two types with one name really are two things, and that collision is left
visible in the report.

What does not fit, stated plainly: **an anonymous class has no name to mint**.
`new Runnable() { public void run() {...} }` declares a type nobody can refer
to, and the 154 methods inside the two corpora's 87 anonymous classes get no
nodes. Their calls are not lost -- they are attributed to the method the class
was written inside, which is the closest true answer -- but "who implements
run" cannot be asked about them. A lambda is the same case with less to work
with. A field is deliberately not a node either, in Java as in every other
language here; its declared type is kept, which is what resolution needs.
"""
from __future__ import annotations

from ..ids import mint
from ..parse import DOC_SUFFIX, CallSite, Edge, Node, ParsedFile

EXTENSIONS = {".java"}
WHY = ""

_parser = None

# Everything Java calls a type. All five become `class` nodes: a reader asking
# "what is a JsonElement" does not care that one is declared with `interface`
# and another with `enum`, and giving each its own word would push Java's
# vocabulary into grouping, ranking and query.
_TYPES = ("class_declaration", "interface_declaration", "enum_declaration",
          "record_declaration", "annotation_type_declaration")

# The declaration list of each of those. `enum_body` is the odd one: its
# members hide one level deeper, inside `enum_body_declarations`.
_BODIES = ("class_body", "interface_body", "enum_body", "annotation_type_body")

# A callable member. `compact_constructor_declaration` is a record's `Pt { ... }`
# form, and `annotation_type_element_declaration` is the `String value();` inside
# an `@interface` -- both are named members and both are called like methods.
_CALLABLES = ("method_declaration", "constructor_declaration",
              "compact_constructor_declaration",
              "annotation_type_element_declaration")

# A declared field. Fields are not nodes -- no language here emits them -- but
# their types are the single richest source of resolution evidence in Java,
# because `this.excluder.excludeClass(...)` is only resolvable if we know what
# `excluder` is.
_FIELDS = ("field_declaration", "constant_declaration")


def available() -> bool:
    """True when the grammar is installed. The registry skips us otherwise, so
    a Python-only user never has to carry a Java grammar."""
    global _parser, WHY
    if _parser is not None:
        return True
    try:
        import tree_sitter_java
        from tree_sitter import Language, Parser
        _parser = Parser(Language(tree_sitter_java.language()))
        return True
    except Exception as exc:                      # pragma: no cover
        WHY = f"needs tree-sitter and tree-sitter-java ({exc})"
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
    """The bare class named by a type, or None if it does not name a class.

    Written out branch by branch rather than as a search for the first
    `type_identifier` anywhere below, because a search finds the wrong one:
    `List<String>` would answer `String`, and `Map<Type, InstanceCreator<?>>`
    would answer `Type`. The outer type is the one a call goes to.

    `int`, `boolean` and `void` deliberately answer None -- a primitive has no
    methods in the corpus and typing a variable with one would only produce
    refusals with a misleading reason.
    """
    if node is None:
        return None
    kind = node.type
    if kind == "type_identifier":
        return _text(node, src)
    if kind == "generic_type":
        # `ArrayList<String>` -> the raw type is the first child.
        return _type_name(node.children[0] if node.children else None, src)
    if kind == "scoped_type_identifier":
        # `com.google.gson.JsonElement` -> the last segment is the type.
        last = None
        for child in node.children:
            if child.type in ("type_identifier", "scoped_type_identifier"):
                last = child
        return _type_name(last, src) if last is not None else None
    if kind == "array_type":
        return _type_name(_field(node, "element"), src)
    if kind in ("annotated_type", "catch_type"):
        # `@Nullable String` and `catch (IOException | SQLException e)`.
        for child in node.children:
            found = _type_name(child, src)
            if found:
                return found
    return None


def _doc_above(node, src: bytes) -> str | None:
    """The Javadoc block directly above a declaration.

    `/** ... */` only. Java has a documentation form the language itself
    defines, and a `//` line above a method is a note about the line below it
    far more often than it is documentation -- taking those too buried the real
    ones. Measured on the two corpora: 6,098 Javadoc blocks against 2,354 line
    comments, so the ones being passed over are the minority as well as the
    weaker signal.

    A blank line anywhere in the run breaks it, and that check has to apply to
    the first comment as well -- guarding it on "we have walked past something
    already" lets a detached note through as documentation.

    Other comments in the run are walked past rather than treated as a wall.
    Annotations never intervene (`@Deprecated` is a child of the declaration),
    but a plain `//` marker does, and dropping the Javadoc for it is not
    theoretical: commons-lang writes `//@Immutable` between the Javadoc and
    `public class StringUtils`, which cost that class -- the single most
    documented type in the corpus -- its entire description.
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


class _Reader:
    def __init__(self, parsed: ParsedFile, src: bytes):
        self.p = parsed
        self.src = src
        self.minted: set[str] = set()

    def emit(self, name: str, kind: str, node, container: str,
             scope=None, bases=None, dedupe: bool = False) -> str | None:
        """One node, its containment edge and its Javadoc. None when dropped.

        `container` is passed in rather than derived from `scope`, because a
        nested type is contained by the type around it while its id is minted
        flat -- the two are genuinely different facts here.
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

    # ---- evidence for resolution -------------------------------------

    def declare(self, scope: str, name: str, cls: str | None) -> None:
        if cls:
            self.p.var_types[f"{scope}::{name}"] = cls

    def fields_of(self, declaration, owner: str) -> None:
        """`private final Excluder excluder;` -- the type is stated, so record it.

        Recorded twice on purpose. `attr_types` answers `this.excluder.x()`, and
        `var_types` answers the bare `excluder.x()` that Java writes far more
        often -- an unqualified field read looks exactly like a local read at
        the call site, and the resolver has one lookup for both.
        """
        cls = _type_name(_field(declaration, "type"), self.src)
        if not cls:
            return
        for declarator in declaration.children:
            if declarator.type != "variable_declarator":
                continue
            name_node = _field(declarator, "name")
            if name_node is None:
                continue
            name = _text(name_node, self.src)
            self.p.attr_types[f"{owner}::{name}"] = cls
            self.declare(owner, name, cls)

    def types_in(self, node, scope: str) -> None:
        """Every declared type inside a body. All four forms are declarations.

            void f(Server srv)              a parameter
            Server srv = ...;               a local
            for (Server srv : all)          a loop variable
            catch (IOException e)           a caught exception

        `var srv = new Server()` states the type on the other side of the `=`,
        so it is read from the initializer instead. Without that, every `var` in
        a modern file would be untyped and its calls unresolvable.
        """
        stack = list(node.children)
        while stack:
            n = stack.pop()
            if n.type in ("formal_parameter", "spread_parameter",
                          "catch_formal_parameter"):
                name = _field(n, "name") or _named(n, "identifier")
                found = _type_name(_field(n, "type") or _named(n, "catch_type"), self.src)
                if name is not None:
                    self.declare(scope, _text(name, self.src), found)
            elif n.type == "local_variable_declaration":
                declared = _field(n, "type")
                found = _type_name(declared, self.src)
                for declarator in n.children:
                    if declarator.type != "variable_declarator":
                        continue
                    name = _field(declarator, "name")
                    if name is None:
                        continue
                    cls = found
                    if cls == "var":
                        cls = _type_name(_field(_field(declarator, "value"), "type"),
                                         self.src)
                    self.declare(scope, _text(name, self.src), cls)
            elif n.type == "enhanced_for_statement":
                name = _field(n, "name")
                found = _type_name(_field(n, "type"), self.src)
                if name is not None:
                    self.declare(scope, _text(name, self.src), found)
            stack.extend(n.children)

    def calls_in(self, body, caller: str, scope: str) -> None:
        """Every call inside a body, with what it was called on.

        Five receiver shapes, and the first is the one that matters most: an
        unqualified `helper()` in Java is a call on the enclosing type, the same
        thing Python spells `self.helper()`. It is also the commonest call in
        both corpora -- 4,250 of them -- and treating it as a free function name
        would have sent it to the resolver's weakest rule, which happily matches
        a same-named method in an unrelated class.

        The exception is a statically imported name: `requireNonNull(x)` after
        `import static java.util.Objects.requireNonNull` is not on this type at
        all, so it is left as a plain name and refused honestly.
        """
        stack = [body]
        while stack:
            node = stack.pop()
            if node.type == "method_invocation":
                self._invocation(node, caller, scope)
            elif node.type == "object_creation_expression":
                # `new Excluder()` is a call to the class, and is often the only
                # edge tying a factory to the thing it builds.
                name = _type_name(_field(node, "type"), self.src)
                if name:
                    self.p.calls.append(CallSite(
                        caller=caller, file=self.p.path,
                        line=node.start_point[0] + 1, name=name))
            stack.extend(node.children)

    def _invocation(self, node, caller: str, scope: str) -> None:
        name_node = _field(node, "name")
        if name_node is None:
            return
        name = _text(name_node, self.src)
        line = node.start_point[0] + 1
        obj = _field(node, "object")

        def record(receiver=None, receiver_is_self=False, on_self=False):
            self.p.calls.append(CallSite(
                caller=caller, file=self.p.path, line=line, name=name,
                receiver=receiver, receiver_is_self=receiver_is_self,
                on_self=on_self))

        if obj is None:
            # Unqualified, so it is a member of the enclosing type -- unless a
            # static import brought the name in from elsewhere, in which case it
            # is left as a plain name and refused.
            record(on_self=name not in self.p.imports)
            return
        if obj.type == "this":
            record(on_self=True)
            return
        if obj.type == "field_access":
            inner = _field(obj, "object")
            field = _field(obj, "field")
            if inner is not None and inner.type == "this" and field is not None:
                record(receiver=_text(field, self.src), receiver_is_self=True)
                return
            record(receiver=_text(obj, self.src))
            return
        if obj.type == "identifier":
            who = _text(obj, self.src)
            # `StringUtils.isEmpty(s)` is a static call, and the receiver names
            # the class outright. Java's naming convention -- types capitalised,
            # variables not -- is the only thing separating it from a local, so
            # this is a convention and not a rule. It is safe because the
            # resolver still has to find that class AND that method on it before
            # it will draw an edge; a variable that happens to be capitalised
            # produces a refusal, not a wrong link. 2,616 calls across the two
            # corpora take this shape.
            if who[:1].isupper():
                self.declare(scope, who, who)
            record(receiver=who)
            return
        # A cast, a chain, a `super`, an array element. There is no name to type
        # here, so the receiver is recorded as the expression itself: it can
        # never match a declared variable, which means the call is counted as a
        # refusal rather than quietly disappearing from the coverage report.
        record(receiver=_text(obj, self.src)[:60])


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
        if node.type == "package_declaration":
            _package_doc(node, src, parsed)
            named = _named(node, "scoped_identifier", "identifier")
            package = _text(named, src) if named is not None else ""
        elif node.type == "import_declaration":
            _import(node, src, parsed)
        elif node.type in _TYPES:
            _type(node, src, reader, parsed, parsed.prefix)
    if package:
        _same_package(root, src, parsed, package)
    return True


def _same_package(root, src: bytes, parsed: ParsedFile, package: str) -> None:
    """A type in the same package needs no import, so nothing records it.

    `Gson.java` writes `new GsonBuilder(this)` with no import line, because both
    live in `com.google.gson`. With nothing to go on, the resolver falls back to
    matching the name across the whole corpus -- and finds two: the CLASS
    GsonBuilder and its own constructor, which carries the class's name by
    definition. Two candidates means it refuses, so every construction of a
    same-package type was being dropped.

    Java's own rule is that an unqualified type name resolves against the file's
    package, so each referenced name is registered as if it had been imported
    from there. It cannot invent a link: the resolver still has to find a file at
    that exact path holding a type of that exact name, so a name from
    `java.lang` simply finds nothing and falls through. Run last, so a real
    single-type import always wins.
    """
    stack = [root]
    seen: set[str] = set()
    while stack:
        node = stack.pop()
        if node.type == "type_identifier":
            seen.add(_text(node, src))
        stack.extend(node.children)
    for name in seen:
        if name[:1].isupper() and name not in parsed.imports \
                and name not in parsed.defined_classes:
            parsed.imports[name] = f"{package}.{name}.{name}"


def _package_doc(node, src: bytes, parsed: ParsedFile) -> None:
    """A `package-info.java` carries the package's documentation above `package`.

    It is the one Javadoc block in Java that describes a file rather than a
    declaration, so it becomes the file node's rationale -- the same shape a
    Python module docstring gets. An ordinary file's licence header sits in the
    same position and is skipped for free: it opens `/*`, not `/**`.
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


def _import(node, src: bytes, parsed: ParsedFile) -> None:
    """Which file an import names, and what local name it binds.

    Java writes the whole path, so this is exact -- but only after one step that
    is not obvious. The file is the prefix up to and including the first
    CAPITALISED segment, because Java packages are lowercase and types are not:

        com.google.gson.internal.Streams          -> file .../Streams.java
        com.google.gson.internal.LinkedTreeMap.Node -> file .../LinkedTreeMap.java
        static org.apache.commons.lang3.Validate.notNull -> file .../Validate.java
        com.google.gson.*                         -> a package, resolves to nothing

    Without it, a nested-type import points at a directory that has no file node
    and resolves to nothing, and a static import points at the class it came
    from instead of the file holding it.

    The local name is bound as `<file>.<name>` rather than the raw path, because
    the resolver locates an imported symbol by dropping the last segment. For
    `Streams` that means storing `...Streams.Streams`, which reads oddly and is
    the difference between the import resolving and not.
    """
    ident = _named(node, "scoped_identifier", "identifier")
    if ident is None:
        return
    dotted = _text(ident, src)
    parts = [p for p in dotted.split(".") if p]
    if not parts:
        return
    cut = next((i for i, p in enumerate(parts) if p[:1].isupper()), len(parts) - 1)
    file_dotted = ".".join(parts[:cut + 1])
    parsed.import_sites.append((file_dotted, node.start_point[0] + 1))
    # `import a.b.*` binds no single name; there is nothing to record.
    if _named(node, "asterisk") is None:
        local = parts[-1]
        parsed.imports[local] = f"{file_dotted}.{local}"


def _bases(node, src: bytes) -> list[str]:
    """`extends Base` and `implements Handler` are both inheritance here.

    Java separates them and the graph does not, because both mean the type gains
    the other's shape -- which is what a reader is asking about. An interface's
    `extends` is a third spelling of the same thing.
    """
    out: list[str] = []
    for holder in node.children:
        if holder.type == "superclass":
            for listed in holder.children:
                found = _type_name(listed, src)
                if found:
                    out.append(found)
        elif holder.type in ("super_interfaces", "extends_interfaces"):
            # `implements A, B` wraps its names in a type_list; `extends A, B`
            # on an interface does the same. A sealed type's `permits` clause is
            # deliberately NOT read: it names the subtypes, which is the edge
            # pointing the other way.
            listing = _named(holder, "type_list")
            for listed in listing.children if listing is not None else []:
                found = _type_name(listed, src)
                if found:
                    out.append(found)
    return out


def _members(body):
    """The declarations inside a type body, in order.

    An enum keeps its members one level down, behind the `;` that ends the
    constant list, so that level is flattened here.

    An enum CONSTANT is never a type of its own -- `DOUBLE` is a value, not a
    class -- but when it carries a body, the methods in that body are what the
    enum actually does, and they belong to the enum. gson's `ToNumberPolicy`
    is the case that forced this: it is a public API enum whose entire behaviour
    lives in four constant bodies, so skipping them left the type in the map
    with no methods at all. They are yielded LAST so that a method the enum
    declares in its own body wins the id, and the constant's override only fills
    a gap. What is lost is which constant an override came from: four
    implementations of `readNumber` become one node, pointing at the first.
    """
    overrides = []
    for child in body.children:
        if child.type == "enum_body_declarations":
            yield from child.children
        elif child.type == "enum_constant":
            constant_body = child.child_by_field_name("body")
            for member in constant_body.children if constant_body is not None else []:
                if member.type in _CALLABLES:
                    overrides.append(member)
        else:
            yield child
    yield from overrides


def _type(node, src: bytes, reader: _Reader, parsed: ParsedFile,
          container: str) -> None:
    name_node = _field(node, "name")
    if name_node is None:
        return
    name = _text(name_node, src)
    nid = reader.emit(name, "class", node, container, bases=_bases(node, src) or None)
    if nid is None:                                # pragma: no cover
        return
    parsed.defined_classes.add(name)

    # A record's components are its fields: `record Range(Node lo, Node hi)`.
    params = _field(node, "parameters")
    if params is not None:
        for param in params.children:
            if param.type == "formal_parameter":
                field_name = _field(param, "name")
                found = _type_name(_field(param, "type"), src)
                if field_name is not None and found:
                    label = _text(field_name, src)
                    parsed.attr_types[f"{name}::{label}"] = found
                    reader.declare(name, label, found)

    body = _named(node, *_BODIES)
    if body is None:
        return
    for member in _members(body):
        if member.type in _FIELDS:
            reader.fields_of(member, name)
        elif member.type in _CALLABLES:
            _callable(member, src, reader, parsed, name, nid)
        elif member.type in _TYPES:
            _type(member, src, reader, parsed, nid)
        elif member.type in ("static_initializer", "block"):
            # `static { ... }` runs real code and calls real methods. The class
            # itself is the caller, which is the same answer Python gives for a
            # call written in a class body.
            reader.types_in(member, name)
            reader.calls_in(member, nid, name)


def _callable(node, src: bytes, reader: _Reader, parsed: ParsedFile,
              owner: str, owner_id: str) -> None:
    name_node = _field(node, "name")
    if name_node is None:
        return
    name = _text(name_node, src)
    nid = reader.emit(name, "method", node, owner_id, scope=[owner], dedupe=True)
    if nid is None:
        return                                     # an overload of one already seen
    parsed.owner_of[nid] = owner
    scope = f"{owner}.{name}"
    reader.types_in(node, scope)
    body = _field(node, "body")
    if body is not None:
        reader.calls_in(body, nid, scope)
