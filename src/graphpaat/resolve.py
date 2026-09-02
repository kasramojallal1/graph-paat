"""Turn call sites into edges, and record every one we could not.

Two halves, and the second is the one that matters.

**Resolution.** A call names something; we try to prove what. Evidence comes
from the file's imports, its variable assignments, and the class a method
belongs to. Nothing here guesses -- an edge is only emitted when the target
exists as a node.

**Refusal.** D6: what we could not resolve is recorded with a reason rather
than dropped. Issue #1219's reporter put the problem exactly -- *"the absence
of a `calls` edge cannot be trusted"* -- and L3 confirmed graphify's artifact
carries no record of what a run failed to do.

One filter, and it is a judgement worth stating. The L4 rebuild found 9,100
unresolved calls on this corpus, and most receivers were dicts and lists
calling built-in methods -- `data.get()`, `lines.append()`. Emitting those as
edges would drown the map D2 has to keep inside a token budget, and they point
at nothing: `dict.get` is not in the corpus. So an UNRESOLVED edge is emitted
only when a symbol of that name EXISTS somewhere in the corpus -- meaning we
know a plausible target and failed to prove the link. Everything else is
counted in the coverage report but not drawn.
"""
from __future__ import annotations

import builtins
from collections import Counter, defaultdict

# Python's own names. A bare `set()` or `len()` is the builtin, even when the
# corpus happens to contain a function of that name -- found by sampling the
# "unique in corpus" rule, which had linked `extract_xaml` to a `set` defined
# in a vendored test fixture. This filter applies ONLY to that weakest rule:
# a same-file definition or an explicit import is real evidence of shadowing
# and still wins.
BUILTIN_NAMES = frozenset(dir(builtins))

# Methods of the built-in container and string types. A call named `get`,
# `items` or `copy` on a receiver we could not type is overwhelmingly
# `dict.get` -- not a corpus symbol that happens to share the name.
#
# These names are still COUNTED as refusals, so the coverage report stays
# honest; they are simply not drawn as gap markers in the graph. Measured
# 2026-09-02: they were 67% of the gap markers on `requests`, 30% on Django.
# A gap marker that appears on every ordinary dictionary access teaches an
# agent nothing and crowds out the ones that mean something.
#
# The cost, stated: a class that genuinely defines `get` and is genuinely
# called loses its gap marker. Resolved calls to such a method are unaffected
# -- only the unproven ones stop being drawn.
CONTAINER_METHODS = frozenset(
    m for tp in (dict, list, set, frozenset, tuple, str, bytes)
    for m in dir(tp) if not m.startswith("__")
)

from .ids import mint, normalise_path_part
from .parse import Edge, ParsedFile


class Symbols:
    """Corpus-wide index. Built after every file is parsed, because a call in
    one file usually lands in another."""

    def __init__(self, files: list[ParsedFile]) -> None:
        self.ids: set[str] = set()
        self.by_file: dict[str, ParsedFile] = {}
        self.label_to_ids: dict[str, list[str]] = defaultdict(list)
        self.class_home: dict[str, list[str]] = defaultdict(list)   # ClassName -> [prefix]
        self.methods: dict[str, set[str]] = defaultdict(set)        # prefix::Class -> {ids}
        # prefix -> {label: [real node ids]}. Real ids, not reconstructed ones:
        # a nested function's id carries its enclosing chain, so rebuilding it
        # from prefix+label alone produced an id nobody owned. That was up to
        # 6.2% of resolved edges pointing at nodes that do not exist.
        self.in_file: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

        for parsed in files:
            self.by_file[parsed.prefix] = parsed
            for node in parsed.nodes:
                if node.kind == "rationale":
                    continue
                self.ids.add(node.id)
                self.label_to_ids[node.label].append(node.id)
                if node.kind == "class":
                    self.class_home[node.label].append(parsed.prefix)
                if node.kind in ("function", "class"):
                    self.in_file[parsed.prefix][node.label].append(node.id)
            for method_id, owner in parsed.owner_of.items():
                self.methods[f"{parsed.prefix}::{owner}"].add(method_id)

    def class_prefix(self, cls: str) -> str | None:
        """Where a class lives. Refuses when two files define the same class
        name -- an ambiguous answer is worse than an honest gap."""
        homes = self.class_home.get(cls)
        return homes[0] if homes and len(homes) == 1 else None

    def method_of(self, prefix: str, cls: str, method: str) -> str | None:
        """The id of `cls.method` in the file at `prefix`, if it has one.

        Checking the method actually exists is what separates an inference from
        a guess: knowing `db` is a `Database` proves nothing if `Database` has
        no `connect`."""
        wanted = mint(prefix, method, [cls])
        return wanted if wanted in self.methods.get(f"{prefix}::{cls}", set()) else None


def _imported_prefix(name: str, parsed: ParsedFile, symbols: Symbols) -> str | None:
    """`from graphpaat.ids import mint` tells us `mint` lives in ids.py.

    The WHOLE module path matters, not its last segment. `from pydantic.v1.main
    import BaseModel` names v1/main.py; matching on `main` alone found the
    top-level main.py instead, so every v1 class appeared to inherit from the v2
    BaseModel. Any package with a module name repeated in a subpackage hits this.
    """
    dotted = parsed.imports.get(name)
    if not dotted:
        return None
    parts = dotted.split(".")
    if len(parts) < 2 or parts[-1] != name:
        return None
    return _module_prefix(".".join(parts[:-1]), symbols)


def _var_type(parsed: ParsedFile, variable: str) -> str | None:
    """The class a local variable was assigned or annotated with.

    Scope is deliberately loose: any binding of that name in the file counts.
    A tighter scope walk is more correct and, on this corpus, would change
    almost nothing -- worth revisiting when there is a measurement saying so.
    """
    for key, cls in parsed.var_types.items():
        if key.rpartition("::")[2] == variable:
            return cls
    return None


def _target_for(symbols: Symbols, cls: str | None, method: str) -> str | None:
    if not cls:
        return None
    prefix = symbols.class_prefix(cls)
    return symbols.method_of(prefix, cls, method) if prefix else None


def resolve_inheritance(files: list[ParsedFile], symbols: "Symbols") -> tuple[list[Edge], Counter]:
    """`class Poll(Model)` becomes an edge from Poll to Model.

    Parsing already records what each class inherits from. Without turning that
    into edges, "what subclasses Model" -- one of the most common questions
    about an object-oriented codebase -- has no answer at all.

    A base outside the corpus (`Exception`, `object`, a third-party class)
    cannot point at a node. It is drawn unresolved with the name, because what
    a class extends is part of what it IS, and saying nothing would imply it
    extends nothing.
    """
    edges: list[Edge] = []
    reasons: Counter = Counter()
    seen: set[tuple] = set()

    for parsed in files:
        for node in parsed.nodes:
            if node.kind != "class" or not node.bases:
                continue
            for base in node.bases:
                if (node.id, base) in seen:
                    continue
                seen.add((node.id, base))
                target = _class_target(base, parsed, symbols)
                if target and target != node.id:
                    edges.append(Edge(source=node.id, target=target, relation="inherits",
                                      file=node.file, line=node.line,
                                      reason="base class in corpus"))
                elif not target:
                    reasons["base class outside this corpus"] += 1
                    edges.append(Edge(
                        source=node.id, target=f"?{base}", relation="inherits",
                        file=node.file, line=node.line, resolved=False,
                        reason="base class outside this corpus (builtin or third party)"))
    return edges, reasons


def _class_target(name: str, parsed: ParsedFile, symbols: "Symbols") -> str | None:
    """Find the class node a base-class name refers to, or None.

    Same evidence as a call: defined here, imported here, or unique by name.
    """
    local = symbols.in_file.get(parsed.prefix, {}).get(name, [])
    for candidate in local:
        if candidate in symbols.ids:
            return candidate
    prefix = _imported_prefix(name, parsed, symbols)
    if prefix:
        candidate = mint(prefix, name)
        if candidate in symbols.ids:
            return candidate
    homes = symbols.class_home.get(name)
    if homes and len(homes) == 1:
        candidate = mint(homes[0], name)
        if candidate in symbols.ids:
            return candidate
    return None


def resolve_imports(files: list[ParsedFile], symbols: "Symbols") -> tuple[list[Edge], Counter]:
    """One edge per import statement, from the importing file to the imported one.

    An import that leaves the corpus (`import json`) cannot point at a node, so
    it is drawn as unresolved with the module named. That is deliberate: which
    third-party libraries a file depends on is exactly the kind of thing an
    agent asks, and answering "nothing" would be a lie.
    """
    edges: list[Edge] = []
    reasons: Counter = Counter()
    seen: set[tuple] = set()

    for parsed in files:
        for dotted, line in parsed.import_sites:
            target = _module_prefix(dotted, symbols)
            key = (parsed.prefix, target or dotted)
            if key in seen:
                continue
            seen.add(key)
            if target and target != parsed.prefix:
                edges.append(Edge(source=parsed.prefix, target=target,
                                  relation="imports", file=parsed.path, line=line,
                                  reason="module in corpus"))
            elif not target:
                reasons["imports a module outside this corpus"] += 1
                edges.append(Edge(
                    source=parsed.prefix, target=f"?{dotted}", relation="imports",
                    file=parsed.path, line=line, resolved=False,
                    reason="module outside this corpus (stdlib or third party)"))
    return edges, reasons


def _module_prefix(dotted: str, symbols: "Symbols") -> str | None:
    """Map `graphpaat.parse` onto the file node `parse`.

    Tried from the most specific end: `pydantic.v1.main` prefers `v1_main` over
    the top-level `main`, so a module name repeated inside a subpackage resolves
    to the right one.
    """
    parts = [normalise_path_part(p) for p in dotted.split(".") if p]
    for start in range(len(parts)):
        candidate = "_".join(parts[start:])
        if candidate in symbols.by_file:
            return candidate
    return None


def resolve(files: list[ParsedFile]) -> tuple[list[Edge], Counter]:
    symbols = Symbols(files)
    edges: list[Edge] = []
    reasons: Counter = Counter()
    seen: set[tuple] = set()

    def emit(site, target: str, how: str) -> None:
        key = (site.caller, target)
        if target == site.caller or key in seen:
            return
        seen.add(key)
        edges.append(Edge(source=site.caller, target=target, relation="calls",
                          file=site.file, line=site.line, reason=how))

    def refuse(site, reason: str) -> None:
        reasons[reason] += 1
        if site.name in CONTAINER_METHODS:
            return
        # Only draw the gap when a symbol of this name exists somewhere: then
        # we know a plausible target and failed to prove the link, which is
        # worth an agent's attention. `data.get()` is not.
        if symbols.label_to_ids.get(site.name):
            edges.append(Edge(
                source=site.caller, target=f"?{site.name}", relation="calls",
                file=site.file, line=site.line, resolved=False, reason=reason))

    for parsed in files:
        for site in parsed.calls:

            # ---- self.foo() ------------------------------------------
            if site.on_self:
                owner = parsed.owner_of.get(site.caller)
                target = symbols.method_of(parsed.prefix, owner, site.name) if owner else None
                if target:
                    emit(site, target, "self-method")
                else:
                    refuse(site, "called on self, but no such method on the class")
                continue

            # ---- self.attr.foo() -------------------------------------
            if site.receiver_is_self:
                owner = parsed.owner_of.get(site.caller)
                cls = parsed.attr_types.get(f"{owner}::{site.receiver}") if owner else None
                target = _target_for(symbols, cls, site.name)
                if target:
                    emit(site, target, "self-attribute-typed")
                else:
                    refuse(site, "receiver is a self attribute of unknown type")
                continue

            # ---- x.foo()  -- D1, receiver typing ----------------------
            if site.receiver is not None:
                cls = _var_type(parsed, site.receiver)
                if cls is None:
                    refuse(site, "receiver type unknown")
                    continue
                target = _target_for(symbols, cls, site.name)
                if target:
                    emit(site, target, "receiver-typed")
                else:
                    refuse(site, "receiver typed, but no such method in the corpus")
                continue

            # ---- bare foo() ------------------------------------------
            local = symbols.in_file.get(parsed.prefix, {}).get(site.name, [])
            if len(local) == 1:
                emit(site, local[0], "same file")
                continue
            if len(local) > 1:
                # Two functions of that name in one file, in different scopes.
                # Choosing would be a guess; the map says so instead.
                refuse(site, f"name defined in {len(local)} scopes of this file, cannot choose")
                continue
            prefix = _imported_prefix(site.name, parsed, symbols)
            if prefix:
                candidate = mint(prefix, site.name)
                if candidate in symbols.ids:
                    emit(site, candidate, "imported")
                    continue
            candidates = symbols.label_to_ids.get(site.name, [])
            if site.name in BUILTIN_NAMES:
                refuse(site, "python builtin")
                continue
            if len(candidates) == 1:
                emit(site, candidates[0], "unique in corpus")
                continue
            if len(candidates) > 1:
                refuse(site, f"name defined in {len(candidates)} places, cannot choose")
            else:
                refuse(site, "not defined in this corpus (builtin or third party)")

    import_edges, import_reasons = resolve_imports(files, symbols)
    edges.extend(import_edges)
    reasons.update(import_reasons)

    inherit_edges, inherit_reasons = resolve_inheritance(files, symbols)
    edges.extend(inherit_edges)
    reasons.update(inherit_reasons)
    return edges, reasons
