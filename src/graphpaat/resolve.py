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
        self.top_level: dict[str, set[str]] = defaultdict(set)      # prefix -> {label}

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
                    self.top_level[parsed.prefix].add(node.label)
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
    """`from graphpaat.ids import mint` tells us `mint` lives in ids.py."""
    dotted = parsed.imports.get(name)
    if not dotted:
        return None
    parts = dotted.split(".")
    if len(parts) < 2 or parts[-1] != name:
        return None
    stem = normalise_path_part(parts[-2])
    for prefix in symbols.by_file:
        if prefix == stem or prefix.endswith(f"_{stem}"):
            return prefix
    return None


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
    """Map `graphify.extractors.base` onto the file node `extractors_base`.

    Matched from the most specific end so `graphify.cache` prefers `cache` over
    any other file whose name merely ends the same way.
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
            if site.name in symbols.top_level.get(parsed.prefix, set()):
                emit(site, mint(parsed.prefix, site.name), "same file")
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
    return edges, reasons
