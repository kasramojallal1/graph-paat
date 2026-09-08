"""Bash, read with tree-sitter.

Bash is the shortest map in this directory, and the shortness is the honest
answer rather than a shortfall: the language has no classes, no methods and no
types, so a shell script produces `file`, `function` and `rationale` nodes and
nothing else. Nothing is invented to fill the other two slots -- a `class` node
made out of a naming convention would be a claim the source never makes.

Three things about the language decide everything here.

**A call is just a word.** `foo` is a call to a shell function, an external
binary, a builtin or an alias, and the syntax is identical for all four. There
is no receiver to type and no import to follow, so every call arrives at the
resolver as a bare name. That means a lot of refusals -- `ls`, `grep`, `echo`
and `printf` are not in the corpus and never will be -- and it is worth being
precise about what those cost. A refusal is counted in the coverage report but
only DRAWN as a gap marker when a symbol of that name exists somewhere in the
corpus, so the ordinary shell vocabulary is recorded as unresolved and stays
out of the map. What an agent sees is the calls between the corpus's own
functions.

**The function namespace is flat and global.** A sourced file does not create a
scope; its functions land in the same single table as everybody else's, and the
last definition of a name wins. So the resolver's weakest rule -- "a name
defined exactly once in the corpus" -- is not a heuristic for shell. It is how
the shell itself resolves a name, which is why no attempt is made here to bind
a sourced file's functions to a local alias: there is no local alias to bind.

**A source path is usually half a variable.** Real scripts write
`source "$BASH_IT/lib/log.bash"`, not a literal path, because the root is only
known at run time. The literal tail after the last expansion is the part we can
honestly claim to know, and it is taken only when that expansion ended on a `/`
-- that is, when it stood for whole path components. When the variable sits
inside a filename (`"$THEME/$THEME.theme.bash"`) the literal tail is not a path
at all and nothing is recorded.

A doc comment is the run of `#` lines directly above a definition, which is all
the documentation shell has. Tool directives are excluded; see `_DIRECTIVE`.
"""
from __future__ import annotations

import posixpath
import re

from ..ids import mint
from ..parse import DOC_SUFFIX, CallSite, Edge, Node, ParsedFile

EXTENSIONS = {".sh", ".bash"}
WHY = ""

_parser = None

# `source x` and `. x` are ordinary commands to the grammar, so they have to be
# named here or every source line would be emitted twice: once as an import and
# once as a call to a function named "source". The second is never true, and it
# would add a refusal per source line to the coverage report.
_SOURCE_WORDS = {"source", "."}

# Nodes whose value is only known at run time. Everything in a path before one
# of these is unusable, because we cannot say what it expanded to.
_EXPANSIONS = ("simple_expansion", "expansion", "command_substitution",
               "arithmetic_expansion", "process_substitution")

# A `#` line that is an instruction to a tool rather than documentation. Shell
# has no second comment syntax to separate the two -- `#` carries the shebang,
# the linter pragmas and the actual prose -- so the noise has to be named. Left
# in, `# shellcheck disable=SC2329` becomes the stated reason for a function,
# which is worse than that function having no stated reason at all: it is a
# claim about the code that is not about the code.
#
# Only these three forms are excluded, and each was seen sitting directly above
# a real function in the corpora read here. A wider filter would start deleting
# sentences, which is the expensive mistake in the other direction.
_DIRECTIVE = re.compile(
    r"^(?:!"                            # #!/usr/bin/env bash
    r"|shellcheck\b"                    # # shellcheck disable=SC2155
    r"|(?:vi|vim|ex|emacs)\s*:"         # # vim: set ft=sh:
    r"|-\*-)",                          # # -*- mode: sh -*-
    re.IGNORECASE)

# Extensions stripped when a source path becomes a module name, so that
# `lib/log.bash` matches the file prefix minted for `lib/log.bash`.
_SUFFIX = re.compile(r"\.(sh|bash)$")


def available() -> bool:
    """True when the grammar is installed. The registry skips us otherwise, so
    a Python-only user never has to carry a Bash grammar."""
    global _parser, WHY
    if _parser is not None:
        return True
    try:
        import tree_sitter_bash
        from tree_sitter import Language, Parser
        _parser = Parser(Language(tree_sitter_bash.language()))
        return True
    except Exception as exc:                      # pragma: no cover
        WHY = f"needs tree-sitter and tree-sitter-bash ({exc})"
        return False


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _field(node, name: str):
    return node.child_by_field_name(name)


def _literal_word(node, src: bytes) -> str | None:
    """The name of a command, but only when the source spells it out.

    A `command_name` is usually a single `word`, and then we know exactly what
    is being run. It can also be a string, an expansion or a concatenation --
    `"$cmd" arg`, `${runner} arg` -- and in those the name is decided at run
    time. Recording the text of one of those as a call name would invent a
    function called `$cmd`, so they are dropped. Measured over two real shell
    repositories, 7,999 of 8,114 command names are plain words; the remaining
    1.4% is genuinely unknowable without running the script.
    """
    if node is None or len(node.children) != 1:
        return None
    child = node.children[0]
    return _text(child, src) if child.type == "word" else None


def _doc_above(node, src: bytes) -> str | None:
    """The run of `#` lines directly above a definition.

    Shell has no docstring; this is its equivalent, and skipping it would leave
    every shell node without the "why" that Python nodes carry.
    """
    lines: list[str] = []
    current = node
    prev = node.prev_sibling
    while prev is not None and prev.type == "comment":
        # Only a comment on the line immediately above is documentation; one
        # separated by a blank line is a note about something else. The check
        # has to apply to the first comment too -- guarding it on "we already
        # have lines" would let any single detached comment through.
        if prev.end_point[0] + 1 < current.start_point[0]:
            break
        # `foo() { :; }  # trailing note` puts a comment on the line of the
        # PREVIOUS definition, where it explains that one. A comment sharing a
        # line with code is never documentation for what comes after it.
        before = prev.prev_sibling
        if before is not None and before.end_point[0] == prev.start_point[0]:
            break
        lines.insert(0, _text(prev, src).lstrip("#").strip())
        current, prev = prev, prev.prev_sibling
    kept = [line for line in lines if line and not _DIRECTIVE.match(line)]
    joined = " ".join(kept).strip()
    return joined or None


class _Reader:
    def __init__(self, parsed: ParsedFile, src: bytes):
        self.p = parsed
        self.src = src

    def emit(self, name: str, kind: str, node, scope: list[str], container: str) -> str:
        nid = mint(self.p.prefix, name, scope)
        line = node.start_point[0] + 1
        self.p.nodes.append(Node(id=nid, label=name, kind=kind, file=self.p.path,
                                 line=line))
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

    def walk(self, node, scope: list[str], container: str) -> None:
        """One pass over the file, carrying the enclosing FUNCTION chain.

        The chain is functions only, deliberately. A definition inside an `if`
        or a `case` branch is still a top-level function -- the shell puts it in
        the same global table as any other -- so a block must not become part of
        its id or of its containment.

        The walk is an explicit stack rather than recursion because shell nests
        one level per `&&`: a generated script joining four thousand commands
        produces a tree four thousand deep, and a recursive walk would exhaust
        the interpreter on a file the parser handled without complaint.
        `reversed` keeps the traversal in document order, so import sites come
        out in the order they were written.
        """
        stack = [(child, scope, container) for child in reversed(node.children)]
        while stack:
            current, here, owner = stack.pop()
            if current.type == "function_definition":
                name_node = _field(current, "name")
                body = _field(current, "body")
                if name_node is None or body is None:   # pragma: no cover
                    continue
                name = _text(name_node, self.src)
                nid = self.emit(name, "function", current, here, owner)
                # A nested definition is qualified by the function it sits in,
                # exactly as a nested Python function is. Shell hoists it into
                # the global table once the outer function runs, but two
                # helpers of the same name in one file are still two different
                # pieces of code and must not mint one id.
                inner = here + [name]
                stack.extend((c, inner, nid) for c in reversed(body.children))
                continue
            if current.type == "command":
                self._command(current, owner)
            stack.extend((c, here, owner) for c in reversed(current.children))

    def _command(self, node, caller: str) -> None:
        """A command is either a source statement or a call. Never both.

        There is no receiver and no self here: shell has one flat function
        table, so a call site carries a name and nothing else.
        """
        name_node = _field(node, "name")
        name = _literal_word(name_node, self.src)
        if name is None:
            return
        if name in _SOURCE_WORDS:
            self._source(node, name_node)
            return
        self.p.calls.append(CallSite(caller=caller, file=self.p.path,
                                     line=node.start_point[0] + 1, name=name))

    def _source(self, node, name_node) -> None:
        """`source lib/log.bash` and `. lib/log.bash`.

        Collected on the walk rather than from the top-level statement list,
        because a shell script sources conditionally -- inside an `if`, inside
        a function -- far more often than it sources at the top of the file.

        `parsed.imports` is deliberately left empty. It maps a local name onto
        the module that defines it, and sourcing binds no local names: the
        sourced file's functions simply join the one global table. Any entry
        here would be a guess about which file defines which name.
        """
        argument = _first_argument(node, name_node)
        if argument is None:
            return
        literal, expanded = _literal_path(argument, self.src)
        if expanded and not literal.startswith("/"):
            # The variable sat inside a path component rather than standing for
            # whole ones, so what is left is a fragment of a name, not a path.
            return
        dotted = _dotted(literal)
        if dotted:
            self.p.import_sites.append((dotted, node.start_point[0] + 1))


def parse(source: str, parsed: ParsedFile) -> bool:
    if not available():                            # pragma: no cover
        return False
    src = source.encode("utf-8")
    root = _parser.parse(src).root_node
    if root.has_error and not root.children:
        return False
    # The file node is the outermost container, so a command written at the top
    # level of a script -- which is where a shell script does much of its work
    # -- is recorded as called by the file rather than thrown away.
    _Reader(parsed, src).walk(root, [], parsed.prefix)
    return True


def _first_argument(node, name_node):
    """The first thing after the command's own name.

    Byte offsets rather than child position: a command may be preceded by
    variable assignments (`FOO=1 source x`) and followed by redirections, and
    only what sits after the name is the path.
    """
    for child in node.children:
        if child.start_byte >= name_node.end_byte and child.type not in (";", "&"):
            return child
    return None


def _literal_path(node, src: bytes) -> tuple[str, bool]:
    """The literal text of a source argument, and whether a variable preceded it.

    Four shapes appear in real scripts:

        source lib/log.bash                 a bare word -- the whole path
        source "${BASH_IT}/lib/log.bash"    a string; the tail is the path
        source "${BASH_IT}"/lib/log.bash    a concatenation of the two
        source "$1"                         entirely a variable -- unknowable

    Accumulation resets at every expansion, so what comes back is the literal
    AFTER the last variable. In the second and third shapes that variable stood
    for a root directory and the tail is a path relative to it; the caller
    checks the leading `/` that proves it.
    """
    literal, expanded = "", False
    stack = [node]
    while stack:
        current = stack.pop(0)
        if current.type in _EXPANSIONS:
            literal, expanded = "", True
        elif current.type in ("word", "string_content"):
            literal += _text(current, src)
        elif current.type == "raw_string":
            literal += _text(current, src).strip("'")
        else:
            stack = list(current.children) + stack
    return literal, expanded


def _dotted(path: str) -> str | None:
    """A filesystem path as the dotted module name the resolver matches on.

    The extension goes, `.`, `..` and `~` at the front go, and `/` becomes `.`
    -- `lib/log.bash` becomes `lib.log`, which matches the file prefix minted
    for that file, `lib_log`.

    A path is NOT anchored to the sourcing file's directory, unlike Ruby's
    `require_relative`. Shell resolves a relative source against the working
    directory of whoever ran the script, not against the script's own location,
    so anchoring it would state a relationship the language does not have.
    """
    cleaned = posixpath.normpath(path.replace("\\", "/"))
    parts = [p for p in cleaned.split("/") if p]
    while parts and parts[0] in (".", "..", "~"):
        parts.pop(0)
    if not parts:
        return None
    parts[-1] = _SUFFIX.sub("", parts[-1])
    return ".".join(p for p in parts if p).strip(".") or None
