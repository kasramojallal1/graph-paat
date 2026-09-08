"""Lua, read with tree-sitter.

Lua has no `class` keyword, so the interesting question is which of its tables
deserve a `class` node. Every module in the language is a table -- `local M = {}`
at the top, `return M` at the bottom -- and so is every object type. The two look
identical until you notice which one uses the colon.

    local M = {}                      a namespace: M.setup, M.reload
    function M.setup(opts) end

    local Picker = {}                 a type: picker:find(), picker:close()
    function Picker:find() end        the colon passes `self`

**A table dissolves into its file only when the file can stand in for it: it must
be a top-level name and have no colon methods.** Everything else that owns
functions becomes a `class` node.

That rule earns its keep twice. `M.helper()` inside the module that defines
`helper` is Lua's way of writing a plain same-file call, and dissolving M is what
lets it resolve -- a class node named "M" cannot, because forty-five files in one
real corpus define a table by that name and the graph refuses an ambiguous
answer. And `M.busted.run` next to `M.minitest.run` needs a container or the two
mint one id; `lazy`'s `minit.lua` loses two of five symbols without it.

The cost is stated plainly: a module table that happens to define one colon
method becomes a class node labelled `M`, which says nothing on its own and only
means something beside its file. Ten files in that same corpus are in exactly
that position, and every one of them is a real object type -- `LazyMeta`,
`LazyHandler` -- so the label is uninformative rather than wrong.

**A function bound to a named field of a table constructor is a declaration.**
It is not a side idiom -- it is how the language states what an object does:

    return Sorter:new {
      scoring_function = function(_, prompt, line) ... end,
    }
    return setmetatable({}, { __index = function(t, k) ... end })

Reading only `function M.f()` and `function T:f()` left those out, and one real
corpus is built almost entirely from them: `sorters.lua` writes eight
`scoring_function` fields and two `__index` fields and the graph had none of
them. See `_Reader.fields_in` for what scope they get and what it costs.

Inheritance comes from `setmetatable`, which is the only mechanism the language
itself offers: `setmetatable({}, {__index = Base})` and `setmetatable({}, Base)`
both mean the new table gains Base's methods, which is what `inherits` means
everywhere else here.

A doc comment is a run of `---` lines or a `--[[ ]]` block directly above a
declaration -- LuaDoc, and what every Lua tool reads. A plain `--` note is left
out; see `_doc_above`.
"""
from __future__ import annotations

import re

from ..ids import mint
from ..parse import DOC_SUFFIX, CallSite, Edge, Node, ParsedFile

EXTENSIONS = {".lua"}
WHY = ""

_parser = None

# A bare type name, as opposed to `table<string, Foo>`, `Foo[]`, `A|B` or an
# inline table type. Only the bare form can name a class node, and anything
# richer is discarded rather than guessed at.
_BARE_TYPE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# `--[[`, `--[=[`, `--[==[` -- Lua's long-bracket comment, at any level.
_BLOCK_COMMENT = re.compile(r"^--\[=*\[")

# Types that can never name a node. Without this, `---@param path string` types
# a variable as `string` and every `path:sub()` in the file reports "receiver
# typed, but no such method in the corpus" -- a refusal that reads like a near
# miss when nothing was ever close. Worse, a corpus that happens to define a
# table called `string` would collect every one of those as a real edge.
_PRIMITIVES = frozenset((
    "any", "boolean", "false", "function", "integer", "lightuserdata", "nil",
    "number", "self", "string", "table", "thread", "true", "unknown",
    "userdata",
))


def available() -> bool:
    """True when the grammar is installed. The registry skips us otherwise, so
    a Python-only user never has to carry a Lua grammar."""
    global _parser, WHY
    if _parser is not None:
        return True
    try:
        import tree_sitter_lua
        from tree_sitter import Language, Parser
        _parser = Parser(Language(tree_sitter_lua.language()))
        return True
    except Exception as exc:                      # pragma: no cover
        WHY = f"needs tree-sitter and tree-sitter-lua ({exc})"
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


def _values(node):
    """The real children of a list-ish node, without punctuation or comments."""
    if node is None:
        return []
    return [c for c in node.children if c.is_named and c.type != "comment"]


def _path(node, src: bytes) -> list[str] | None:
    """The dotted chain of a name expression, as identifiers.

        foo                 -> ["foo"]
        M.busted.run        -> ["M", "busted", "run"]
        Picker:new          -> ["Picker", "new"]
        require("x").select -> None

    None means a segment is not a plain name -- a call, an index by expression,
    a string method. Those are recorded as text and never resolved, because
    pretending `t[k]` names a symbol would put an edge where the source has a
    lookup.
    """
    if node is None:
        return None
    if node.type == "identifier":
        return [_text(node, src)]
    if node.type in ("dot_index_expression", "method_index_expression"):
        head = _path(node.children[0], src) if node.children else None
        tail = node.children[-1] if node.children else None
        if head is None or tail is None or tail.type != "identifier":
            return None
        return head + [_text(tail, src)]
    return None


def _assignment(stmt):
    """The assignment inside a statement, whether or not it says `local`."""
    if stmt.type == "assignment_statement":
        return stmt
    if stmt.type == "variable_declaration":
        return _named(stmt, "assignment_statement")
    return None


def _pairs(stmt, src: bytes):
    """(target expression, value expression) for one assignment statement.

    Lua assigns lists to lists -- `local ok, err = pcall(f)` -- and the two
    sides can be different lengths, so a target with no value still yields a
    pair. `local M = {}` and `M.Spec = Spec` are both handled here; the second
    is why binding cannot be looked for under `local` alone.
    """
    assign = _assignment(stmt)
    if assign is None:
        return []
    targets = _values(_named(assign, "variable_list"))
    values = _values(_named(assign, "expression_list"))
    return [(t, values[i] if i < len(values) else None)
            for i, t in enumerate(targets)]


def _decl_name(node):
    """The name expression of a `function` declaration, in any of its forms."""
    return _named(node, "identifier", "dot_index_expression",
                  "method_index_expression")


def _field_function(field, src: bytes):
    """`name = function() end` inside a table constructor, as (name, function).

    The key has to be the field's *first* child. `[k] = function() end` puts an
    identifier there too, but `k` is a variable holding the key rather than the
    key itself, and one real corpus writes three such fields in a single file
    (`lazy/view/init.lua`, all keyed `commit_pattern`) -- naming them after the
    variable would mint one id for the three and lose two. A computed key names
    nothing we can print, so it is skipped rather than guessed at.
    """
    if field.type != "field":
        return None
    key = field.children[0] if field.children else None
    if key is None or key.type != "identifier":
        return None
    fn = _named(field, "function_definition")
    return (_text(key, src), fn) if fn is not None else None


def _statement_of(node):
    """The statement a node sits in, which is where its doc comment lives."""
    while node.parent is not None and node.parent.type != "block":
        node = node.parent
    return node


def _bound_path(fn, src: bytes) -> list[str] | None:
    """The name a `function ... end` value was assigned, if it was assigned one.

        local wrapped_fn = function(...) end   -> ["wrapped_fn"]
        opts.attach_mappings = function(...)   -> ["opts", "attach_mappings"]
        cb(function() end)                     -> None

    Lua pairs a list of targets with a list of values, so the name is the target
    at this value's own position -- `local ok, run = nil, function() end` names
    the function `run`, not `ok`.

    The whole dotted target is kept, not just its last segment. `builtin/init.lua`
    sets `defaults.attach_mappings` and `opts.attach_mappings` eleven lines
    apart inside one function, and the table each belongs to is the only thing
    that tells them apart.
    """
    holder = fn.parent
    if holder is None or holder.type != "expression_list":
        return None
    assign = holder.parent
    if assign is None or assign.type != "assignment_statement":
        return None
    values = _values(holder)
    index = next((i for i, v in enumerate(values)
                  if v.start_byte == fn.start_byte), None)
    targets = _values(_named(assign, "variable_list"))
    if index is None or index >= len(targets):
        return None
    return _path(targets[index], src)


def _named_function(node, src: bytes):
    """`(name path, where the doc comment is, the function node)`, or None.

    One predicate for the three ways a function inside a body gets a name, so
    that emission and call attribution can never disagree about what counts:

        local function overlapping_ngrams(s, n)   a nested declaration
        local wrapped_fn = function(...)          a nested binding
        scoring_function = function(...)          a table field

    Anything it returns None for is anonymous -- a callback handed straight to
    another function -- and stays folded into whatever encloses it, because
    there is no name to answer a question with.
    """
    if node.type == "field":
        found = _field_function(node, src)
        return ([found[0]], node, found[1]) if found is not None else None
    if node.type == "function_declaration":
        name_node = _decl_name(node)
        path = _path(name_node, src) if name_node is not None else None
        return (path, node, node) if path else None
    if node.type == "function_definition":
        path = _bound_path(node, src)
        return (path, _statement_of(node), node) if path else None
    return None


def _declares_self(fn, src: bytes) -> bool:
    """Does this function take its own `self`, rather than closing over one?

    A closure written inside a method shares that method's `self`, so a
    `self:foo()` inside it resolves against the same class. `local function
    get_bufnr(self)` does not -- it names its own parameter, of a type nothing
    here states -- and inheriting the enclosing class there would put a
    confident edge on a guess.
    """
    for param in _values(_named(fn, "parameters")):
        if param.type == "identifier" and _text(param, src) == "self":
            return True
    return False


# --------------------------------------------------------------------------
# doc comments
# --------------------------------------------------------------------------

def _comment_body(node, src: bytes) -> list[str] | None:
    """The prose of one comment, or None when it is not documentation.

    Lua's documented form is `---` (LuaDoc, and what the language server reads)
    and the `--[[ ]]` block. A single `--` is excluded deliberately: it is
    overwhelmingly a note about the next line or a stretch of commented-out
    code, and measured over both corpora here only 25 of roughly a thousand
    top-level functions carry one, against 333 carrying a `---` run. Treating
    every `--` as documentation would have buried those 333 in line notes.

    The caller skips one rather than ending the run at it, which is not the
    same thing. telescope documents its central type on one line and annotates
    it on the next, and stopping at the second line threw the first away --
    that alone was thirteen lost docstrings across the two corpora here.
    """
    raw = _text(node, src)
    content = _named(node, "comment_content")
    body = _text(content, src) if content is not None else ""
    if raw.startswith("---"):
        # The grammar strips only the first `--`, so a LuaDoc line arrives with
        # a leading `-` still attached.
        return [body.lstrip("-")]
    if _BLOCK_COMMENT.match(raw):
        return body.splitlines() or [""]
    return None


def _prev_of(node):
    """The node above this one, stepping out of a block where the grammar put it.

    A comment that opens a block is hoisted out of the block and attached to
    the enclosing declaration, so the first statement of a function body has no
    previous sibling at all. Without this the `---@type` on the first line of
    every `while` body is invisible, which is exactly where it tends to sit.
    """
    prev = node.prev_sibling
    while prev is None and node.parent is not None and node.parent.type == "block":
        node = node.parent
        prev = node.prev_sibling
    return prev


def _doc_lines(node, src: bytes) -> list[str]:
    """The raw documentation lines directly above a declaration, in order."""
    lines: list[str] = []
    prev = _prev_of(node)
    while prev is not None and prev.type == "comment":
        # Only a comment on the line immediately above is documentation; one
        # separated by a blank line is a note about something else. The check
        # has to apply to the first comment too -- guarding it on "we already
        # have lines" lets any single detached comment through.
        if prev.end_point[0] + 1 < node.start_point[0]:
            break
        body = _comment_body(prev, src)
        if body is not None:
            lines[:0] = body
        node, prev = prev, prev.prev_sibling
    return lines


def _doc_above(node, src: bytes) -> str | None:
    """The prose of the doc comment above a declaration.

    Annotation lines are dropped. `---@param opts table` states a type, and the
    graph has nowhere to put a type; keeping them would spend the whole 600
    character budget on `@param @param @return` and push out the one sentence
    that says why the function exists. The description some authors write after
    an annotation goes with it, which is the cost.
    """
    kept = [line.strip() for line in _doc_lines(node, src)
            if not line.strip().startswith("@")]
    joined = " ".join(l for l in kept if l).strip()
    return joined[:600] or None


def _annotations(node, src: bytes) -> list[str]:
    """The `@...` lines of the doc comment above a declaration."""
    return [line.strip() for line in _doc_lines(node, src)
            if line.strip().startswith("@")]


def _annotated_types(node, src: bytes) -> dict[str, str]:
    """Variable types the source states in LuaDoc.

        ---@param picker Picker      ->  {"picker": "Picker"}
        ---@type Picker              ->  {"": "Picker"}

    This is a declaration, not an inference: the annotation is Lua's type
    system as far as any tool in the ecosystem is concerned, and it is the only
    place a Lua file says what a parameter is. Anything but a bare name --
    `table<string, X>`, `X[]`, `A|B` -- is discarded; see `_BARE_TYPE`.
    """
    out: dict[str, str] = {}
    for ann in _annotations(node, src):
        parts = ann.split()
        if parts[0] == "@param" and len(parts) >= 3:
            name, kind = parts[1].rstrip("?"), parts[2]
        elif parts[0] == "@type" and len(parts) >= 2:
            name, kind = "", parts[1]
        else:
            continue
        if kind in _PRIMITIVES or not _BARE_TYPE.match(kind):
            continue
        if name == "" or _BARE_TYPE.match(name):
            out[name] = kind
    return out


# --------------------------------------------------------------------------
# pass one: which tables are types
# --------------------------------------------------------------------------

def _top_level(root):
    """Every top-level statement, stepping through `do ... end`.

    A bare `do ... end` at the top of a file is Lua's way of keeping a few
    locals out of the module's namespace; it is a scope, not a container of a
    different kind, and a function declared inside it carries exactly the name
    it would carry outside. `make_entry.lua` wraps three of its public
    generators in one, and treating the block as opaque hid all three.

    Source order is preserved, because the artifact has to be the same on every
    run.
    """
    out, stack = [], list(reversed(root.children))
    while stack:
        node = stack.pop()
        if node.type == "do_statement":
            block = _named(node, "block")
            if block is not None:
                stack.extend(reversed(block.children))
            continue
        out.append(node)
    return out


class _Survey:
    """Which top-level names are bound, and which of them own functions.

    Two passes are needed because `function M.setup()` cannot be read until we
    know what M is, and M's colon methods may not appear until later in the
    file.
    """

    def __init__(self, root, src: bytes):
        self.bindings: dict[str, tuple] = {}   # name -> (statement, value)
        self.owners: set[str] = set()          # names that own a function here
        self.colon: set[str] = set()           # names with a `T:method()`
        self.nested: set[str] = set()          # names reached through a table

        for stmt in _top_level(root):
            for target, value in _pairs(stmt, src):
                path = _path(target, src)
                if path and path[-1] not in self.bindings:
                    self.bindings[path[-1]] = (stmt, value)
                if path and len(path) >= 2 and value is not None \
                        and value.type == "function_definition":
                    self._own(path, colon=False)
                if value is not None and value.type == "table_constructor":
                    for field in value.children:
                        if field.type != "field":
                            continue
                        name = _named(field, "identifier")
                        fn = _named(field, "function_definition")
                        if name is not None and fn is not None and path:
                            self._own(path + [_text(name, src)], colon=False)
            if stmt.type == "function_declaration":
                name_node = _decl_name(stmt)
                path = _path(name_node, src) if name_node is not None else None
                if path and len(path) >= 2:
                    self._own(path,
                              colon=name_node.type == "method_index_expression")

        # A table is a type when it uses the colon -- the language's own method
        # syntax, which passes `self` -- or when it is reached through another
        # table, where the file cannot serve as its container.
        self.classes = {n for n in self.owners
                        if (n in self.colon or n in self.nested)
                        and n in self.bindings}

    def _own(self, path: list[str], colon: bool) -> None:
        owner = path[-2]
        self.owners.add(owner)
        if colon:
            self.colon.add(owner)
        if len(path) >= 3:
            self.nested.add(owner)


# --------------------------------------------------------------------------
# emission
# --------------------------------------------------------------------------

class _Reader:
    def __init__(self, parsed: ParsedFile, src: bytes, survey: _Survey):
        self.p = parsed
        self.src = src
        self.survey = survey

    def emit(self, name: str, kind: str, node, doc_node=None, scope=None,
             bases=None, container=None) -> str:
        nid = mint(self.p.prefix, name, scope or [])
        line = node.start_point[0] + 1
        self.p.nodes.append(Node(id=nid, label=name, kind=kind, file=self.p.path,
                                 line=line, bases=bases))
        # A scope of one class name mints its own container; anything deeper
        # has to be told, because `scope[0]` is then the outermost enclosing
        # function and the thing that actually contains this one is the
        # innermost. Guessing would point `contains` at a node that is not
        # there.
        if container is None:
            container = mint(self.p.prefix, scope[0]) if scope else self.p.prefix
        self.p.edges.append(Edge(source=container, target=nid, relation="contains",
                                 file=self.p.path, line=line))
        doc = _doc_above(doc_node if doc_node is not None else node, self.src)
        if doc:
            doc_id = f"{nid}{DOC_SUFFIX}"
            self.p.nodes.append(Node(id=doc_id, label=f"docstring of {nid}",
                                     kind="rationale", file=self.p.path,
                                     line=line, text=doc))
            self.p.edges.append(Edge(source=nid, target=doc_id,
                                     relation="rationale_for",
                                     file=self.p.path, line=line))
        return nid

    # -- functions ---------------------------------------------------------

    def function(self, path: list[str], body, decl, doc_node, is_method: bool):
        """One function, wherever it was written and however it was named.

        `function M.setup()`, `M.setup = function()` and a `setup = function()`
        field of a table literal are the same declaration in Lua, and a reader
        asking where `setup` lives does not care which was used.
        """
        owner = path[-2] if len(path) >= 2 else None
        if owner is not None and owner in self.survey.classes:
            label, scope = path[-1], [owner]
            nid = self.emit(label, "method", decl, doc_node=doc_node, scope=scope)
            self.p.owner_of[nid] = owner
            qualified = f"{owner}.{label}"
        else:
            # The table dissolved into the file, so the member keeps its own
            # name: `utils.flatten` in utils.lua is the function `flatten`.
            label, owner = path[-1], None
            scope = []
            nid = self.emit(label, "function", decl, doc_node=doc_node)
            qualified = label
        self.types_in(doc_node if doc_node is not None else decl, qualified)
        has_self = is_method or owner is not None
        if body is not None:
            self.local_types(body, qualified)
            self.calls_in(body, nid, has_self)
            self.nested_in(body, scope + [label], nid, qualified, owner, has_self)
        return nid

    def nested_in(self, node, chain: list[str], container: str, qualified: str,
                  owner: str | None, has_self: bool) -> None:
        """Every function the source names below `node`, however it named it.

        Most of Lua's behaviour is written this way once a file gets past the
        module-table stage. `Sorter:new { scoring_function = ... }` declares a
        function as plainly as `function M.f()` does -- the table it sits in is
        an argument to a call, so no assignment target ever names it -- and
        `local function overlapping_ngrams(s, n)` inside a factory is a real
        function that other lines in the same factory call by name. Reading
        only the top level left both out: telescope's `sorters.lua` writes
        eight `scoring_function` fields and two `overlapping_ngrams` helpers
        and the graph had none of the ten.

        **The scope is the chain of enclosing named functions.** The names
        repeat hard at this depth -- those eight fields are all called
        `scoring_function`, and the file alone as scope would mint one id for
        the eight and drop seven. The function each one sits in is what tells
        them apart, so the id reads `sorters.get_fuzzy_file.scoring_function`.

        The cost is a longer id, and for something nested three deep, one that
        names every closure on the way down. That is the trade `mint` already
        makes for nested functions: verbose beats invisible, and beats a
        collision that silently keeps one of eight.

        Kind is `function`, not `method`. `Sorter:new {...}` plainly builds a
        Sorter, but `new` is a name a library chose rather than a rule of Lua --
        the same reason `_bases` refuses to read `Base:extend()`.
        """
        stack = [node]
        while stack:
            n = stack.pop()
            found = _named_function(n, self.src)
            if found is None:
                stack.extend(n.children)
                continue
            # Its body is walked below with the deeper scope, so it is not
            # pushed here -- doing both would emit everything inside it twice.
            path, decl, fn = found
            label, here = path[-1], chain + path[:-1]
            nid = self.emit(label, "function", decl, doc_node=decl, scope=here,
                            container=container)
            # A closure written inside a method shares that method's `self`;
            # one that declares its own does not. See `_declares_self`.
            mine = None if _declares_self(fn, self.src) else owner
            if mine is not None:
                self.p.owner_of[nid] = mine
            dotted = ".".join(path)
            inner = f"{qualified}.{dotted}" if qualified else dotted
            self.types_in(decl, inner)
            body = _named(fn, "block")
            if body is not None:
                self.local_types(body, inner)
                self.calls_in(body, nid, has_self and mine == owner)
                self.nested_in(body, here + [label], nid, inner, mine,
                               has_self and mine == owner)

    def types_in(self, doc_node, scope: str) -> None:
        """`---@param picker Picker` types the parameter it names."""
        for name, kind in _annotated_types(doc_node, self.src).items():
            if name:
                self.p.var_types[f"{scope}::{name}"] = kind

    def local_types(self, body, scope: str) -> None:
        """`---@type Async` above a local declaration types that local.

        This is Lua's `var buf Builder`: the file states outright what the
        variable is, and the alternative -- reading it back off whatever
        function produced the value -- would be a guess. Thirty-seven locals
        across the two corpora measured here say so and nothing else does.
        """
        stack = [body]
        while stack:
            node = stack.pop()
            if node.type in ("variable_declaration", "assignment_statement"):
                declared = _annotated_types(node, self.src).get("")
                targets = _values(_named(_assignment(node), "variable_list"))
                path = _path(targets[0], self.src) if targets else None
                if declared and path:
                    self.p.var_types[f"{scope}::{path[-1]}"] = declared
            stack.extend(node.children)

    # -- calls -------------------------------------------------------------

    def calls_in(self, body, caller: str, has_self: bool) -> None:
        stack = [body]
        while stack:
            node = stack.pop()
            # A function the source names has a node of its own, and its calls
            # belong to it. The same predicate decides both, so the two can
            # never disagree and count a call once per enclosing function.
            if node is not body and _named_function(node, self.src) is not None:
                continue
            if node.type == "function_call":
                self._call(node, caller, has_self)
            stack.extend(node.children)

    def _call(self, node, caller: str, has_self: bool) -> None:
        callee = node.children[0] if node.children else None
        if callee is None:
            return
        line = node.start_point[0] + 1
        if callee.type == "identifier":
            self.p.calls.append(CallSite(caller=caller, file=self.p.path,
                                         line=line, name=_text(callee, self.src)))
            return
        if callee.type not in ("dot_index_expression", "method_index_expression"):
            return
        tail = callee.children[-1] if callee.children else None
        if tail is None or tail.type != "identifier":
            return
        name = _text(tail, self.src)
        head = callee.children[0]
        path = _path(head, self.src)

        # `self:foo()` and `self.foo()` are both a call on the object's own
        # table. Lua's dot form does not pass self, but it still names a member
        # of the same type, which is what the graph is being asked about.
        if has_self and path == ["self"]:
            self.p.calls.append(CallSite(caller=caller, file=self.p.path,
                                         line=line, name=name, on_self=True))
            return
        if has_self and path is not None and len(path) == 2 and path[0] == "self":
            self.p.calls.append(CallSite(caller=caller, file=self.p.path, line=line,
                                         name=name, receiver=path[1],
                                         receiver_is_self=True))
            return
        # A call on this file's own namespace table -- `M.helper()` inside the
        # module that defines helper -- is a same-file call written the long
        # way. Left as a receiver it could never resolve: the receiver is a
        # module, and a module is not a type.
        if path is not None and len(path) == 1 and path[0] in self.survey.owners \
                and path[0] not in self.survey.classes:
            self.p.calls.append(CallSite(caller=caller, file=self.p.path,
                                         line=line, name=name))
            return
        receiver = ".".join(path) if path is not None else _text(head, self.src)
        self.p.calls.append(CallSite(caller=caller, file=self.p.path, line=line,
                                     name=name, receiver=receiver))


# --------------------------------------------------------------------------
# inheritance and imports
# --------------------------------------------------------------------------

def _bases(value, src: bytes) -> list[str]:
    """What a table inherits, when `setmetatable` says so.

        setmetatable({}, Base)                 Base's fields are looked up
        setmetatable({}, {__index = Base})     the explicit form

    This is the only inheritance the language itself has. Class libraries
    layer their own `Base:extend()` on top of it, and those are deliberately
    not read: `extend` is a name a library chose, not a rule of Lua, and
    inventing an `inherits` edge from a method name is a guess.
    """
    if value is None or value.type != "function_call":
        return []
    callee = value.children[0] if value.children else None
    if callee is None or callee.type != "identifier" \
            or _text(callee, src) != "setmetatable":
        return []
    args = _values(_named(value, "arguments"))
    if len(args) < 2:
        return []
    meta = args[1]
    if meta.type == "table_constructor":
        for field in meta.children:
            if field.type != "field":
                continue
            key = _named(field, "identifier")
            if key is not None and _text(key, src) == "__index":
                path = _path(field.children[-1], src)
                return [path[-1]] if path else []
        return []
    path = _path(meta, src)
    return [path[-1]] if path else []


def _require_path(value, src: bytes) -> tuple[str, str | None] | None:
    """The module a `require` names, and the member it was indexed with.

        require "telescope.utils"              -> ("telescope.utils", None)
        require("telescope.state").get_status  -> ("telescope.state", "get_status")

    Both call forms matter: Lua lets a single string or table argument be
    passed without parentheses, and telescope writes it that way throughout.
    """
    member = None
    if value.type == "dot_index_expression":
        tail = value.children[-1] if value.children else None
        if tail is not None and tail.type == "identifier":
            member = _text(tail, src)
        value = value.children[0]
    if value.type != "function_call":
        return None
    callee = value.children[0] if value.children else None
    if callee is None or callee.type != "identifier" \
            or _text(callee, src) != "require":
        return None
    arg = _named(_named(value, "arguments"), "string")
    body = _named(arg, "string_content") if arg is not None else None
    if body is None:
        return None
    text = _text(body, src).strip()
    return (text, member) if text else None


def _imports(stmt, src: bytes, parsed: ParsedFile) -> None:
    """Record every `require` in one statement.

    `require "a.b"` is already the dotted form the resolver matches against
    file prefixes, so nothing has to be rewritten. What it cannot reach is a
    package directory: `require "telescope.actions"` names
    `telescope/actions/init.lua`, whose prefix carries the `init`, and the
    string as written does not. Python's `__init__.py` has the identical shape,
    so this is left where it is rather than papered over here.
    """
    for target, value in _pairs(stmt, src):
        if value is None:
            continue
        required = _require_path(value, src)
        if required is None:
            continue
        module, member = required
        path = _path(target, src)
        if path:
            # `local f = require("m").f` binds a symbol, and naming the whole
            # dotted path is what lets a bare `f()` later resolve to it.
            parsed.imports[path[-1]] = f"{module}.{member}" if member else module
        parsed.import_sites.append((module, stmt.start_point[0] + 1))


# --------------------------------------------------------------------------

def parse(source: str, parsed: ParsedFile) -> bool:
    if not available():                            # pragma: no cover
        return False
    src = source.encode("utf-8")
    root = _parser.parse(src).root_node
    if root.has_error and not root.children:
        return False
    survey = _Survey(root, src)
    reader = _Reader(parsed, src, survey)

    # Class nodes are emitted before anything else, so a method's `contains`
    # edge always has a node to point at even when the type is bound after its
    # first use. Source order, not set order: the artifact has to be the same
    # on every run.
    for name in sorted(survey.classes,
                       key=lambda n: survey.bindings[n][0].start_byte):
        stmt, value = survey.bindings[name]
        reader.emit(name, "class", stmt, bases=_bases(value, src) or None)
        parsed.defined_classes.add(name)
        # The identifier IS the type -- `Picker:new()` and `Picker.static()`
        # are calls on the table that holds those methods. Stated by the
        # source, not inferred from a constructor's return value.
        parsed.var_types[f"{parsed.prefix}::{name}"] = name

    for stmt in _top_level(root):
        if stmt.type == "function_declaration":
            _declaration(stmt, src, reader)
        elif _assignment(stmt) is not None:
            _imports(stmt, src, parsed)
            _bound_functions(stmt, src, reader)
    return True


def _declaration(stmt, src: bytes, reader: _Reader) -> None:
    """`function f()`, `function M.f()`, `function T:f()`, `local function f()`."""
    name_node = _decl_name(stmt)
    path = _path(name_node, src) if name_node is not None else None
    if not path:
        return
    reader.function(path, _named(stmt, "block"), stmt, stmt,
                    is_method=name_node.type == "method_index_expression")


def _bound_functions(stmt, src: bytes, reader: _Reader) -> None:
    """Functions written as values rather than declarations.

    `utils.flatten = function(t)` outnumbers `function utils.flatten(t)` in
    real Lua -- 321 against 242 at the top level of one corpus measured here --
    so skipping this form would lose more functions than it kept.
    """
    for target, value in _pairs(stmt, src):
        path = _path(target, src)
        if not path or value is None:
            continue
        if value.type == "function_definition":
            reader.function(path, _named(value, "block"), stmt, stmt,
                            is_method=False)
        elif value.type == "table_constructor":
            rest = []
            for field in value.children:
                found = _field_function(field, src)
                if found is None:
                    rest.append(field)
                    continue
                key, fn = found
                reader.function(path + [key], _named(fn, "block"),
                                field, field, is_method=False)
            # A field that is itself a table keeps going: `M = { git = { run =
            # function() end } }` declares `run`, and the field names on the way
            # down are what keep it apart from the next table's `run`.
            for field in rest:
                key = field.children[0] if field.children else None
                deeper = [_text(key, src)] if key is not None \
                    and key.type == "identifier" else []
                _fields_under(field, path[-1], deeper, reader)
        else:
            # `actions.git_track_branch = make_git_branch_action { command = ... }`
            # -- the table is an argument to a call, so it is not the value the
            # name is bound to and the dissolve rule above does not apply to it.
            # The bound name is the only thing in the source that names what the
            # call builds, and it is load-bearing: `actions/init.lua` binds three
            # of these and every one of them has a field called `command`.
            _fields_under(value, path[-1], [], reader)


def _fields_under(node, name: str, deeper: list[str], reader: _Reader) -> None:
    """Named functions inside a top-level binding, below its own value.

    The bound name always opens the scope, because at this depth the field name
    alone is not distinctive and the file already holds several of these.
    """
    chain = [name] + deeper
    container = mint(reader.p.prefix, name) if name in reader.survey.classes \
        else reader.p.prefix
    owner = name if name in reader.survey.classes else None
    reader.nested_in(node, chain, container, name, owner,
                     has_self=owner is not None)
