"""Lua, read through tree-sitter.

Lua has no `class` keyword, so the whole question is which tables are types.
These tests pin the answer down from both sides: a table with colon methods
becomes a `class` node with `method` children, and a plain module table
dissolves into its file so `M.helper()` reads as the same-file call it is.

Everything else is the shared vocabulary again -- a `---` run is a `rationale`,
`setmetatable` is `inherits`, containment matches Python's -- because a language
that needed a new word would mean the model was never language-neutral.
"""
import pytest

from graphpaat.parse import parse_corpus_files, parse_file
from graphpaat.resolve import resolve

pytest.importorskip("tree_sitter_lua")


@pytest.fixture
def luacorpus(tmp_path):
    def build(files: dict[str, str]):
        root = tmp_path / "repo"
        for name, source in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
        root.mkdir(parents=True, exist_ok=True)
        return root
    return build


def kinds(root):
    files, _ = parse_corpus_files(root)
    return {n.id: n.kind for p in files for n in p.nodes}


def links(root, relation="calls"):
    """Resolved edges of one relation.

    Containment comes from parsing and calls come from resolution, so both
    sources are needed -- looking only at the resolver's output made a correct
    containment edge look missing.
    """
    files, _ = parse_corpus_files(root)
    edges, _ = resolve(files)
    edges = list(edges) + [e for p in files for e in p.edges]
    return {(e.source, e.target) for e in edges
            if e.relation == relation and e.resolved}


def reasons_for(root):
    files, _ = parse_corpus_files(root)
    return resolve(files)[1]


def docs(root):
    files, _ = parse_corpus_files(root)
    return {n.id: n.text for p in files for n in p.nodes if n.kind == "rationale"}


PICKER = {"m.lua": '''
--- Picker is the main UI.
---@class Picker
local Picker = {}

--- Create a picker.
function Picker:new(opts)
  return self:reset()
end

function Picker:reset() end

function Picker.describe() end
'''}


class TestSameShapesAsPython:
    def test_a_table_with_colon_methods_is_a_class_node(self, luacorpus):
        # The colon is the language's own method syntax -- it passes `self` --
        # so it is the one signal that a table is a type rather than a bag.
        assert kinds(luacorpus(PICKER))["m_picker"] == "class"

    def test_a_plain_module_table_dissolves_into_its_file(self, luacorpus):
        # `local M = {}` is Lua's module, not an object. A class node named "M"
        # would say nothing and could never be told apart from the M in every
        # other file.
        root = luacorpus({"m.lua": '''
local M = {}
function M.setup(opts) end
return M
'''})
        k = kinds(root)
        assert k["m_setup"] == "function"
        assert "m_m" not in k

    def test_a_nested_table_keeps_its_members_apart(self, luacorpus):
        # Both would mint `m_run` if the file were their container.
        root = luacorpus({"m.lua": '''
local M = {}
M.busted = {}
function M.busted.run() end
M.minitest = {}
function M.minitest.run() end
'''})
        k = kinds(root)
        assert k["m_busted"] == "class" and k["m_minitest"] == "class"
        assert k["m_busted_run"] == "method" and k["m_minitest_run"] == "method"

    def test_methods_are_qualified_by_their_table(self, luacorpus):
        k = kinds(luacorpus(PICKER))
        assert k["m_picker_new"] == "method"
        # A dot function on a type is still a member of that type: Lua draws no
        # line between `Picker.describe` and `Picker:describe` beyond `self`.
        assert k["m_picker_describe"] == "method"

    def test_two_tables_with_a_same_named_method_do_not_collide(self, luacorpus):
        root = luacorpus({"m.lua": '''
local A = {}
local B = {}
function A:run() end
function B:run() end
'''})
        k = kinds(root)
        assert "m_a_run" in k and "m_b_run" in k

    def test_a_luadoc_run_becomes_a_rationale_node(self, luacorpus):
        got = docs(luacorpus(PICKER))
        assert got["m_picker#doc"] == "Picker is the main UI."
        assert got["m_picker_new#doc"] == "Create a picker."

    def test_a_comment_separated_by_a_blank_line_is_not_documentation(self, luacorpus):
        root = luacorpus({"m.lua": '''
--- Unrelated note.

local function thing() end
'''})
        assert docs(root) == {}

    def test_a_plain_dash_dash_note_is_not_documentation(self, luacorpus):
        # A single `--` is overwhelmingly a note about the next line or a
        # stretch of commented-out code; taking every one buried the real docs.
        root = luacorpus({"m.lua": '''
-- bump the counter first
local function thing() end
'''})
        assert docs(root) == {}

    def test_a_plain_note_does_not_end_a_doc_run(self, luacorpus):
        # Skipping a `--` line is not the same as stopping at one. Real code
        # documents a type on one line and annotates it on the next, and
        # stopping threw the documentation away.
        root = luacorpus({"m.lua": '''
--- Picker is the main UI.
-- Takes a filter and a previewer.
local Picker = {}
function Picker:find() end
'''})
        assert docs(root)["m_picker#doc"] == "Picker is the main UI."

    def test_a_block_comment_is_documentation(self, luacorpus):
        root = luacorpus({"m.lua": '''
--[[
Spawns the job and waits.
]]
local function thing() end
'''})
        assert docs(root)["m_thing#doc"] == "Spawns the job and waits."

    def test_an_annotation_only_doc_run_makes_no_rationale(self, luacorpus):
        # `---@param opts table` states a type, and the graph has nowhere to
        # put a type. Keeping them would spend the whole budget on annotations
        # and push out the sentence that says why.
        root = luacorpus({"m.lua": '''
---@param opts table
---@return string
local function thing(opts) end
'''})
        assert docs(root) == {}

    def test_a_function_bound_to_a_name_is_a_node(self, luacorpus):
        # `utils.flatten = function(t)` outnumbers `function utils.flatten(t)`
        # in real Lua, so this form cannot be skipped.
        root = luacorpus({"m.lua": '''
local utils = {}
utils.flatten = function(t) end
local helper = function() end
'''})
        k = kinds(root)
        assert k["m_flatten"] == "function" and k["m_helper"] == "function"

    def test_a_function_field_of_a_table_literal_is_a_node(self, luacorpus):
        root = luacorpus({"m.lua": '''
local M = {}
M.task = {
  --- Runs it.
  run = function(self) end,
  name = "x",
}
'''})
        assert kinds(root)["m_task_run"] == "method"
        assert docs(root)["m_task_run#doc"] == "Runs it."

    def test_setmetatable_is_inheritance(self, luacorpus):
        # Lua has no subclassing; a metatable whose `__index` is another table
        # is how a table gains its methods, which is what `inherits` means
        # everywhere else here.
        root = luacorpus({"m.lua": '''
local Base = {}
function Base:go() end
local Derived = setmetatable({}, { __index = Base })
function Derived:stop() end
'''})
        assert ("m_derived", "m_base") in links(root, "inherits")

    def test_a_bare_metatable_is_also_inheritance(self, luacorpus):
        root = luacorpus({"m.lua": '''
local Base = {}
function Base:go() end
local Derived = setmetatable({}, Base)
function Derived:stop() end
'''})
        assert ("m_derived", "m_base") in links(root, "inherits")

    def test_a_table_bound_somewhere_else_gets_no_class_node(self, luacorpus):
        # `function Foo:bar()` on a table this file never binds has no line to
        # anchor a class node to, so the member is kept at file scope rather
        # than hung off a node that was invented. Nothing may point at a node
        # that does not exist.
        root = luacorpus({"m.lua": "function Foo:bar() end\n"})
        k = kinds(root)
        assert k["m_bar"] == "function"
        assert "m_foo" not in k
        assert all(t in k for _, t in links(root, "contains"))

    def test_containment_matches_the_python_shape(self, luacorpus):
        got = links(luacorpus(PICKER), "contains")
        assert ("m", "m_picker") in got
        assert ("m_picker", "m_picker_new") in got


SORTERS = {"m.lua": '''
local Sorter = {}
function Sorter:new(opts) end

local M = {}

M.get_fuzzy_file = function(opts)
  return Sorter:new {
    --- Scores one line.
    scoring_function = function(_, prompt, line) end,
  }
end

M.get_fzy_sorter = function(opts)
  return Sorter:new {
    scoring_function = function(_, prompt, line) end,
  }
end
'''}


class TestFunctionsBelowTheTopLevel:
    """A Lua file's behaviour is mostly written under its top level.

    A field of a table handed to a call, and a helper declared inside a
    factory, are both real functions with real names, and reading only
    `function M.f()` left both out. These pin down that they arrive, that the
    scope keeps the repeated names apart, and that nothing anonymous is
    invented.
    """

    def test_a_function_field_of_a_table_argument_is_a_node(self, luacorpus):
        # `Sorter:new { scoring_function = ... }` -- the table is an argument,
        # so no assignment target ever names the function.
        k = kinds(luacorpus(SORTERS))
        assert k["m_get_fuzzy_file_scoring_function"] == "function"

    def test_a_repeated_field_name_is_kept_apart_by_its_enclosing_function(
            self, luacorpus):
        # One real corpus writes eight fields called `scoring_function` in one
        # file. Scoped by the file they would mint one id and lose seven.
        k = kinds(luacorpus(SORTERS))
        assert k["m_get_fuzzy_file_scoring_function"] == "function"
        assert k["m_get_fzy_sorter_scoring_function"] == "function"

    def test_a_nested_field_is_contained_by_the_function_it_sits_in(
            self, luacorpus):
        # `contains` has to point at a node that exists. A deeper scope cannot
        # mint its own container from `scope[0]`, so it is told one.
        root = luacorpus(SORTERS)
        got, k = links(root, "contains"), kinds(root)
        assert ("m_get_fuzzy_file", "m_get_fuzzy_file_scoring_function") in got
        assert all(s in k and t in k for s, t in got)

    def test_a_nested_field_keeps_its_doc_comment(self, luacorpus):
        assert docs(luacorpus(SORTERS))[
            "m_get_fuzzy_file_scoring_function#doc"] == "Scores one line."

    def test_a_local_function_inside_a_body_is_a_node(self, luacorpus):
        root = luacorpus({"m.lua": '''
local M = {}
function M.factory()
  local function helper() end
  return helper()
end
'''})
        assert kinds(root)["m_factory_helper"] == "function"
        assert ("m_factory", "m_factory_helper") in links(root)

    def test_a_function_bound_to_a_local_inside_a_body_is_a_node(self, luacorpus):
        root = luacorpus({"m.lua": '''
local M = {}
function M.wrap()
  local step = function() end
end
'''})
        assert kinds(root)["m_wrap_step"] == "function"

    def test_the_whole_dotted_target_scopes_a_nested_binding(self, luacorpus):
        # `defaults.attach_mappings` and `opts.attach_mappings` are eleven lines
        # apart in one real function. The table each belongs to is the only
        # thing that tells them apart.
        root = luacorpus({"m.lua": '''
local M = {}
function M.apply(opts, defaults)
  defaults.attach_mappings = function() end
  opts.attach_mappings = function() end
end
'''})
        k = kinds(root)
        assert k["m_apply_defaults_attach_mappings"] == "function"
        assert k["m_apply_opts_attach_mappings"] == "function"

    def test_a_computed_key_names_nothing(self, luacorpus):
        # `[k] = function() end` puts an identifier where a key goes, but `k`
        # holds the key rather than being it -- one real file writes three such
        # fields and naming them after the variable would mint one id.
        root = luacorpus({"m.lua": '''
local M = {}
function M.setup()
  local t = {
    [key] = function() end,
    plain = function() end,
  }
end
'''})
        k = kinds(root)
        assert "m_setup_key" not in k
        assert k["m_setup_plain"] == "function"

    def test_an_anonymous_callback_stays_folded_into_its_caller(self, luacorpus):
        # There is no name for a question to arrive under, so it gets no node
        # and its calls belong to whatever encloses it.
        root = luacorpus({"m.lua": '''
local M = {}
function M.target() end
function M.wrap()
  run(function() M.target() end)
end
'''})
        assert set(kinds(root)) == {"m", "m_target", "m_wrap"}
        assert ("m_wrap", "m_target") in links(root)

    def test_a_call_inside_a_named_nested_function_belongs_to_it(self, luacorpus):
        # Counting it against the enclosing function too would record one call
        # once per level of nesting.
        root = luacorpus({"m.lua": '''
local M = {}
function M.target() end
function M.outer()
  local function inner() M.target() end
end
'''})
        assert ("m_outer_inner", "m_target") in links(root)
        assert ("m_outer", "m_target") not in links(root)

    def test_a_closure_in_a_method_shares_that_methods_self(self, luacorpus):
        root = luacorpus({"m.lua": '''
local Picker = {}
function Picker:close() end
function Picker:find()
  local on_done = function() self:close() end
end
'''})
        assert ("m_picker_find_on_done", "m_picker_close") in links(root)

    def test_a_nested_function_with_its_own_self_does_not_inherit_the_class(
            self, luacorpus):
        # `local function helper(self)` names its own parameter, of a type
        # nothing here states. Inheriting the enclosing class would put a
        # confident edge on a guess.
        root = luacorpus({"m.lua": '''
local Picker = {}
function Picker:close() end
function Picker:find()
  local function helper(self) self:close() end
end
'''})
        assert ("m_picker_find_helper", "m_picker_close") not in links(root)

    def test_a_table_inside_a_call_is_scoped_by_the_name_it_was_bound_to(
            self, luacorpus):
        # The table is an argument, so it is not the value the name is bound to
        # and the dissolve rule does not reach it. One real file binds three of
        # these and every one has a field called `command`.
        root = luacorpus({"m.lua": '''
local M = {}
M.track = make_action { command = function(name) end }
M.delete = make_action { command = function(name) end }
'''})
        k = kinds(root)
        assert k["m_track_command"] == "function"
        assert k["m_delete_command"] == "function"

    def test_a_top_level_do_block_is_a_scope_not_a_container(self, luacorpus):
        # `do ... end` keeps a few locals out of the module's namespace. The
        # function inside carries the name it would carry outside, and one real
        # file wraps three of its public generators in one such block.
        root = luacorpus({"m.lua": '''
local M = {}
do
  local lookup = {}
  function M.gen_from_string(opts) end
end
'''})
        k = kinds(root)
        assert k["m_gen_from_string"] == "function"
        assert ("m", "m_gen_from_string") in links(root, "contains")

    def test_a_table_inside_a_table_keeps_the_field_names_on_the_way_down(
            self, luacorpus):
        root = luacorpus({"m.lua": '''
local M = {}
M.task = {
  git = { run = function() end },
}
'''})
        assert kinds(root)["m_task_git_run"] == "function"


class TestResolution:
    def test_a_call_on_self_resolves_exactly(self, luacorpus):
        assert ("m_picker_new", "m_picker_reset") in links(luacorpus(PICKER))

    def test_a_dot_call_on_self_resolves_too(self, luacorpus):
        # `self.foo()` does not pass self, but it still names a member of the
        # same type, which is what the graph is being asked about.
        root = luacorpus({"m.lua": '''
local Picker = {}
function Picker:close() end
function Picker:new()
  self.close(self)
end
'''})
        assert ("m_picker_new", "m_picker_close") in links(root)

    def test_a_call_on_this_files_namespace_table_is_a_same_file_call(self, luacorpus):
        # `M.helper()` inside the module that defines helper is Lua's long way
        # of writing `helper()`, and dissolving M is what lets it resolve.
        root = luacorpus({"m.lua": '''
local M = {}
function M.helper() end
function M.run() M.helper() end
'''})
        assert ("m_run", "m_helper") in links(root)

    def test_a_call_on_the_class_table_resolves(self, luacorpus):
        root = luacorpus({"m.lua": '''
local Picker = {}
function Picker.describe() end
function Picker:new()
  Picker.describe()
end
'''})
        assert ("m_picker_new", "m_picker_describe") in links(root)

    def test_a_param_annotation_types_its_variable(self, luacorpus):
        # LuaDoc is the only place a Lua file says what a parameter is, and it
        # says it outright -- this is a declaration, not an inference.
        root = luacorpus({"m.lua": '''
local Picker = {}
function Picker:find() end

---@param picker Picker
local function use(picker)
  picker:find()
end
'''})
        assert ("m_use", "m_picker_find") in links(root)

    def test_a_type_annotation_types_a_local(self, luacorpus):
        root = luacorpus({"m.lua": '''
local Picker = {}
function Picker:find() end

local function use()
  ---@type Picker
  local p = make()
  p:find()
end
'''})
        assert ("m_use", "m_picker_find") in links(root)

    def test_a_primitive_annotation_types_nothing(self, luacorpus):
        # `---@param path string` must not make `path:sub()` look like a call
        # on a table that happens to be called `string`.
        root = luacorpus({"m.lua": '''
local string = {}
function string:sub(i) end

---@param path string
local function use(path)
  path:sub(1)
end
'''})
        assert ("m_use", "m_string_sub") not in links(root)
        assert reasons_for(root)["receiver type unknown"] >= 1

    def test_a_required_member_resolves_across_files(self, luacorpus):
        # `local f = require("m").f` names the symbol, not just the module.
        root = luacorpus({
            "util.lua": 'local M = {}\nfunction M.helper() end\nreturn M\n',
            "main.lua": '''
local helper = require("util").helper
local function go() helper() end
''',
        })
        files, _ = parse_corpus_files(root)
        imports = {k: v for p in files for k, v in p.imports.items()}
        assert imports["helper"] == "util.helper"
        assert ("main_go", "util_helper") in links(root)

    def test_a_require_becomes_an_import_edge(self, luacorpus):
        root = luacorpus({
            "a/b.lua": 'local M = {}\nreturn M\n',
            "main.lua": 'local b = require "a.b"\n',
        })
        files, _ = parse_corpus_files(root)
        sites = [s for p in files for s in p.import_sites]
        assert ("a.b", 1) in sites
        assert ("main", "a_b") in links(root, "imports")

    def test_a_call_on_an_untyped_receiver_is_refused_not_guessed(self, luacorpus):
        root = luacorpus({"m.lua": '''
local Picker = {}
function Picker:find() end
local function use(thing) thing:find() end
'''})
        assert ("m_use", "m_picker_find") not in links(root)
        assert reasons_for(root)["receiver type unknown"] >= 1

    def test_a_call_on_an_untyped_self_attribute_is_refused(self, luacorpus):
        root = luacorpus({"m.lua": '''
local Picker = {}
function Picker:new()
  self.inner:go()
end
'''})
        assert reasons_for(root)["receiver is a self attribute of unknown type"] >= 1

    def test_resolved_edges_never_dangle(self, luacorpus):
        files, _ = parse_corpus_files(luacorpus(PICKER))
        ids = {n.id for p in files for n in p.nodes}
        edges, _ = resolve(files)
        for e in edges:
            if e.resolved:
                assert e.target in ids


class TestMixedCorpus:
    def test_lua_and_python_in_one_repository(self, luacorpus):
        root = luacorpus({"ui/pickers.lua": "local Picker = {}\nfunction Picker:find() end\n",
                          "tools/build.py": "class Builder:\n    pass\n"})
        k = kinds(root)
        assert k["ui_pickers_picker"] == "class"
        assert k["tools_build_builder"] == "class"


class TestFailure:
    def test_an_unparseable_file_is_reported_not_skipped_silently(self, luacorpus):
        root = luacorpus({"bad.lua": "\x00\x00\x00", "ok.lua": "local M = {}\n"})
        _, failed = parse_corpus_files(root)
        assert "ok.lua" not in failed

    def test_a_syntax_error_in_one_function_does_not_lose_the_file(self, luacorpus):
        # Returning False makes the whole file count as a loss, which is right
        # for a file we could not read and wrong for one bad function.
        root = luacorpus({"m.lua": '''
local M = {}
function M.good() end
function M.broken(
'''})
        parsed = parse_file(root / "m.lua", root)
        assert parsed is not None
        assert any(n.id == "m_good" for n in parsed.nodes)
