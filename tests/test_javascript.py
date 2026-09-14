"""JavaScript, read through tree-sitter.

Same test as Go and TypeScript: does another language produce the SAME shapes,
without anything downstream learning it exists?

JavaScript stresses two things neither of those did. It has **two module
systems** -- `import` and `require` -- living in the same language and often in
the same file, and it has **no type information at all**, so nearly every
receiver has to be refused rather than resolved. The refusal tests below are
the important half: a wrong edge is worse than a missing one.
"""
import pytest

from graphpaat.parse import parse_corpus_files
from graphpaat.resolve import resolve

pytest.importorskip("tree_sitter_javascript")


@pytest.fixture
def js(tmp_path):
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


def ids(root):
    """Every id as a list, so a duplicate is visible rather than deduplicated."""
    files, _ = parse_corpus_files(root)
    return [n.id for p in files for n in p.nodes]


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


def docs_for(root):
    files, _ = parse_corpus_files(root)
    return {n.id: n.text for p in files for n in p.nodes if n.kind == "rationale"}


SERVER = {"srv.js": '''import Base from './base.js';

/**
 * A server that accepts connections.
 */
export class Server extends Base {
  constructor(addr) { this.addr = addr; }

  /** Begin serving. */
  start() { this.listen(); }

  listen() {}
}

export function helper(x) { return x; }

export const make = (addr) => new Server(addr);
''', "base.js": "export default class Base {}\n"}


class TestSameShapesAsPython:
    def test_a_class_is_a_class_node(self, js):
        assert kinds(js(SERVER))["srv_server"] == "class"

    def test_methods_are_qualified_by_their_class(self, js):
        k = kinds(js(SERVER))
        assert k["srv_server_start"] == "method"
        assert k["srv_helper"] == "function"

    def test_a_const_arrow_is_a_function(self, js):
        # Most functions in modern JavaScript are written this way; reading
        # only `function` declarations would leave axios's utils.js with one
        # node instead of forty-two.
        assert kinds(js(SERVER))["srv_make"] == "function"

    def test_a_const_function_expression_is_a_function(self, js):
        root = js({"m.js": "const run = function () { return 1; };\n"})
        assert kinds(root)["m_run"] == "function"

    def test_a_generator_declaration_is_a_function(self, js):
        root = js({"m.js": "function* walk() { yield 1; }\n"})
        assert kinds(root)["m_walk"] == "function"

    def test_a_class_expression_takes_the_name_it_is_bound_to(self, js):
        # `const CancelToken = class CancelToken {}` is imported and called by
        # the binding's name everywhere; the inner name is visible only inside
        # its own body.
        root = js({"m.js": "const Outer = class Inner { go() {} };\n"})
        k = kinds(root)
        assert k["m_outer"] == "class" and k["m_outer_go"] == "method"
        assert "m_inner" not in k

    def test_export_does_not_hide_a_declaration(self, js):
        root = js({"m.js": "export function a() {}\nexport default class B {}\n"})
        k = kinds(root)
        assert k["m_a"] == "function" and k["m_b"] == "class"

    def test_a_class_field_holding_an_arrow_is_a_method(self, js):
        root = js({"m.js": '''class C {
  onClick = () => { this.go(); };
  go() {}
}
'''})
        assert kinds(root)["m_c_onclick"] == "method"
        assert ("m_c_onclick", "m_c_go") in links(root)

    def test_a_getter_and_setter_pair_mint_one_id_not_two(self, js):
        # Two `method_definition`s share one name. Emitting both makes an id
        # that two symbols claim, and the second is lost silently.
        root = js({"m.js": "class C { get v() { return 1; } set v(x) {} }\n"})
        assert ids(root).count("m_c_v") == 1

    def test_a_private_method_keeps_its_name_without_the_hash(self, js):
        # `#` is reserved for the docstring suffix, so it must not appear in an
        # ordinary id.
        root = js({"m.js": "class C { #hide() {} }\n"})
        assert kinds(root)["m_c_hide"] == "method"

    def test_two_classes_with_a_same_named_method_do_not_collide(self, js):
        root = js({"m.js": "class A { run() {} }\nclass B { run() {} }\n"})
        k = kinds(root)
        assert "m_a_run" in k and "m_b_run" in k

    def test_a_nested_function_is_not_a_top_level_one(self, js):
        # A binding inside a function body is not a module-level symbol.
        # Emitting it unscoped mints an id that collides the moment two
        # functions each keep a local `handler`.
        #
        # It IS emitted -- see TestNestedFunctions below -- but never at file
        # scope. Both halves matter: present, and qualified.
        root = js({"m.js": "function outer() { const inner = () => 1; return inner; }\n"})
        k = kinds(root)
        assert "m_inner" not in k
        assert k["m_outer_inner"] == "function"

    def test_a_jsdoc_block_becomes_a_rationale_node(self, js):
        docs = docs_for(js(SERVER))
        assert "A server that accepts connections." in docs["srv_server#doc"]
        assert "Begin serving." in docs["srv_server_start#doc"]

    def test_a_jsdoc_block_one_blank_line_above_still_documents_it(self, js):
        # express writes 140 of its 143 JSDoc blocks with a blank line between
        # the comment and the declaration. Requiring adjacency would throw away
        # that repository's entire "why" lane.
        root = js({"m.js": '''/**
 * Set the status code.
 */

exports.status = function status(code) { return code; };
'''})
        assert "Set the status code." in docs_for(root)["m_status#doc"]

    def test_a_jsdoc_block_two_blank_lines_above_is_not_documentation(self, js):
        root = js({"m.js": '''/**
 * A note about the file.
 */


function unrelated() {}
'''})
        assert docs_for(root) == {}

    def test_a_line_comment_is_never_documentation(self, js):
        # A `//` note is usually about a line, not a declaration. Taking them
        # all buries the real documentation under eslint pragmas and banners.
        root = js({"m.js": "// not a doc comment\nfunction thing() {}\n"})
        assert docs_for(root) == {}

    def test_extends_is_inheritance(self, js):
        assert ("srv_server", "base_base") in links(js(SERVER), "inherits")

    def test_a_namespaced_base_keeps_its_whole_name(self, js):
        # `extends React.Component` must not be recorded as `Component`, which
        # would match any corpus class of that name. It stays unresolved and
        # says what the class actually extends.
        root = js({"app.jsx": '''import React from 'react';
export default class App extends React.Component { render() { return 1; } }
''', "component.js": "export class Component {}\n"})
        files, _ = parse_corpus_files(root)
        bases = {n.id: n.bases for p in files for n in p.nodes if n.kind == "class"}
        assert bases["app_app"] == ["React.Component"]
        assert ("app_app", "component_component") not in links(root, "inherits")

    def test_jsx_is_read_by_the_same_grammar(self, js):
        root = js({"app.jsx": '''class App {
  render() { return <div onClick={this.go}>hi</div>; }
  go() {}
}
'''})
        assert kinds(root)["app_app_render"] == "method"

    def test_containment_matches_the_python_shape(self, js):
        got = links(js(SERVER), "contains")
        assert ("srv", "srv_server") in got
        assert ("srv_server", "srv_server_start") in got


class TestNestedFunctions:
    """Functions declared inside other functions, which is most of a real
    adapter file.

    axios settles a request in `done()` declared inside the XHR adapter and
    measures a body in `getBodyLength`, a const inside the fetch adapter.
    Neither name appears at the file's top level. Reading only the top level
    left symbols like these out of the map entirely -- measured on
    two corpora, that one shape was most of the gap between what the files
    declare and what the graph held.
    """

    def test_a_nested_function_declaration_is_emitted(self, js):
        root = js({"m.js": '''function outer() {
  function done() { return 1; }
  return done;
}
'''})
        assert kinds(root)["m_outer_done"] == "function"

    def test_a_nested_const_arrow_is_emitted(self, js):
        # `const getBodyLength = async (body) => ...` inside another function is
        # how axios writes most of its helpers.
        root = js({"m.js": '''export default (config) => {
  const getBodyLength = async (body) => body.size;
  return getBodyLength;
};
'''})
        assert kinds(root)["m_getbodylength"] == "function"

    def test_two_same_named_helpers_in_one_file_do_not_collide(self, js):
        # The reason ids carry a scope chain at all. Unqualified, both `done`s
        # mint one id and the second symbol is destroyed with no error.
        root = js({"m.js": '''function first() {
  function done() { return 1; }
  return done;
}
function second() {
  function done() { return 2; }
  return done;
}
'''})
        got = ids(root)
        assert len(got) == len(set(got))
        k = kinds(root)
        assert k["m_first_done"] == "function" and k["m_second_done"] == "function"

    def test_the_chain_runs_through_a_class_method(self, js):
        # Three deep: class, method, helper. Qualifying by the first name in the
        # chain instead of the last would hang the helper off the class and
        # skip the method that actually holds it.
        root = js({"m.js": '''class C {
  go() {
    function helper() {}
    return helper;
  }
}
'''})
        assert kinds(root)["m_c_go_helper"] == "function"
        assert ("m_c_go", "m_c_go_helper") in links(root, "contains")

    def test_an_anonymous_callback_adds_no_scope(self, js):
        # A callback has no name to be called by, so it gets no node and the
        # walk passes straight through it. `helper` belongs to `outer`.
        root = js({"m.js": '''function outer() {
  run(() => {
    function helper() {}
    helper();
  });
}
'''})
        assert kinds(root)["m_outer_helper"] == "function"

    def test_a_named_function_expression_is_a_definition(self, js):
        # `new Promise(function dispatchXhrRequest(...) {...})` -- authors name
        # these for the stack trace, and axios's entire XHR adapter lives in
        # one. Without a node for it the helpers inside would hang off the file
        # with nothing between them and it.
        root = js({"m.js": '''function outer() {
  return make(function dispatch(a) {
    function deep() {}
    return deep;
  });
}
'''})
        k = kinds(root)
        assert k["m_outer_dispatch"] == "function"
        assert k["m_outer_dispatch_deep"] == "function"

    def test_an_export_default_expression_is_read(self, js):
        # `export default isSupported && function (config) {...}` is a whole
        # axios adapter. Unwrapping finds no declaration in it, so before this
        # the file arrived with a file node and nothing else.
        root = js({"m.js": '''const supported = true;
export default supported && function (config) {
  function done() {}
  return done;
};
'''})
        assert kinds(root)["m_done"] == "function"

    def test_a_call_belongs_to_the_innermost_function_around_it(self, js):
        # Same rule the Python reader uses. Leaving the call on the outer
        # function would say the outer one calls things it never runs.
        root = js({"m.js": '''function target() {}
function outer() {
  function inner() { target(); }
  return inner;
}
'''})
        got = links(root)
        assert ("m_outer_inner", "m_target") in got
        assert ("m_outer", "m_target") not in got

    def test_a_class_declared_inside_a_function_is_scoped_too(self, js):
        # A class built by a factory is still a class. Two factories in one
        # file would otherwise mint the same ids for two different classes.
        root = js({"m.js": '''function make() {
  class Inner { go() {} }
  return Inner;
}
'''})
        k = kinds(root)
        assert k["m_make_inner"] == "class" and k["m_make_inner_go"] == "method"

    def test_a_nested_generator_is_a_function(self, js):
        root = js({"m.js": '''function outer() {
  function* walk() { yield 1; }
  return walk;
}
'''})
        assert kinds(root)["m_outer_walk"] == "function"

    def test_a_jsdoc_block_above_a_nested_const_documents_it(self, js):
        # The comment sits above the whole `const ...` statement, not above the
        # binding inside it, so the doc has to be looked for on the statement.
        root = js({"m.js": '''function outer() {
  /**
   * Length of the body.
   */
  const getBodyLength = () => 0;
  return getBodyLength;
}
'''})
        assert "Length of the body." in docs_for(root)["m_outer_getbodylength#doc"]

    def test_nested_containment_edges_never_dangle(self, js):
        # More symbols only help if every containment edge still points at a
        # node that exists.
        root = js({"m.js": '''class C {
  go() {
    const codes = { ok() {} };
    function helper() { return codes; }
    return helper;
  }
}
'''})
        files, _ = parse_corpus_files(root)
        node_ids = {n.id for p in files for n in p.nodes}
        for p in files:
            for edge in p.edges:
                if edge.relation == "contains":
                    assert edge.source in node_ids


class TestObjectLiterals:
    """`{ foo() {} }` and `{ foo: function () {} }` -- how JavaScript wrote a
    namespace before it had modules."""

    def test_both_spellings_of_an_object_method_are_functions(self, js):
        # Emitted as functions, not methods: nothing here is an instance of a
        # class, and calling them methods would send the resolver looking for
        # an owning class that does not exist.
        root = js({"m.js": '''const codes = {
  ok() { return 1; },
  fail: function () { return 0; },
  limit: 5,
};
'''})
        k = kinds(root)
        assert k["m_codes_ok"] == "function" and k["m_codes_fail"] == "function"
        assert "m_codes_limit" not in k        # a constant is not a function

    def test_a_top_level_object_hangs_its_members_off_the_file(self, js):
        # The object itself gets no node, so the file has to be the container
        # or the containment edge would point at an id nobody minted.
        root = js({"m.js": "const codes = { ok() { return 1; } };\n"})
        assert ("m", "m_codes_ok") in links(root, "contains")

    def test_two_objects_with_a_same_named_member_do_not_collide(self, js):
        root = js({"m.js": '''const a = { run() {} };
const b = { run() {} };
'''})
        got = ids(root)
        assert len(got) == len(set(got))
        assert "m_a_run" in got and "m_b_run" in got

    def test_an_anonymous_object_gets_no_nodes(self, js):
        # An options bag passed straight to a call has no name to qualify its
        # members by. Two of them in one function would mint one id per shared
        # key, and losing one is worse than never emitting either.
        root = js({"m.js": "function outer() { register({ onLoad() {} }); }\n"})
        assert "m_outer_onload" not in kinds(root)

    def test_an_object_inside_a_function_carries_the_whole_chain(self, js):
        root = js({"m.js": '''function outer() {
  const codes = { ok() {} };
  return codes;
}
'''})
        assert kinds(root)["m_outer_codes_ok"] == "function"
        assert ("m_outer", "m_outer_codes_ok") in links(root, "contains")

    def test_a_call_in_a_non_function_member_stays_with_its_writer(self, js):
        # `{ limit: compute() }` is code the enclosing function runs, not
        # something the object does.
        root = js({"m.js": '''function compute() { return 1; }
function outer() {
  const opts = { limit: compute(), ok() {} };
  return opts;
}
'''})
        assert ("m_outer", "m_compute") in links(root)


class TestCommonJS:
    """The older module system, which is most of npm and all of express."""

    def test_exports_assignment_is_a_function(self, js):
        root = js({"u.js": "exports.helper = function helper(x) { return x; };\n"})
        assert kinds(root)["u_helper"] == "function"

    def test_module_exports_takes_the_name_of_the_function_it_exports(self, js):
        root = js({"v.js": "module.exports = function View(name) { return name; };\n"})
        assert kinds(root)["v_view"] == "function"

    def test_a_prototype_assignment_is_a_method_of_the_constructor(self, js):
        # `X.prototype.f = function () {}` is what a method was before `class`
        # existed, and express's `lib/` is written entirely this way.
        root = js({"v.js": '''function View(name) { this.name = name; }
View.prototype.render = function render() { this.lookup(); };
View.prototype.lookup = function lookup() {};
'''})
        k = kinds(root)
        assert k["v_view"] == "function"
        assert k["v_view_render"] == "method" and k["v_view_lookup"] == "method"
        assert ("v_view", "v_view_render") in links(root, "contains")
        assert ("v_view_render", "v_view_lookup") in links(root)

    def test_a_property_on_a_class_is_a_static_method_of_it(self, js):
        root = js({"m.js": "class E {}\nE.from = function from(e) { return e; };\n"})
        assert kinds(root)["m_e_from"] == "method"

    def test_a_property_on_an_undeclared_object_is_a_plain_function(self, js):
        # `res` is created by `Object.create(...)`, so there is no class node
        # to hang `status` off. Scoping it by `res` anyway would point the
        # containment edge at an id nobody owns.
        root = js({"r.js": '''var res = Object.create(null);
res.status = function status(code) { return code; };
'''})
        k = kinds(root)
        assert k["r_status"] == "function"
        assert ("r", "r_status") in links(root, "contains")

    def test_an_assigned_constant_is_not_a_function_node(self, js):
        root = js({"m.js": "class E {}\nE.CODE = 'ERR_BAD_REQUEST';\n"})
        assert "m_e_code" not in kinds(root)

    def test_require_is_recorded_as_an_import(self, js):
        root = js({"a.js": "var utils = require('./lib/utils.js');\n",
                   "lib/utils.js": "exports.trim = function trim(s) { return s; };\n"})
        assert ("a", "lib_utils") in links(root, "imports")

    def test_a_required_symbol_resolves_a_bare_call(self, js):
        root = js({"a.js": '''var trim = require('./p.js').trim;
function use(s) { return trim(s); }
''', "p.js": "exports.trim = function trim(s) { return s; };\n"})
        assert ("a_use", "p_trim") in links(root)

    def test_a_destructured_require_binds_each_name(self, js):
        root = js({"a.js": '''const { trim } = require('./p.js');
function use(s) { return trim(s); }
''', "p.js": "exports.trim = function trim(s) { return s; };\n"})
        assert ("a_use", "p_trim") in links(root)

    def test_a_require_inside_a_function_still_counts_as_a_dependency(self, js):
        # `require` is a call, so it can be lazy. Reading only the top level
        # makes a conditionally-loaded module look like no dependency at all.
        root = js({"a.js": "function load() { return require('./p.js'); }\n",
                   "p.js": "exports.trim = function trim(s) { return s; };\n"})
        assert ("a", "p") in links(root, "imports")


class TestResolution:
    def test_a_relative_import_is_anchored_to_the_importing_file(self, js):
        # `../utils.js` means a different file from every directory. Stripping
        # the leading dots and stopping there resolves it from none of them.
        root = js({"core/a.js": "import trim from '../utils.js';\n",
                   "utils.js": "export function trim(s) { return s; }\n"})
        assert ("core_a", "utils") in links(root, "imports")

    def test_an_imported_symbol_resolves_a_bare_call(self, js):
        root = js({"a.js": '''import { trim } from './p.js';
export function use(s) { return trim(s); }
''', "p.js": "export function trim(s) { return s; }\n"})
        assert ("a_use", "p_trim") in links(root)

    def test_a_re_export_is_an_import(self, js):
        # A barrel file is nothing but these; reading only `import` makes it
        # look like it depends on nothing.
        root = js({"index.js": "export * from './p.js';\n",
                   "p.js": "export function trim(s) { return s; }\n"})
        assert ("index", "p") in links(root, "imports")

    def test_a_default_exported_string_is_not_a_dependency(self, js):
        # `export default 'x'` has a string child exactly like a re-export
        # does; taking it as a module specifier invented a dependency on the
        # value of the string.
        root = js({"m.js": "export default 'a literal string';\n"})
        files, _ = parse_corpus_files(root)
        assert files[0].import_sites == []

    def test_an_import_alias_records_the_symbol_it_actually_names(self, js):
        root = js({"a.js": "import { trim as t } from './p.js';\n",
                   "p.js": "export function trim(s) { return s; }\n"})
        files, _ = parse_corpus_files(root)
        got = {p.prefix: p.imports for p in files}
        assert got["a"]["t"] == "p.trim"

    def test_a_plain_call_in_the_same_file(self, js):
        root = js({"m.js": "function helper() {}\nfunction go() { helper(); }\n"})
        assert ("m_go", "m_helper") in links(root)

    def test_a_constructed_variable_types_its_receiver(self, js):
        # `const s = new Server()` is the ONLY place JavaScript states a type.
        root = js({"m.js": '''class Server { ping() {} }
function use() { const s = new Server(); s.ping(); }
'''})
        assert ("m_use", "m_server_ping") in links(root)

    def test_a_call_on_this_resolves_to_a_sibling_method(self, js):
        assert ("srv_server_start", "srv_server_listen") in links(js(SERVER))

    def test_a_constructed_attribute_types_a_this_receiver(self, js):
        root = js({"m.js": '''class Server { ping() {} }
class Client {
  constructor() { this.server = new Server(); }
  send() { this.server.ping(); }
}
'''})
        assert ("m_client_send", "m_server_ping") in links(root)

    def test_new_is_a_call_to_the_class(self, js):
        assert ("srv_make", "srv_server") in links(js(SERVER))

    def test_a_call_on_an_untyped_receiver_is_refused_not_guessed(self, js):
        root = js({"m.js": '''class Server { ping() {} }
function use(thing) { thing.ping(); }
'''})
        assert ("m_use", "m_server_ping") not in links(root)
        assert reasons_for(root)["receiver type unknown"] >= 1

    def test_a_call_on_an_expression_is_refused_not_treated_as_a_bare_call(self, js):
        # `/re/.test(x)` is a call ON something. Letting it fall through to a
        # bare `test()` linked it to an unrelated helper of that name: measured
        # on axios, 9 of 27 corpus-unique links were wrong exactly this way.
        root = js({"a.js": "export function test(x) { return x; }\n",
                   "b.js": "export function isAbs(u) { return /^https?:/.test(u); }\n"})
        assert ("b_isabs", "a_test") not in links(root)
        assert reasons_for(root)["receiver type unknown"] >= 1

    def test_a_package_import_stays_outside_the_corpus(self, js):
        # Correct and useful output, not a failure: which third-party modules a
        # file pulls in is exactly what an agent asks.
        root = js({"a.js": "import pad from 'left-pad';\n"})
        assert links(root, "imports") == set()
        assert reasons_for(root)["imports a module outside this corpus"] == 1

    def test_resolved_edges_never_dangle(self, js):
        files, _ = parse_corpus_files(js(SERVER))
        node_ids = {n.id for p in files for n in p.nodes}
        edges, _ = resolve(files)
        for edge in edges:
            if edge.resolved:
                assert edge.target in node_ids


class TestMixedCorpus:
    def test_javascript_and_python_in_one_repository(self, js):
        root = js({"api/srv.js": "export class Server {}\n",
                   "tools/build.py": "class Builder:\n    pass\n"})
        k = kinds(root)
        assert k["api_srv_server"] == "class"
        assert k["tools_build_builder"] == "class"


class TestFailure:
    def test_an_unparseable_file_is_reported_not_skipped_silently(self, js):
        root = js({"bad.js": "class \x00\x00 {{{{ ((((", "ok.js": "function f() {}\n"})
        _, failed = parse_corpus_files(root)
        # One bad file must never take the rest of the corpus down with it.
        assert "ok.js" not in failed
        assert kinds(root)["ok_f"] == "function"
