"""Swift, read through tree-sitter.

The point of another language is not that it works but that it produces the
SAME shapes: a struct is a `class` node, a conformance is `inherits`, a `///`
run is a `rationale`. If any of that needed a new word, the model was never
language-neutral and grouping, ranking and query would have to learn Swift too.

Swift's own two habits are tested here as well. A type can be extended from
anywhere, so a method often belongs to a type declared in another file; and
`self.` is optional, so the commonest call in the language is written bare.
"""
import pytest

from graphpaat.parse import parse_corpus_files, parse_file
from graphpaat.resolve import resolve

pytest.importorskip("tree_sitter_swift")


@pytest.fixture
def swiftcorpus(tmp_path):
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


SESSION = {"srv.swift": '''import Foundation

/// Accepts connections.
public class Session: Sendable {
    /// The queue everything runs on.
    let queue: Queue

    /// Begins serving.
    public func start() {
        self.listen()
    }

    func listen() {}
}

class Queue {
    func drain() {}
}

func make() -> Session { Session(queue: Queue()) }
'''}


class TestSameShapesAsPython:
    def test_a_class_is_a_class_node(self, swiftcorpus):
        assert kinds(swiftcorpus(SESSION))["srv_session"] == "class"

    def test_a_struct_is_also_a_class_node(self, swiftcorpus):
        root = swiftcorpus({"m.swift": "struct Point { let x: Int }\n"})
        assert kinds(root)["m_point"] == "class"

    def test_an_enum_is_also_a_class_node(self, swiftcorpus):
        root = swiftcorpus({"m.swift": "enum Color { case red }\n"})
        assert kinds(root)["m_color"] == "class"

    def test_an_actor_is_also_a_class_node(self, swiftcorpus):
        # `actor` arrives under the same node type as `class` and `struct`,
        # told apart only by the keyword.
        root = swiftcorpus({"m.swift": "actor Counter { var n = 0 }\n"})
        assert kinds(root)["m_counter"] == "class"

    def test_a_protocol_is_also_a_class_node(self, swiftcorpus):
        root = swiftcorpus({"m.swift": "protocol Handler { func run() }\n"})
        assert kinds(root)["m_handler"] == "class"

    def test_a_typealias_is_also_a_class_node(self, swiftcorpus):
        root = swiftcorpus({"m.swift": "typealias Done = (Int) -> Void\n"})
        assert kinds(root)["m_done"] == "class"

    def test_methods_are_qualified_by_their_type(self, swiftcorpus):
        k = kinds(swiftcorpus(SESSION))
        assert k["srv_session_start"] == "method"
        assert k["srv_make"] == "function"

    def test_two_types_with_a_same_named_method_do_not_collide(self, swiftcorpus):
        root = swiftcorpus({"m.swift": '''struct A { func run() {} }
struct B { func run() {} }
'''})
        k = kinds(root)
        assert "m_a_run" in k and "m_b_run" in k

    def test_containment_matches_the_python_shape(self, swiftcorpus):
        got = links(swiftcorpus(SESSION), "contains")
        assert ("srv", "srv_session") in got
        assert ("srv_session", "srv_session_start") in got

    def test_an_init_is_a_method_called_init(self, swiftcorpus):
        root = swiftcorpus({"m.swift": "struct P { init(x: Int) {} }\n"})
        assert kinds(root)["m_p_init"] == "method"

    def test_deinit_and_subscript_are_methods_too(self, swiftcorpus):
        root = swiftcorpus({"m.swift": '''class P {
    deinit {}
    subscript(i: Int) -> Int { 0 }
}
'''})
        k = kinds(root)
        assert k["m_p_deinit"] == "method"
        assert k["m_p_subscript"] == "method"

    def test_an_operator_declaration_keeps_its_symbol_as_the_name(self, swiftcorpus):
        # `==` is real API in Swift and a symbol that is invisible is worse
        # than one with an awkward name.
        root = swiftcorpus({"m.swift":
                            "struct P { static func == (a: P, b: P) -> Bool { true } }\n"})
        assert kinds(root)["m_p_=="] == "method"

    def test_an_overload_is_one_node_not_many(self, swiftcorpus):
        # Swift overloads by argument label, so a type declares `request`
        # several times. They are one entry point and mint one id; the first
        # declaration wins.
        root = swiftcorpus({"m.swift": '''struct S {
    func request(url: Int) {}
    func request(path: String) {}
}
'''})
        files, _ = parse_corpus_files(root)
        assert [n.line for p in files for n in p.nodes if n.id == "m_s_request"] == [2]

    def test_a_computed_property_is_a_method_and_a_stored_one_is_not(self, swiftcorpus):
        root = swiftcorpus({"m.swift": '''struct R {
    let stored: Int
    var doubled: Int { stored * 2 }
}
'''})
        k = kinds(root)
        assert k["m_r_doubled"] == "method"
        assert "m_r_stored" not in k

    def test_a_protocol_requirement_is_a_node_even_with_no_body(self, swiftcorpus):
        root = swiftcorpus({"m.swift": '''protocol Thing {
    var name: String { get }
    func run()
}
'''})
        k = kinds(root)
        assert k["m_thing_name"] == "method"
        assert k["m_thing_run"] == "method"

    def test_a_nested_type_is_flat_in_its_id_and_nested_in_containment(self, swiftcorpus):
        root = swiftcorpus({"m.swift": '''class Outer {
    struct Inner { func go() {} }
}
'''})
        assert kinds(root)["m_inner"] == "class"
        got = links(root, "contains")
        assert ("m_outer", "m_inner") in got
        assert ("m_inner", "m_inner_go") in got

    def test_a_backtick_escaped_name_loses_its_backticks(self, swiftcorpus):
        # `default` is a keyword, so Swift writes it escaped; a caller writes
        # `Encoder.default` with no backticks and must find the same node.
        root = swiftcorpus({"m.swift":
                            "struct Encoder { public static var `default`: Int { 1 } }\n"})
        assert kinds(root)["m_encoder_default"] == "method"

    def test_a_doc_comment_becomes_a_rationale_node(self, swiftcorpus):
        got = docs(swiftcorpus(SESSION))
        assert "Accepts connections." in got["srv_session#doc"]
        assert "Begins serving." in got["srv_session_start#doc"]

    def test_a_block_doc_comment_is_taken_too(self, swiftcorpus):
        root = swiftcorpus({"m.swift": '''/** Holds a point. */
struct Point {}
'''})
        assert "Holds a point." in docs(root)["m_point#doc"]

    def test_a_plain_double_slash_run_is_not_documentation(self, swiftcorpus):
        # Every file in a real Swift repository opens with a licence header
        # written this way, and taking it would make the licence the first
        # declaration's rationale.
        root = swiftcorpus({"m.swift": '''//
//  Copyright (c) 2020 Somebody.
//
struct Point {}
'''})
        files, _ = parse_corpus_files(root)
        assert not any(n.kind == "rationale" for p in files for n in p.nodes)

    def test_a_doc_comment_separated_by_a_blank_line_is_not_documentation(self, swiftcorpus):
        root = swiftcorpus({"m.swift": '''/// Unrelated note.

struct Thing {}
'''})
        files, _ = parse_corpus_files(root)
        assert not any(n.kind == "rationale" for p in files for n in p.nodes)

    def test_conformance_is_inheritance(self, swiftcorpus):
        root = swiftcorpus({"m.swift": '''protocol Base {}
struct Derived: Base {}
'''})
        assert ("m_derived", "m_base") in links(root, "inherits")

    def test_a_retroactive_conformance_is_inheritance_too(self, swiftcorpus):
        # `extension Derived: Base` further down the file is how Swift states
        # inheritance after the fact, and it has to reach the class node that
        # was already emitted above it.
        root = swiftcorpus({"m.swift": '''protocol Base {}
struct Derived {}
extension Derived: Base {}
'''})
        assert ("m_derived", "m_base") in links(root, "inherits")


class TestExtensions:
    def test_an_extension_adds_methods_to_the_type_it_extends(self, swiftcorpus):
        root = swiftcorpus({"m.swift": '''struct Point {}
extension Point {
    func move() {}
}
'''})
        k = kinds(root)
        assert k["m_point_move"] == "method"
        assert ("m_point", "m_point_move") in links(root, "contains")

    def test_extending_a_type_declared_here_does_not_mint_a_second_class(self, swiftcorpus):
        root = swiftcorpus({"m.swift": '''struct Point {}
extension Point { func move() {} }
'''})
        files, _ = parse_corpus_files(root)
        assert [n.id for p in files for n in p.nodes if n.kind == "class"] == ["m_point"]

    def test_extending_a_foreign_type_mints_no_class_and_the_file_holds_it(self, swiftcorpus):
        # `Collection` lives in the standard library, which is not in the
        # corpus. Claiming it is defined here would make the name ambiguous
        # with wherever it really lives.
        root = swiftcorpus({"m.swift": "extension Collection { func trim() {} }\n"})
        files, _ = parse_corpus_files(root)
        assert not [n for p in files for n in p.nodes if n.kind == "class"]
        assert kinds(root)["m_collection_trim"] == "method"
        assert ("m", "m_collection_trim") in links(root, "contains")

    def test_an_extension_of_a_nested_type_scopes_to_the_nested_name(self, swiftcorpus):
        # `extension Outer.Inner` names the inner type, which is minted flat,
        # so its members have to land on that flat id.
        root = swiftcorpus({"m.swift": '''class Outer { struct Inner {} }
extension Outer.Inner { func go() {} }
'''})
        assert ("m_inner", "m_inner_go") in links(root, "contains")


class TestResolution:
    def test_an_explicit_self_call_resolves(self, swiftcorpus):
        assert ("srv_session_start", "srv_session_listen") in links(swiftcorpus(SESSION))

    def test_a_bare_call_inside_a_method_is_a_call_on_self(self, swiftcorpus):
        # Swift does not require `self.`, and most Swift omits it. Without this
        # the commonest call in the language resolves to nothing.
        root = swiftcorpus({"m.swift": '''struct S {
    func outer() { inner() }
    func inner() {}
}
'''})
        assert ("m_s_outer", "m_s_inner") in links(root)

    def test_a_bare_call_to_a_file_level_function(self, swiftcorpus):
        root = swiftcorpus({"m.swift": '''func helper() {}
func go() { helper() }
'''})
        assert ("m_go", "m_helper") in links(root)

    def test_an_initialiser_call_resolves_to_the_type(self, swiftcorpus):
        # Construction and a plain call are the same shape in Swift, so
        # `Session()` needs no special case at all.
        root = swiftcorpus({"m.swift": '''struct Session {}
func make() { _ = Session() }
'''})
        assert ("m_make", "m_session") in links(root)

    def test_a_typed_parameter_types_its_variable(self, swiftcorpus):
        root = swiftcorpus({"m.swift": '''struct Server { func ping() {} }
func use(srv: Server) { srv.ping() }
'''})
        assert ("m_use", "m_server_ping") in links(root)

    def test_an_annotated_local_types_its_variable(self, swiftcorpus):
        root = swiftcorpus({"m.swift": '''struct Server { func ping() {} }
func use(any: Int) {
    let srv: Server = build()
    srv.ping()
}
'''})
        assert ("m_use", "m_server_ping") in links(root)

    def test_an_initialiser_assignment_types_its_variable(self, swiftcorpus):
        root = swiftcorpus({"m.swift": '''struct Server { func ping() {} }
func use() {
    let srv = Server()
    srv.ping()
}
'''})
        assert ("m_use", "m_server_ping") in links(root)

    def test_a_stored_property_types_a_bare_receiver(self, swiftcorpus):
        # `preprocessor.run()` with no `self.` is how Swift reads a property,
        # and the property's declared type is the evidence.
        root = swiftcorpus({"m.swift": '''struct Pre { func run() {} }
struct Holder {
    let pre: Pre
    func go() { pre.run() }
}
'''})
        assert ("m_holder_go", "m_pre_run") in links(root)

    def test_a_self_attribute_call_resolves(self, swiftcorpus):
        root = swiftcorpus({"m.swift": '''struct Pre { func run() {} }
struct Holder {
    let pre: Pre
    func go() { self.pre.run() }
}
'''})
        assert ("m_holder_go", "m_pre_run") in links(root)

    def test_a_generic_type_argument_does_not_hide_the_type(self, swiftcorpus):
        root = swiftcorpus({"m.swift": '''struct Protected<T> { func write() {} }
struct Holder {
    let state: Protected<Int>
    func go() { state.write() }
}
'''})
        assert ("m_holder_go", "m_protected_write") in links(root)

    def test_a_call_on_an_untyped_receiver_is_refused_not_guessed(self, swiftcorpus):
        # A loop variable states no type, and `ping` exists in the corpus, so
        # guessing would be easy and wrong.
        root = swiftcorpus({"m.swift": '''struct Server { func ping() {} }
func use() {
    for thing in everything {
        thing.ping()
    }
}
'''})
        assert reasons_for(root)["receiver type unknown"] >= 1
        assert ("m_use", "m_server_ping") not in links(root)

    def test_a_lowercase_factory_call_does_not_type_its_variable(self, swiftcorpus):
        # `let s = build()` states nothing about what `s` is, so nothing is
        # recorded and the call on it is refused rather than guessed.
        root = swiftcorpus({"m.swift": '''struct Server { func ping() {} }
func build() -> Server { Server() }
func use() {
    let srv = build()
    srv.ping()
}
'''})
        assert ("m_use", "m_server_ping") not in links(root)
        assert reasons_for(root)["receiver type unknown"] >= 1

    def test_a_module_import_is_recorded_and_stays_unresolved(self, swiftcorpus):
        # `import Foundation` names a module, not a file, so it cannot point at
        # a node in this corpus -- which is the useful answer, not a failure.
        root = swiftcorpus({"m.swift": "import Foundation\nstruct P {}\n"})
        files, _ = parse_corpus_files(root)
        assert files[0].imports["Foundation"] == "Foundation"
        assert files[0].import_sites == [("Foundation", 1)]
        edges, _ = resolve(files)
        assert any(e.relation == "imports" and not e.resolved for e in edges)

    def test_a_submodule_import_binds_its_last_segment(self, swiftcorpus):
        root = swiftcorpus({"m.swift": "import class Foundation.Thread\n"})
        files, _ = parse_corpus_files(root)
        assert files[0].imports["Thread"] == "Foundation.Thread"

    def test_files_in_one_module_see_each_other_without_importing(self, swiftcorpus):
        # Swift has no file-level imports inside a module, so a cross-file call
        # has to resolve on the name alone.
        root = swiftcorpus({"a.swift": "struct Server { func ping() {} }\n",
                            "b.swift": "func use(srv: Server) { srv.ping() }\n"})
        assert ("b_use", "a_server_ping") in links(root)

    def test_resolved_edges_never_dangle(self, swiftcorpus):
        files, _ = parse_corpus_files(swiftcorpus(SESSION))
        ids = {n.id for p in files for n in p.nodes}
        edges, _ = resolve(files)
        for e in edges:
            if e.resolved:
                assert e.target in ids


class TestMixedCorpus:
    def test_python_and_swift_in_one_repository(self, swiftcorpus):
        root = swiftcorpus({"app/srv.swift": "struct Server {}\n",
                            "tools/build.py": "class Builder:\n    pass\n"})
        k = kinds(root)
        assert k["app_srv_server"] == "class"
        assert k["tools_build_builder"] == "class"


class TestFailure:
    def test_an_unreadable_file_never_takes_the_others_with_it(self, swiftcorpus):
        # Stated honestly: the Swift grammar recovers from almost any input, so
        # a file of pure garbage still yields a tree with children and is not
        # refused. What must never happen is the good file beside it vanishing.
        root = swiftcorpus({"bad.swift": "\x00\x00 ((((", "ok.swift": "struct P {}\n"})
        _, failed = parse_corpus_files(root)
        assert "ok.swift" not in failed
        assert kinds(root)["ok_p"] == "class"

    def test_a_syntax_error_in_one_function_keeps_the_rest_of_the_file(self, swiftcorpus):
        # Real Swift outruns the grammar; eight of the seventy-one files in the
        # measured corpora carry an error node. Dropping the file whole would
        # cost far more than the statement the error sits in.
        root = swiftcorpus({"m.swift": '''struct Good { func fine() {} }
func broken() { let x = @@@ }
'''})
        _, failed = parse_corpus_files(root)
        assert failed == []
        assert kinds(root)["m_good_fine"] == "method"
