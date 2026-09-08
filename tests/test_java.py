"""Java, read through tree-sitter.

The point of another language is not that it works, but that it produces the
SAME shapes: a Java interface is a `class` node, `implements` is `inherits`, a
Javadoc block is a `rationale`. If any of that needed a new word, the model was
never language-neutral and everything downstream would have to learn Java too.

Java's own advantage is tested here as well. Every type is written down -- a
field, a parameter, a local, a loop variable -- so a call on any of them
resolves from what the source states rather than from a guess.
"""
import pytest

from graphpaat.parse import parse_corpus_files
from graphpaat.resolve import resolve

pytest.importorskip("tree_sitter_java")


@pytest.fixture
def javacorpus(tmp_path):
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


def nodes(root):
    files, _ = parse_corpus_files(root)
    return [n for p in files for n in p.nodes]


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


SERVER = {"Srv.java": '''package app;

/** Accepts connections. */
public class Srv extends Base implements Handler {
  private final Pool pool;

  /** Builds a server on a pool. */
  public Srv(Pool pool) {
    this.pool = pool;
  }

  /** Begins serving. */
  public void start() {
    listen();
  }

  private void listen() {}
}
''',
          "Base.java": "package app;\npublic class Base {}\n",
          "Handler.java": "package app;\npublic interface Handler {}\n",
          "Pool.java": "package app;\npublic class Pool { public void open() {} }\n"}


class TestSameShapesAsPython:
    def test_a_class_is_a_class_node(self, javacorpus):
        assert kinds(javacorpus(SERVER))["srv_srv"] == "class"

    def test_an_interface_is_also_a_class_node(self, javacorpus):
        root = javacorpus({"I.java": "interface I { void go(); }\n"})
        assert kinds(root)["i_i"] == "class"

    def test_an_enum_is_also_a_class_node(self, javacorpus):
        root = javacorpus({"E.java": "enum E { A, B; void go() {} }\n"})
        k = kinds(root)
        assert k["e_e"] == "class"
        assert k["e_e_go"] == "method"

    def test_a_record_is_also_a_class_node(self, javacorpus):
        root = javacorpus({"Pt.java": "record Pt(int x, int y) {}\n"})
        assert kinds(root)["pt_pt"] == "class"

    def test_an_annotation_type_is_also_a_class_node(self, javacorpus):
        root = javacorpus({"Ann.java": '@interface Ann { String value(); }\n'})
        k = kinds(root)
        assert k["ann_ann"] == "class"
        assert k["ann_ann_value"] == "method"

    def test_methods_are_qualified_by_their_type(self, javacorpus):
        k = kinds(javacorpus(SERVER))
        assert k["srv_srv_start"] == "method"
        assert k["srv_srv_listen"] == "method"

    def test_a_constructor_is_a_method_named_for_its_class(self, javacorpus):
        # `Srv(Pool)` mints srv_srv_srv: the class is srv_srv, the method inside
        # it carries the class name again, exactly as Java writes it.
        assert kinds(javacorpus(SERVER))["srv_srv_srv"] == "method"

    def test_two_types_with_a_same_named_method_do_not_collide(self, javacorpus):
        root = javacorpus({"M.java": '''
class A { void run() {} }
class B { void run() {} }
'''})
        k = kinds(root)
        assert "m_a_run" in k and "m_b_run" in k

    def test_a_javadoc_becomes_a_rationale_node(self, javacorpus):
        found = docs(javacorpus(SERVER))
        assert found["srv_srv#doc"] == "Accepts connections."
        assert found["srv_srv_start#doc"] == "Begins serving."

    def test_a_javadoc_separated_by_a_blank_line_is_not_documentation(self, javacorpus):
        root = javacorpus({"M.java": '''
/** Unrelated note. */

class Thing {}
'''})
        assert not any(n.kind == "rationale" for n in nodes(root))

    def test_a_line_comment_is_not_documentation(self, javacorpus):
        # `//` above a declaration is a note about the line far more often than
        # it is documentation, and taking those too buried the real Javadoc.
        root = javacorpus({"M.java": "// not documentation\nclass Thing {}\n"})
        assert not any(n.kind == "rationale" for n in nodes(root))

    def test_an_annotation_does_not_hide_the_javadoc(self, javacorpus):
        # `@Deprecated` is a child of the declaration, so the Javadoc is still
        # the sibling immediately before it.
        root = javacorpus({"M.java": '''
class Thing {
  /** Still documented. */
  @Deprecated
  public void go() {}
}
'''})
        assert docs(root)["m_thing_go#doc"] == "Still documented."

    def test_a_marker_comment_does_not_hide_the_javadoc(self, javacorpus):
        # commons-lang writes `//@Immutable` between the Javadoc and `public
        # class StringUtils`. Stopping at the first non-Javadoc comment cost
        # that class its entire description.
        root = javacorpus({"M.java": '''
/** Still documented. */
//@Immutable
class Thing {}
'''})
        assert docs(root)["m_thing#doc"] == "Still documented."

    def test_a_blank_line_inside_the_comment_run_still_breaks_it(self, javacorpus):
        root = javacorpus({"M.java": '''
/** Unrelated note. */

//@Immutable
class Thing {}
'''})
        assert not any(n.kind == "rationale" for n in nodes(root))

    def test_a_package_info_javadoc_documents_the_file(self, javacorpus):
        # The one Javadoc in Java that describes a file rather than a
        # declaration. A licence header in the same position opens `/*` and is
        # correctly ignored.
        root = javacorpus({"package-info.java": '''/*
 * Licence header, not documentation.
 */

/** What this package is for. */
package app;
'''})
        assert docs(root)["package-info#doc"] == "What this package is for."

    def test_extends_and_implements_are_both_inheritance(self, javacorpus):
        got = links(javacorpus(SERVER), "inherits")
        assert ("srv_srv", "base_base") in got
        assert ("srv_srv", "handler_handler") in got

    def test_containment_matches_the_python_shape(self, javacorpus):
        got = links(javacorpus(SERVER), "contains")
        assert ("srv", "srv_srv") in got
        assert ("srv_srv", "srv_srv_start") in got

    def test_a_nested_type_is_contained_by_the_type_around_it(self, javacorpus):
        # The id stays flat inside the file -- `m_outer_inner`, not
        # `m_outer_outer_inner` -- because that is what an import of the nested
        # type and every receiver-typed call go looking for.
        root = javacorpus({"M.java": "class Outer { static class Inner { void go() {} } }\n"})
        assert kinds(root)["m_inner"] == "class"
        assert ("m_outer", "m_inner") in links(root, "contains")
        assert ("m_inner", "m_inner_go") in links(root, "contains")

    def test_an_overload_does_not_become_a_second_node(self, javacorpus):
        # Two signatures, one name: a caller writes `pad(...)` for both, so they
        # are one entry point. Emitting each would mint the same id twice and
        # report a symbol as lost that was never distinct.
        root = javacorpus({"U.java": '''
class U {
  static String pad(String s) { return pad(s, 1); }
  static String pad(String s, int n) { return s; }
}
'''})
        found = [n for n in nodes(root) if n.id == "u_u_pad"]
        assert len(found) == 1
        assert found[0].line == 3

    def test_an_enum_constant_is_not_a_class_but_its_methods_are_the_enums(self, javacorpus):
        # `DOUBLE` is a value, not a type, so it gets no node. The method inside
        # its body is what the enum actually does, so it belongs to the enum.
        root = javacorpus({"P.java": '''
enum P {
  DOUBLE {
    public int read() { return 1; }
  },
  LONG {
    public int read() { return 2; }
  };
}
'''})
        k = kinds(root)
        assert "p_double" not in k
        assert k["p_p_read"] == "method"


class TestResolution:
    def test_an_unqualified_call_is_a_call_on_the_enclosing_type(self, javacorpus):
        # Java's `listen()` is Python's `self.listen()`. Treating it as a free
        # name would send it to the resolver's weakest rule, which will happily
        # match a same-named method in an unrelated class.
        assert ("srv_srv_start", "srv_srv_listen") in links(javacorpus(SERVER))

    def test_a_call_on_this_resolves(self, javacorpus):
        root = javacorpus({"M.java": '''
class A {
  void go() { this.helper(); }
  void helper() {}
}
'''})
        assert ("m_a_go", "m_a_helper") in links(root)

    def test_a_typed_parameter_types_its_variable(self, javacorpus):
        root = javacorpus({"M.java": '''
class Pool { void open() {} }
class Use { void go(Pool p) { p.open(); } }
'''})
        assert ("m_use_go", "m_pool_open") in links(root)

    def test_a_local_declaration_types_its_variable(self, javacorpus):
        root = javacorpus({"M.java": '''
class Pool { void open() {} }
class Use { void go() { Pool p = make(); p.open(); } }
'''})
        assert ("m_use_go", "m_pool_open") in links(root)

    def test_var_reads_its_type_from_the_initializer(self, javacorpus):
        # `var` states the type on the other side of the `=`. Without reading it
        # every `var` in a modern file would be untyped.
        root = javacorpus({"M.java": '''
class Pool { void open() {} }
class Use { void go() { var p = new Pool(); p.open(); } }
'''})
        assert ("m_use_go", "m_pool_open") in links(root)

    def test_a_loop_variable_is_typed(self, javacorpus):
        root = javacorpus({"M.java": '''
class Pool { void open() {} }
class Use { void go(java.util.List<Pool> all) { for (Pool p : all) { p.open(); } } }
'''})
        assert ("m_use_go", "m_pool_open") in links(root)

    def test_a_field_type_resolves_a_call_on_this(self, javacorpus):
        root = javacorpus({"M.java": '''
class Pool { void open() {} }
class Use {
  private final Pool pool = null;
  void go() { this.pool.open(); }
}
'''})
        assert ("m_use_go", "m_pool_open") in links(root)

    def test_a_field_type_resolves_the_bare_field_read(self, javacorpus):
        # Java writes `pool.open()` far more often than `this.pool.open()`, and
        # at the call site an unqualified field looks exactly like a local.
        root = javacorpus({"M.java": '''
class Pool { void open() {} }
class Use {
  private final Pool pool = null;
  void go() { pool.open(); }
}
'''})
        assert ("m_use_go", "m_pool_open") in links(root)

    def test_a_static_call_on_a_class_name_resolves(self, javacorpus):
        root = javacorpus({"M.java": '''
class Util { static boolean isEmpty(String s) { return false; } }
class Use { void go() { Util.isEmpty("a"); } }
'''})
        assert ("m_use_go", "m_util_isempty") in links(root)

    def test_new_is_a_call_to_the_class(self, javacorpus):
        root = javacorpus({"M.java": '''
class Pool {}
class Use { void go() { new Pool(); } }
'''})
        assert ("m_use_go", "m_pool") in links(root)

    def test_an_import_of_a_nested_type_resolves_to_the_file_holding_it(self, javacorpus):
        # `com.example.Outer.Inner` is a type inside Outer.java, not a file
        # com/example/Outer/Inner.java. Java's capitalisation is what separates
        # the package part of the path from the type part.
        root = javacorpus({
            "com/example/Outer.java":
                "package com.example;\npublic class Outer { public static class Inner {} }\n",
            "org/app/User.java": '''package org.app;
import com.example.Outer.Inner;
public class User { void go() { new Inner(); } }
'''})
        assert ("org_app_user_user_go", "com_example_outer_inner") in links(root)

    def test_a_type_in_the_same_package_resolves_with_no_import(self, javacorpus):
        # Java needs no import for a sibling in its own package, so nothing in
        # the file records the link. Without the package rule the resolver sees
        # two candidates named Pool -- the class and its own constructor -- and
        # refuses.
        root = javacorpus({
            "com/example/Pool.java": "package com.example;\npublic class Pool { public Pool() {} }\n",
            "com/example/Use.java":
                "package com.example;\npublic class Use { void go() { new Pool(); } }\n"})
        assert ("com_example_use_use_go", "com_example_pool_pool") in links(root)

    def test_a_call_on_a_foreign_type_is_refused_not_guessed(self, javacorpus):
        # Java types everything, so the interesting refusal is not an unknown
        # receiver but a known one that is not in the corpus. `thing` is an
        # Object; a method called `open` exists here, and the link must not be
        # drawn on the strength of the name alone.
        root = javacorpus({"M.java": '''
class Pool { void open() {} }
class Use { void go(Object thing) { thing.open(); } }
'''})
        assert ("m_use_go", "m_pool_open") not in links(root)
        assert reasons_for(root)["receiver typed, but no such method in the corpus"] >= 1

    def test_a_call_on_a_chained_expression_is_refused_not_guessed(self, javacorpus):
        # `a.b().open()` has no name to type. Recording the expression itself as
        # the receiver guarantees it can never match a declared variable, so the
        # call is counted as a refusal instead of vanishing from the report.
        root = javacorpus({"M.java": '''
class Pool { void open() {} }
class Use { void go(Object a) { a.get().open(); } }
'''})
        assert reasons_for(root)["receiver type unknown"] >= 1

    def test_imports_are_recorded(self, javacorpus):
        root = javacorpus({"M.java": "import java.util.List;\nclass A {}\n"})
        files, _ = parse_corpus_files(root)
        # Stored as <file>.<name>: the resolver finds an imported symbol by
        # dropping the last segment, and in Java the file IS the type name.
        assert files[0].imports["List"] == "java.util.List.List"
        assert ("java.util.List", 1) in files[0].import_sites

    def test_a_static_import_names_the_file_not_the_member(self, javacorpus):
        root = javacorpus({"M.java": "import static java.util.Objects.requireNonNull;\nclass A {}\n"})
        files, _ = parse_corpus_files(root)
        assert ("java.util.Objects", 1) in files[0].import_sites

    def test_a_wildcard_import_binds_no_name(self, javacorpus):
        root = javacorpus({"M.java": "import java.util.*;\nclass A {}\n"})
        files, _ = parse_corpus_files(root)
        assert files[0].imports == {}
        assert ("java.util", 1) in files[0].import_sites

    def test_an_import_outside_the_corpus_stays_unresolved(self, javacorpus):
        root = javacorpus({"M.java": "import java.util.List;\nclass A {}\n"})
        assert reasons_for(root)["imports a module outside this corpus"] >= 1

    def test_resolved_edges_never_dangle(self, javacorpus):
        files, _ = parse_corpus_files(javacorpus(SERVER))
        ids = {n.id for p in files for n in p.nodes}
        edges, _ = resolve(files)
        for e in edges:
            if e.resolved:
                assert e.target in ids


class TestMixedCorpus:
    def test_python_and_java_in_one_repository(self, javacorpus):
        root = javacorpus({"api/Srv.java": "package api;\npublic class Srv {}\n",
                           "tools/build.py": "class Builder:\n    pass\n"})
        k = kinds(root)
        assert k["api_srv_srv"] == "class"
        assert k["tools_build_builder"] == "class"


class TestFailure:
    def test_a_broken_file_does_not_take_down_the_file_beside_it(self, javacorpus):
        root = javacorpus({"Bad.java": "class \x00\x00 ((((", "Ok.java": "class Ok {}\n"})
        _, failed = parse_corpus_files(root)
        assert "Ok.java" not in failed
        assert kinds(root)["ok_ok"] == "class"

    def test_a_broken_file_invents_no_symbols(self, javacorpus):
        # tree-sitter always returns a tree, so an unreadable Java file is never
        # reported in `failed`; what it must never do is mint a symbol that is
        # not in the source. The file node is all that is left.
        root = javacorpus({"Bad.java": "class \x00\x00 (((("})
        assert [n.kind for n in nodes(root)] == ["file"]
