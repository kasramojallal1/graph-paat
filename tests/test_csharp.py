"""C#, read through tree-sitter.

The same question as every other language: does it produce the SAME shapes,
without anything downstream learning it exists? C# pushes on two places the
earlier languages did not. Its declarations are nested inside a namespace, so a
reader that walks only the top of the file finds an empty repository. And it
has method overloading, which means several distinct symbols legitimately claim
one id -- the tests below check that this is reported rather than hidden.
"""
import pytest

from graphpaat.parse import parse_corpus, parse_corpus_files
from graphpaat.resolve import resolve

pytest.importorskip("tree_sitter_c_sharp")


@pytest.fixture
def cs(tmp_path):
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


def nodes_of(root):
    files, _ = parse_corpus_files(root)
    return {n.id: n for p in files for n in p.nodes}


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


def docs_of(root):
    files, _ = parse_corpus_files(root)
    return {n.id: n.text for p in files for n in p.nodes if n.kind == "rationale"}


SERVER = {"srv.cs": '''namespace Demo;

/// <summary>Accepts connections.</summary>
public class Server
{
    readonly Listener _listener;

    /// <summary>Builds a server.</summary>
    public Server(Listener listener) { _listener = listener; }

    /// <summary>Begins serving.</summary>
    public void Start() { this.Listen(); }

    void Listen() { _listener.Accept(); }
}

public class Listener
{
    public void Accept() { }
}
'''}


class TestSameShapesAsPython:
    def test_a_class_is_a_class_node(self, cs):
        assert kinds(cs(SERVER))["srv_server"] == "class"

    def test_an_interface_is_also_a_class_node(self, cs):
        root = cs({"i.cs": "namespace D;\npublic interface IHandler { void Do(); }\n"})
        assert kinds(root)["i_ihandler"] == "class"

    def test_a_struct_is_also_a_class_node(self, cs):
        root = cs({"s.cs": "namespace D;\npublic struct Point { public int X; }\n"})
        assert kinds(root)["s_point"] == "class"

    def test_a_record_is_also_a_class_node(self, cs):
        root = cs({"r.cs": "namespace D;\npublic record Pair(int A, int B);\n"
                           "public record struct Half(int A);\n"})
        k = kinds(root)
        assert k["r_pair"] == "class" and k["r_half"] == "class"

    def test_an_enum_is_also_a_class_node(self, cs):
        root = cs({"e.cs": "namespace D;\npublic enum Colour { Red, Green }\n"})
        assert kinds(root)["e_colour"] == "class"

    def test_a_delegate_is_also_a_class_node(self, cs):
        # It names a type, so it is the same shape as every other named type.
        root = cs({"d.cs": "namespace D;\npublic delegate void Notify(int x);\n"})
        assert kinds(root)["d_notify"] == "class"

    def test_methods_are_qualified_by_their_class(self, cs):
        k = kinds(cs(SERVER))
        assert k["srv_server_start"] == "method"
        assert k["srv_listener_accept"] == "method"

    def test_two_classes_with_a_same_named_method_do_not_collide(self, cs):
        root = cs({"m.cs": '''namespace D;
class A { public void Run() { } }
class B { public void Run() { } }
'''})
        k = kinds(root)
        assert "m_a_run" in k and "m_b_run" in k

    def test_a_constructor_is_a_method_named_after_its_type(self, cs):
        assert kinds(cs(SERVER))["srv_server_server"] == "method"

    def test_a_property_with_a_body_is_a_method(self, cs):
        root = cs({"p.cs": '''namespace D;
class Box { public int Size { get { return 1; } } }
'''})
        assert kinds(root)["p_box_size"] == "method"

    def test_a_property_without_a_body_is_still_a_method(self, cs):
        # A C# property is an accessor pair even when the compiler writes the
        # bodies, and the settings objects of a real library are nothing but
        # these -- dropping them would delete the documented surface.
        root = cs({"p.cs": '''namespace D;
class Options { public int Size { get; set; } }
'''})
        assert kinds(root)["p_options_size"] == "method"

    def test_a_field_is_not_a_node(self, cs):
        # A field is private state, not a member anyone calls.
        root = cs({"f.cs": "namespace D;\nclass Box { int _size; }\n"})
        assert "f_box__size" not in kinds(root)

    def test_a_local_function_is_a_function_not_a_method(self, cs):
        root = cs({"l.cs": '''namespace D;
class Box { public void Go() { void Helper() { } Helper(); } }
'''})
        assert kinds(root)["l_box_go_helper"] == "function"

    def test_a_doc_comment_becomes_a_rationale_node(self, cs):
        docs = docs_of(cs(SERVER))
        assert docs["srv_server#doc"] == "Accepts connections."
        assert docs["srv_server_start#doc"] == "Begins serving."

    def test_a_line_comment_is_not_documentation(self, cs):
        # Every file in a real C# repository opens with a licence header of
        # these, and taking them made the header the first class's rationale.
        root = cs({"m.cs": '''namespace D;
// Copyright someone.
// All rights reserved.
public class Thing { }
'''})
        assert not docs_of(root)

    def test_a_doc_comment_separated_by_a_blank_line_is_not_documentation(self, cs):
        root = cs({"m.cs": '''namespace D;

/// <summary>Unrelated note.</summary>

public class Thing { }
'''})
        assert not docs_of(root)

    def test_doc_xml_is_flattened_and_a_cref_keeps_its_name(self, cs):
        root = cs({"m.cs": '''namespace D;
/// <summary>
/// Wraps <see cref="Thing"/> for <paramref name="count"/> items.
/// </summary>
public class Box { }
'''})
        assert docs_of(root)["m_box#doc"] == "Wraps Thing for count items."

    def test_a_code_example_is_dropped_from_the_doc(self, cs):
        # A sample crowds the actual sentence out of the size budget.
        root = cs({"m.cs": '''namespace D;
/// <summary>Builds a box.</summary>
/// <example><code>var b = new Box();</code></example>
public class Box { }
'''})
        assert docs_of(root)["m_box#doc"] == "Builds a box."

    def test_a_langword_reference_keeps_its_keyword(self, cs):
        # Dropping the tag left "returns  if ok; otherwise, ." Serilog alone
        # writes 41 of these.
        root = cs({"m.cs": '''namespace D;
class Box
{
    /// <returns><see langword="true"/> if ok; otherwise, <see langword="false"/>.</returns>
    public bool Ok() { return true; }
}
'''})
        assert docs_of(root)["m_box_ok#doc"] == "true if ok; otherwise, false."

    def test_an_inheritdoc_alone_is_not_a_claim(self, cs):
        # It says "read the base class", which claims nothing about this one.
        root = cs({"m.cs": "namespace D;\n/// <inheritdoc/>\npublic class Box { }\n"})
        assert not docs_of(root)

    def test_an_attribute_between_the_doc_and_the_signature_keeps_the_doc(self, cs):
        # The attribute is a child of the declaration, so the comment is still
        # the declaration's previous sibling -- but the declaration now starts
        # on the attribute's line, which is what the adjacency check measures.
        root = cs({"m.cs": '''namespace D;
class Box
{
    /// <summary>Runs it.</summary>
    [Obsolete("no")]
    public void Run() { }
}
'''})
        assert docs_of(root)["m_box_run#doc"] == "Runs it."

    def test_a_base_class_and_an_interface_are_both_inheritance(self, cs):
        root = cs({"m.cs": '''namespace D;
class Base { }
interface IThing { }
class Derived : Base, IThing { }
'''})
        got = links(root, "inherits")
        assert ("m_derived", "m_base") in got
        assert ("m_derived", "m_ithing") in got

    def test_an_enums_underlying_type_is_not_a_base_class(self, cs):
        # `: byte` looks like a base list and names no class.
        root = cs({"m.cs": "namespace D;\npublic enum Flag : byte { On }\n"})
        assert nodes_of(root)["m_flag"].bases is None

    def test_containment_matches_the_python_shape(self, cs):
        got = links(cs(SERVER), "contains")
        assert ("srv", "srv_server") in got
        assert ("srv_server", "srv_server_start") in got


class TestWhereDeclarationsHide:
    """The failure mode that would empty the whole graph."""

    def test_a_file_scoped_namespace_does_not_hide_its_types(self, cs):
        root = cs({"m.cs": "namespace D.Sub;\nclass Thing { }\n"})
        assert kinds(root)["m_thing"] == "class"

    def test_a_braced_namespace_does_not_hide_its_types(self, cs):
        # This is how most of a mature C# codebase is written, and a walk of
        # the file's own children finds nothing at all in it.
        root = cs({"m.cs": '''namespace D.Sub
{
    class Thing { public void Go() { } }
}
'''})
        k = kinds(root)
        assert k["m_thing"] == "class" and k["m_thing_go"] == "method"

    def test_a_namespace_is_not_part_of_an_id(self, cs):
        # The file path already separates two types of the same name.
        root = cs({"a/m.cs": "namespace Very.Long.Name;\nclass Thing { }\n"})
        assert "a_m_thing" in kinds(root)

    def test_a_declaration_inside_a_preprocessor_block_is_found(self, cs):
        root = cs({"m.cs": '''namespace D;
class Thing
{
#if FEATURE_X
    public void Extra() { }
#endif
    public void Always() { }
}
'''})
        k = kinds(root)
        assert k["m_thing_extra"] == "method" and k["m_thing_always"] == "method"

    def test_a_preprocessor_line_in_a_class_header_does_not_rename_the_class(self, cs):
        # The grammar cannot parse this and recovers by promoting the
        # preprocessor symbol to the class's name, which files every method in
        # the file under a type that does not exist.
        root = cs({"m.cs": '''namespace D;
public abstract partial class Writer
#if HAVE_ASYNC
    : IDisposable
#endif
{
    public void Flush() { }
}
'''})
        k = kinds(root)
        assert k["m_writer"] == "class"
        assert k["m_writer_flush"] == "method"
        assert "m_have_async" not in k

    def test_a_nested_type_is_named_by_the_type_that_holds_it(self, cs):
        root = cs({"m.cs": '''namespace D;
class Outer { class Inner { public void Deep() { } } }
'''})
        k = kinds(root)
        assert k["m_outer_inner"] == "class"
        assert k["m_outer_inner_deep"] == "method"
        contains = links(root, "contains")
        assert ("m_outer", "m_outer_inner") in contains
        assert ("m_outer_inner", "m_outer_inner_deep") in contains


class TestResolution:
    def test_a_call_on_this_resolves_within_the_class(self, cs):
        assert ("srv_server_start", "srv_server_listen") in links(cs(SERVER))

    def test_a_plain_call_in_the_same_file(self, cs):
        root = cs({"m.cs": '''namespace D;
class Box
{
    void Helper() { }
    public void Go() { Helper(); }
}
'''})
        assert ("m_box_go", "m_box_helper") in links(root)

    def test_a_typed_parameter_types_its_variable(self, cs):
        root = cs({"m.cs": '''namespace D;
class Server { public void Ping() { } }
class User { public void Go(Server srv) { srv.Ping(); } }
'''})
        assert ("m_user_go", "m_server_ping") in links(root)

    def test_a_typed_local_types_its_variable(self, cs):
        root = cs({"m.cs": '''namespace D;
class Server { public void Ping() { } }
class User { public void Go() { Server srv = Make(); srv.Ping(); } }
'''})
        assert ("m_user_go", "m_server_ping") in links(root)

    def test_var_with_a_construction_types_its_variable(self, cs):
        root = cs({"m.cs": '''namespace D;
class Server { public void Ping() { } }
class User { public void Go() { var srv = new Server(); srv.Ping(); } }
'''})
        assert ("m_user_go", "m_server_ping") in links(root)

    def test_a_field_types_a_receiver_used_without_this(self, cs):
        # C# writes the field's type out and then reads it bare, which is how
        # most calls inside a class are written.
        assert ("srv_server_listen", "srv_listener_accept") in links(cs(SERVER))

    def test_a_field_types_a_receiver_reached_through_this(self, cs):
        root = cs({"m.cs": '''namespace D;
class Listener { public void Accept() { } }
class Server
{
    readonly Listener _listener;
    public void Go() { this._listener.Accept(); }
}
'''})
        assert ("m_server_go", "m_listener_accept") in links(root)

    def test_a_typed_foreach_variable_resolves(self, cs):
        root = cs({"m.cs": '''namespace D;
class Server { public void Ping() { } }
class User
{
    public void Go(System.Collections.Generic.List<Server> all)
    {
        foreach (Server srv in all) { srv.Ping(); }
    }
}
'''})
        assert ("m_user_go", "m_server_ping") in links(root)

    def test_a_null_conditional_call_resolves(self, cs):
        root = cs({"m.cs": '''namespace D;
class Server { public void Ping() { } }
class User { public void Go(Server srv) { srv?.Ping(); } }
'''})
        assert ("m_user_go", "m_server_ping") in links(root)

    def test_a_generic_call_resolves_by_its_plain_name(self, cs):
        root = cs({"m.cs": '''namespace D;
class Server { public void Ping<T>() { } }
class User { public void Go(Server srv) { srv.Ping<int>(); } }
'''})
        assert ("m_user_go", "m_server_ping") in links(root)

    def test_constructing_a_class_is_a_call_to_it(self, cs):
        # Often the only edge tying a factory to what it builds.
        root = cs({"m.cs": '''namespace D;
class Server { }
class Factory { public Server Make() { return new Server(); } }
'''})
        assert ("m_factory_make", "m_server") in links(root)

    def test_a_call_on_an_untyped_receiver_is_refused_not_guessed(self, cs):
        root = cs({"m.cs": '''namespace D;
class Server { public void Ping() { } }
class User { public void Go(object thing) { thing.Ping(); } }
'''})
        assert reasons_for(root)["receiver type unknown"] >= 1
        assert ("m_user_go", "m_server_ping") not in links(root)

    def test_a_call_on_a_chained_receiver_is_not_treated_as_a_bare_call(self, cs):
        # `holder.Inner.Ping()` names no variable we can type. Reporting no
        # receiver at all would make resolution answer it with any same-file
        # Ping -- a wrong edge instead of an honest gap.
        root = cs({"m.cs": '''namespace D;
class Server { public void Ping() { } }
class Holder { public Server Inner; }
class User { public void Go(Holder h) { h.Inner.Ping(); } }
'''})
        assert ("m_user_go", "m_server_ping") not in links(root)
        assert reasons_for(root)["receiver type unknown"] >= 1

    def test_a_call_on_base_is_not_treated_as_a_call_on_self(self, cs):
        # `base.Ping()` names the parent's method. Calling it self-typed would
        # resolve it to the override that is asking for the original.
        root = cs({"m.cs": '''namespace D;
class Base { public virtual void Ping() { } }
class Child : Base { public void Go() { base.Ping(); } }
'''})
        assert reasons_for(root)["receiver type unknown"] >= 1
        assert ("m_child_go", "m_base_ping") not in links(root)

    def test_a_plain_using_is_recorded_as_an_import(self, cs):
        root = cs({"m.cs": "using System.Text;\nnamespace D;\nclass Thing { }\n"})
        files, _ = parse_corpus_files(root)
        assert files[0].import_sites == [("System.Text", 1)]

    def test_a_plain_using_binds_no_local_name(self, cs):
        # It opens a namespace: every type in it becomes visible without any
        # one of them being named, so there is no local name to record.
        root = cs({"m.cs": "using System.Text;\nnamespace D;\nclass Thing { }\n"})
        files, _ = parse_corpus_files(root)
        assert files[0].imports == {}

    def test_an_alias_using_binds_its_local_name(self, cs):
        root = cs({"m.cs": "using Box = Deep.Inner.Thing;\nnamespace D;\nclass A { }\n"})
        files, _ = parse_corpus_files(root)
        assert files[0].imports == {"Box": "Deep.Inner.Thing"}

    def test_a_using_that_names_a_file_in_the_corpus_resolves_to_it(self, cs):
        root = cs({"util/helpers.cs": "namespace App.Util.Helpers;\nclass H { }\n",
                   "main.cs": "using App.Util.Helpers;\nnamespace App;\nclass M { }\n"})
        assert ("main", "util_helpers") in links(root, "imports")

    def test_resolved_edges_never_dangle(self, cs):
        root = cs({"srv.cs": SERVER["srv.cs"], "n.cs": '''namespace D;
class Outer
{
    class Inner { public void Deep() { } }
    public void Go() { var i = new Inner(); i.Deep(); }
    public void Go(int x) { }
}
'''})
        files, _ = parse_corpus_files(root)
        ids = {n.id for p in files for n in p.nodes}
        edges, _ = resolve(files)
        for edge in edges:
            if edge.resolved:
                assert edge.target in ids


class TestOverloads:
    def test_an_overload_set_is_reported_as_a_collision_not_hidden(self, cs):
        # C# lets four methods share a name, and the id scheme gives them one
        # id. Three of them are lost; the point is that the loss is counted.
        root = cs({"m.cs": '''namespace D;
class Writer
{
    public void Write(int x) { }
    public void Write(string x) { }
    public void Write(bool x) { }
}
'''})
        _, _, collisions, _ = parse_corpus(root)
        assert collisions.collided()["m_writer_write"] == [
            "m.cs:L4", "m.cs:L5", "m.cs:L6"]


class TestMixedCorpus:
    def test_python_and_csharp_in_one_repository(self, cs):
        root = cs({"api/srv.cs": "namespace Api;\npublic class Server { }\n",
                   "tools/build.py": "class Builder:\n    pass\n"})
        k = kinds(root)
        assert k["api_srv_server"] == "class"
        assert k["tools_build_builder"] == "class"


class TestFailure:
    def test_an_unparseable_file_does_not_take_the_good_ones_with_it(self, cs):
        root = cs({"bad.cs": "class \x00\x00 ((((", "ok.cs": "namespace D;\nclass Ok { }\n"})
        _, failed = parse_corpus_files(root)
        assert "ok.cs" not in failed
        assert "ok_ok" in kinds(root)

    def test_a_file_we_could_not_read_is_visible_as_an_empty_one(self, cs):
        # The grammar hands back an ERROR node rather than nothing for any
        # input at all, so a garbage file is never counted as a failure. What
        # must not happen is it inventing symbols, or vanishing: the file node
        # is still there with nothing under it, which is a gap you can see.
        root = cs({"bad.cs": "}}}} >>> ,,,,"})
        files, failed = parse_corpus_files(root)
        assert failed == []
        assert [n.kind for n in files[0].nodes] == ["file"]
