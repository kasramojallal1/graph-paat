"""C++, read through tree-sitter.

The point of another language is not that it works, but that it produces the
SAME shapes: a struct is a `class` node, a base class is `inherits`, a comment
run above a declaration is a `rationale`. If any of that needed a new word,
nothing downstream was ever language-neutral.

C++ adds two problems no other language here has, and both get their own tests.
There is no preprocessor, so a macro that expands to real syntax is read as
whatever it happens to look like -- and the recovery from that has to be
checked, not assumed. And this module owns `.h`, so it reads the headers of
plain C projects too; a shape that is wrong there damages a different
language's graph.
"""
import pytest

from graphpaat.parse import parse_corpus_files, parse_file
from graphpaat.resolve import resolve

pytest.importorskip("tree_sitter_cpp")


@pytest.fixture
def cppcorpus(tmp_path):
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


def lines(root):
    files, _ = parse_corpus_files(root)
    return {n.id: n.line for p in files for n in p.nodes}


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


SERVER = {"srv.cc": '''
/// A server that accepts connections.
class Server : public Base {
 public:
  Server();
  /// Starts serving.
  void Start();
  void listen();
 private:
  int fd_;
};

void Server::Start() { this->listen(); }

Server* make(int port) { return new Server(); }
'''}


class TestSameShapesAsPython:
    def test_a_class_is_a_class_node(self, cppcorpus):
        assert kinds(cppcorpus(SERVER))["srv_cc_server"] == "class"

    def test_a_struct_is_also_a_class_node(self, cppcorpus):
        root = cppcorpus({"m.cc": "struct Point { int x; int y; };\n"})
        assert kinds(root)["m_cc_point"] == "class"

    def test_a_union_is_also_a_class_node(self, cppcorpus):
        root = cppcorpus({"m.cc": "union Value { int i; float f; };\n"})
        assert kinds(root)["m_cc_value"] == "class"

    def test_an_enum_is_also_a_class_node(self, cppcorpus):
        root = cppcorpus({"m.cc": "enum class Color { Red, Green };\n"})
        assert kinds(root)["m_cc_color"] == "class"

    def test_a_type_alias_is_also_a_class_node(self, cppcorpus):
        root = cppcorpus({"m.cc": "struct Server {};\nusing Handle = Server;\n"})
        assert kinds(root)["m_cc_handle"] == "class"

    def test_a_typedef_struct_makes_one_node_not_two(self, cppcorpus):
        # C names a struct twice -- once as the tag, once as the typedef -- and
        # emitting both would mint one id for two nodes and report a collision
        # against itself.
        root = cppcorpus({"m.h": "typedef struct raxNode {\n  int size;\n} raxNode;\n"})
        files, _ = parse_corpus_files(root)
        got = [n.id for p in files for n in p.nodes if n.kind == "class"]
        assert got == ["m_h_raxnode"]

    def test_an_anonymous_typedef_takes_the_typedef_name(self, cppcorpus):
        root = cppcorpus({"m.h": "typedef struct {\n  int n;\n} Counter;\n"})
        assert kinds(root)["m_h_counter"] == "class"

    def test_methods_are_qualified_by_their_class(self, cppcorpus):
        k = kinds(cppcorpus(SERVER))
        assert k["srv_cc_server_start"] == "method"
        assert k["srv_cc_make"] == "function"

    def test_two_classes_with_a_same_named_method_do_not_collide(self, cppcorpus):
        root = cppcorpus({"m.cc": '''
struct A { void Run(); };
struct B { void Run(); };
'''})
        k = kinds(root)
        assert "m_cc_a_run" in k and "m_cc_b_run" in k

    def test_a_line_comment_run_becomes_a_rationale_node(self, cppcorpus):
        # C++ takes every comment form, unlike TypeScript. googletest documents
        # its whole public API with plain `//` runs and never writes a JSDoc
        # block, so a JSDoc-only rule would leave that codebase with no "why".
        root = cppcorpus({"m.cc": '''
struct T {
  // Opens the file.
  // Throws on failure.
  void open();
};
'''})
        assert docs(root)["m_cc_t_open#doc"] == "Opens the file. Throws on failure."

    def test_a_block_comment_becomes_a_rationale_node(self, cppcorpus):
        root = cppcorpus({"m.cc": '''
/**
 * A buffered file.
 */
class File {};
'''})
        assert docs(root)["m_cc_file#doc"] == "A buffered file."

    def test_a_comment_separated_by_a_blank_line_is_not_documentation(self, cppcorpus):
        root = cppcorpus({"m.cc": '''
// Unrelated note.

class Thing {};
'''})
        files, _ = parse_corpus_files(root)
        assert not any(n.kind == "rationale" for p in files for n in p.nodes)

    def test_a_base_class_is_inheritance(self, cppcorpus):
        root = cppcorpus({"m.cc": '''
class Base {};
class Derived : public Base {};
'''})
        assert ("m_cc_derived", "m_cc_base") in links(root, "inherits")

    def test_access_and_virtual_are_not_base_classes(self, cppcorpus):
        root = cppcorpus({"m.cc": '''
class Base {};
class Derived : virtual public Base {};
'''})
        files, _ = parse_corpus_files(root)
        derived = [n for p in files for n in p.nodes if n.id == "m_cc_derived"][0]
        assert derived.bases == ["Base"]

    def test_containment_matches_the_python_shape(self, cppcorpus):
        got = links(cppcorpus(SERVER), "contains")
        assert ("srv_cc", "srv_cc_server") in got
        assert ("srv_cc_server", "srv_cc_server_start") in got

    def test_a_nested_class_is_contained_but_keeps_a_flat_id(self, cppcorpus):
        # The resolver rebuilds a class id as prefix plus label, so a nested
        # class cannot carry its outer class in the id. Containment still says
        # where it lives.
        root = cppcorpus({"m.cc": "struct Outer {\n  struct Inner { void deep(); };\n};\n"})
        k = kinds(root)
        assert k["m_cc_inner"] == "class"
        assert k["m_cc_inner_deep"] == "method"
        assert ("m_cc_outer", "m_cc_inner") in links(root, "contains")

    def test_a_forward_declaration_is_not_a_node(self, cppcorpus):
        # `class Server;` is a promise. Emitting it would claim the id the real
        # definition needs and record the wrong line.
        root = cppcorpus({"a.h": "class Server;\n",
                          "b.cc": "class Server { public: void go(); };\n"})
        assert "a_h_server" not in kinds(root)
        assert lines(root)["b_cc_server"] == 1

    def test_a_method_records_its_owner(self, cppcorpus):
        files, _ = parse_corpus_files(cppcorpus(SERVER))
        assert files[0].owner_of["srv_cc_server_start"] == "Server"


class TestNoPreprocessor:
    """A grammar reads the source as written. These are the shapes that come
    out when a macro would have been expanded first, and each one was found in
    a real corpus rather than imagined."""

    def test_declarations_inside_an_include_guard_are_found(self, cppcorpus):
        # Every header wraps its whole contents in `#ifndef GUARD`. Stopping
        # there would read every header as empty.
        root = cppcorpus({"m.h": '''#ifndef M_H
#define M_H
struct Point { int x; };
#endif
'''})
        assert kinds(root)["m_h_point"] == "class"

    def test_a_template_declaration_is_unwrapped(self, cppcorpus):
        root = cppcorpus({"m.h": '''
template <typename T> class Box { public: T get(); };
template <typename T> T identity(T v) { return v; }
'''})
        k = kinds(root)
        assert k["m_h_box"] == "class"
        assert k["m_h_box_get"] == "method"
        assert k["m_h_identity"] == "function"

    def test_a_macro_that_opens_a_namespace_does_not_swallow_the_file(self, cppcorpus):
        # `FMT_BEGIN_NAMESPACE` expands to two `namespace` openings. Unexpanded
        # it reads as the return type of a function whose body is the rest of
        # the file, and everything after it would be lost.
        root = cppcorpus({"m.h": '''
FMT_BEGIN_NAMESPACE
namespace detail {
struct node { int n; };
void helper() {}
}
FMT_END_NAMESPACE
'''})
        k = kinds(root)
        assert k["m_h_node"] == "class"
        assert k["m_h_helper"] == "function"

    def test_an_export_macro_before_a_class_name_keeps_the_real_name(self, cppcorpus):
        # `class GTEST_API_ UnitTest {` reads as a class called GTEST_API_ and
        # a variable called UnitTest, so the class vanishes and its members
        # leak out to file scope as free functions.
        root = cppcorpus({"m.h": '''
class FMT_API File : public Base {
 public:
  void close();
};
'''})
        k = kinds(root)
        assert k["m_h_file"] == "class"
        assert k["m_h_file_close"] == "method"
        assert ("m_h_file", "m_h_file_close") in links(root, "contains")

    def test_an_export_macro_before_a_class_keeps_its_base(self, cppcorpus):
        # The base clause lands in a recovery node instead of a base_class_clause,
        # because the grammar has already given up on the name before it gets
        # there. What a class extends is part of what it is, so it is read back
        # out of the wreckage.
        root = cppcorpus({"m.h": '''
class Base {};
class FMT_API File : public Base {
 public:
  void close();
};
'''})
        assert ("m_h_file", "m_h_base") in links(root, "inherits")

    def test_a_macro_with_a_block_after_it_is_not_a_function(self, cppcorpus):
        # `TEST(HeapTest, Grows) { ... }` has no return type and an unqualified
        # name, which no real function definition can have. Kept, it puts every
        # test body in a file onto one node called TEST.
        root = cppcorpus({"m.cc": '''
TEST(HeapTest, Grows) { }
TEST(HeapTest, Shrinks) { }
void real() {}
'''})
        k = kinds(root)
        assert "m_cc_test" not in k
        assert k["m_cc_real"] == "function"

    def test_a_macro_field_in_a_struct_is_not_a_method(self, cppcorpus):
        # curl writes bitfields as `BIT(secure);`. Read as a declaration it is
        # a member function called BIT, and there are hundreds of them.
        root = cppcorpus({"m.h": '''
struct Cookie {
  BIT(secure);
  BIT(httponly);
  void real();
};
'''})
        k = kinds(root)
        assert "m_h_cookie_bit" not in k
        assert k["m_h_cookie_real"] == "method"

    def test_a_constructor_is_still_read_without_a_return_type(self, cppcorpus):
        # The rule that rejects `BIT(secure);` must not reject `Cookie();`.
        root = cppcorpus({"m.h": "struct Cookie {\n  Cookie();\n  ~Cookie();\n};\n"})
        k = kinds(root)
        assert k["m_h_cookie_cookie"] == "method"
        assert k["m_h_cookie_~cookie"] == "method"

    def test_a_function_pointer_field_is_not_a_method(self, cppcorpus):
        # `void (*cb)(void*)` has a parameter list and is a field, not a method.
        # The parentheses around the name are the only thing that says so.
        root = cppcorpus({"m.h": '''
struct Handler {
  void (*cb)(void*);
  void run();
};
'''})
        k = kinds(root)
        assert "m_h_handler_cb" not in k
        assert k["m_h_handler_run"] == "method"

    def test_an_invented_scope_does_not_become_an_owner(self, cppcorpus):
        # Recovery invents a zero-width `::` to make sense of a macro, a return
        # type and a name in a row, so `CURL_EXTERN curl_socket_t open(...)`
        # arrives as `curl_socket_t::open` and open becomes a method of a
        # socket handle.
        root = cppcorpus({"m.h": "CURL_EXTERN curl_socket_t dbg_open(int fd);\n"})
        assert not any(k == "method" for k in kinds(root).values())

    def test_a_class_the_grammar_filed_as_a_function_is_still_a_class(self, cppcorpus):
        # A class whose body ends in a way the grammar did not expect is filed
        # under `auto`, which lost fmt's basic_format_args and every class
        # written the same way.
        root = cppcorpus({"m.h": '''
template <typename Context> class basic_format_args {
 private:
  int desc_;
  union {
    const int* values_;
    const char* args_;
  };
 public:
  void get();
};
'''})
        k = kinds(root)
        assert k["m_h_basic_format_args"] == "class"
        assert k["m_h_basic_format_args_get"] == "method"


class TestResolution:
    def test_a_call_on_this_resolves_to_the_method(self, cppcorpus):
        assert ("srv_cc_server_start", "srv_cc_server_listen") in links(cppcorpus(SERVER))

    def test_an_unqualified_call_to_a_sibling_member_resolves(self, cppcorpus):
        # An unqualified call inside a member function is C++'s `self.foo()` --
        # but only when the class really has that member. Marking every bare
        # call as a self-call would refuse each one that is a free function.
        root = cppcorpus({"m.cc": '''
struct Server {
  void Start() { listen(); }
  void listen() {}
};
'''})
        assert ("m_cc_server_start", "m_cc_server_listen") in links(root)

    def test_a_bare_call_to_a_free_function_is_not_read_as_a_self_call(self, cppcorpus):
        root = cppcorpus({"m.cc": '''
void helper() {}
struct Server {
  void Start() { helper(); }
};
'''})
        assert ("m_cc_server_start", "m_cc_helper") in links(root)

    def test_a_plain_call_in_the_same_file(self, cppcorpus):
        root = cppcorpus({"m.cc": "void helper() {}\nvoid go() { helper(); }\n"})
        assert ("m_cc_go", "m_cc_helper") in links(root)

    def test_a_typed_parameter_types_its_variable(self, cppcorpus):
        root = cppcorpus({"m.cc": '''
struct Server { void Ping(); };
void use(Server* s) { s->Ping(); }
'''})
        assert ("m_cc_use", "m_cc_server_ping") in links(root)

    def test_a_local_declaration_types_its_variable(self, cppcorpus):
        root = cppcorpus({"m.cc": '''
struct Server { void Ping(); };
void use() {
  Server s;
  s.Ping();
}
'''})
        assert ("m_cc_use", "m_cc_server_ping") in links(root)

    def test_a_declared_member_type_resolves_a_call_on_it(self, cppcorpus):
        # A member's type is stated in the source, so this is a declaration and
        # not an inference.
        root = cppcorpus({"m.cc": '''
struct Buffer { void append(); };
struct Writer {
  Buffer buf_;
  void write() { buf_.append(); }
};
'''})
        assert ("m_cc_writer_write", "m_cc_buffer_append") in links(root)

    def test_a_call_through_this_on_a_member_resolves(self, cppcorpus):
        root = cppcorpus({"m.cc": '''
struct Buffer { void append(); };
struct Writer {
  Buffer buf_;
  void write() { this->buf_.append(); }
};
'''})
        assert ("m_cc_writer_write", "m_cc_buffer_append") in links(root)

    def test_a_new_expression_is_a_call_to_the_class(self, cppcorpus):
        root = cppcorpus({"m.cc": "struct Server {};\nServer* make() { return new Server(); }\n"})
        assert ("m_cc_make", "m_cc_server") in links(root)

    def test_an_out_of_line_definition_is_owned_by_its_class(self, cppcorpus):
        root = cppcorpus({"srv.h": "class Server {\n public:\n  void Start();\n  void listen();\n};\n",
                          "srv.cc": "void Server::Start() { this->listen(); }\n"
                                    "void Server::listen() {}\n"})
        k = kinds(root)
        assert k["srv_cc_server_start"] == "method"
        assert ("srv_cc_server_start", "srv_cc_server_listen") in links(root)

    def test_a_method_defined_out_of_line_hangs_off_the_file_not_a_missing_class(self, cppcorpus):
        # The class lives in another file, so a containment edge to a class node
        # this file never created would be a link that points at nothing.
        root = cppcorpus({"srv.h": "class Server { public: void Start(); };\n",
                          "srv.cc": "void Server::Start() {}\n"})
        assert ("srv_cc", "srv_cc_server_start") in links(root, "contains")

    def test_a_call_on_an_untyped_receiver_is_refused_not_guessed(self, cppcorpus):
        root = cppcorpus({"m.cc": '''
struct Server { void Ping(); };
void use(void* thing) { thing->Ping(); }
'''})
        assert reasons_for(root)["receiver type unknown"] >= 1

    def test_a_free_prototype_is_not_a_node_so_a_call_still_resolves(self, cppcorpus):
        # A prototype and its definition would be two nodes with one name, and
        # the resolver refuses a name it finds in two places -- so keeping the
        # prototype turns every cross-file call into a gap.
        root = cppcorpus({"util.h": "void listRelease(int n);\n",
                          "a.cc": "void listRelease(int n) {}\n",
                          "b.cc": "void go() { listRelease(1); }\n"})
        assert "util_h_listrelease" not in kinds(root)
        assert ("b_cc_go", "a_cc_listrelease") in links(root)

    def test_a_member_declaration_in_a_header_is_kept(self, cppcorpus):
        # The opposite case: a C++ class lists its methods in the header and
        # defines them in the source. Dropping those leaves the class empty.
        root = cppcorpus({"srv.h": "class Server { public: void Start(); };\n"})
        assert kinds(root)["srv_h_server_start"] == "method"

    def test_a_quoted_include_resolves_against_the_including_directory(self, cppcorpus):
        root = cppcorpus({"lib/url.cc": '#include "curl_setup.h"\n',
                          "lib/curl_setup.h": "struct Setup { int n; };\n"})
        assert ("lib_url_cc", "lib_curl_setup_h") in links(root, "imports")

    def test_a_relative_include_is_normalised(self, cppcorpus):
        root = cppcorpus({"lib/vtls/tls.cc": '#include "../urldata.h"\n',
                          "lib/urldata.h": "struct Data { int n; };\n"})
        assert ("lib_vtls_tls_cc", "lib_urldata_h") in links(root, "imports")

    def test_a_system_include_stays_unresolved(self, cppcorpus):
        # Which libraries a file depends on is worth answering, and answering
        # "nothing" would be a lie.
        root = cppcorpus({"m.cc": "#include <vector>\nvoid go() {}\n"})
        files, _ = parse_corpus_files(root)
        edges, _ = resolve(files)
        assert any(e.relation == "imports" and not e.resolved and e.target == "?vector"
                   for e in edges)

    def test_the_extension_is_part_of_a_files_identity(self, cppcorpus):
        # `url.c` and `url.h` are two files. Dropping the extension would merge
        # a third of a real C corpus into shared ids.
        root = cppcorpus({"url.h": "struct A { int n; };\n",
                          "url.cc": "struct B { int n; };\n"})
        k = kinds(root)
        assert k["url_h"] == "file" and k["url_cc"] == "file"

    def test_resolved_edges_never_dangle(self, cppcorpus):
        root = cppcorpus(dict(SERVER, **{
            "other.cc": '#include "srv.h"\nvoid use(Server* s) { s->Start(); }\n',
            "srv.h": "class Server { public: void Start(); };\n"}))
        files, _ = parse_corpus_files(root)
        ids = {n.id for p in files for n in p.nodes}
        edges, _ = resolve(files)
        for e in edges:
            if e.resolved:
                assert e.target in ids


class TestCHeaders:
    """This module owns `.h`, so it reads plain C headers too. A shape that is
    wrong here damages a C project's graph, not a C++ one."""

    def test_a_c_header_yields_its_types(self, cppcorpus):
        root = cppcorpus({"rax.h": '''
/* A radix tree node. */
typedef struct raxNode {
  uint32_t iskey;
} raxNode;

typedef int (*raxNodeCallback)(raxNode **noderef);
'''})
        k = kinds(root)
        assert k["rax_h_raxnode"] == "class"
        assert k["rax_h_raxnodecallback"] == "class"
        assert docs(root)["rax_h_raxnode#doc"] == "A radix tree node."

    def test_a_c_header_contributes_no_methods(self, cppcorpus):
        # C has no methods. If any appear from a header, something was misread.
        root = cppcorpus({"h.h": '''
struct connectdata {
  int (*done)(void *data);
  char *name;
};
void Curl_close(struct connectdata *c);
'''})
        assert not any(k == "method" for k in kinds(root).values())


class TestMixedCorpus:
    def test_cpp_and_python_in_one_repository(self, cppcorpus):
        root = cppcorpus({"api/srv.cc": "class Server { public: void go(); };\n",
                          "tools/build.py": "class Builder:\n    pass\n"})
        k = kinds(root)
        assert k["api_srv_cc_server"] == "class"
        assert k["tools_build_builder"] == "class"


class TestFailure:
    def test_a_readable_file_is_not_reported_as_failed(self, cppcorpus):
        root = cppcorpus({"bad.cc": "class \x00\x00 ((((", "ok.cc": "void go() {}\n"})
        _, failed = parse_corpus_files(root)
        assert "ok.cc" not in failed

    def test_a_file_that_yields_no_tree_at_all_is_reported(self, cppcorpus):
        # C++ recovery is total: the grammar produces a tree for almost any
        # input, so this is the only case that counts as a failure. A file with
        # a syntax error in one function is not one -- it still has a map.
        root = cppcorpus({"empty.cc": ""})
        parsed = parse_file(root / "empty.cc", root)
        assert parsed is not None and len(parsed.nodes) == 1
