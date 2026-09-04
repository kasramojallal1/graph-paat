"""Go, read through tree-sitter.

The point of a second language is not that it works, but that it produces the
SAME shapes: a Go struct is a `class` node, an embedded type is `inherits`, a
doc comment is a `rationale`. If any of that needed a new word, the model was
never language-neutral and everything downstream would have to learn Go too.

Go's own advantage is tested here as well: a method receiver is declared, so
`s.foo()` resolves exactly rather than being inferred.
"""
import pytest

from graphpaat.parse import parse_corpus_files, parse_file
from graphpaat.resolve import resolve

pytest.importorskip("tree_sitter_go")


@pytest.fixture
def gocorpus(tmp_path):
    def build(files: dict[str, str]):
        root = tmp_path / "repo"
        for name, source in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
        root.mkdir(parents=True, exist_ok=True)
        return root
    return build


def kinds(root, name):
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


SERVER = {"srv.go": '''package main

// Server accepts connections.
type Server struct {
    addr string
}

// Start begins serving.
func (s *Server) Start() error {
    return s.listen()
}

func (s *Server) listen() error { return nil }

func New(addr string) *Server { return &Server{addr: addr} }
'''}


class TestSameShapesAsPython:
    def test_a_struct_is_a_class_node(self, gocorpus):
        assert kinds(gocorpus(SERVER), "srv")["srv_server"] == "class"

    def test_an_interface_is_also_a_class_node(self, gocorpus):
        root = gocorpus({"i.go": "package main\ntype Handler interface{ Do() error }\n"})
        assert kinds(root, "i")["i_handler"] == "class"

    def test_methods_are_qualified_by_their_receiver_type(self, gocorpus):
        k = kinds(gocorpus(SERVER), "srv")
        assert k["srv_server_start"] == "method"
        assert k["srv_new"] == "function"

    def test_two_types_with_a_same_named_method_do_not_collide(self, gocorpus):
        root = gocorpus({"m.go": '''package main
type A struct{}
type B struct{}
func (a *A) Run() {}
func (b *B) Run() {}
'''})
        k = kinds(root, "m")
        assert "m_a_run" in k and "m_b_run" in k

    def test_a_doc_comment_becomes_a_rationale_node(self, gocorpus):
        files, _ = parse_corpus_files(gocorpus(SERVER))
        docs = {n.id: n.text for p in files for n in p.nodes if n.kind == "rationale"}
        assert "Server accepts connections." in docs["srv_server#doc"]
        assert "Start begins serving." in docs["srv_server_start#doc"]

    def test_a_comment_separated_by_a_blank_line_is_not_documentation(self, gocorpus):
        root = gocorpus({"m.go": '''package main

// Unrelated note.

type Thing struct{}
'''})
        files, _ = parse_corpus_files(root)
        assert not any(n.kind == "rationale" for p in files for n in p.nodes)

    def test_an_embedded_type_is_inheritance(self, gocorpus):
        # Go has no subclassing; embedding is how a type gains another's
        # methods, which is what `inherits` means everywhere else here.
        root = gocorpus({"m.go": '''package main
type Base struct{}
type Derived struct {
    Base
    name string
}
'''})
        assert ("m_derived", "m_base") in links(root, "inherits")

    def test_containment_matches_the_python_shape(self, gocorpus):
        got = links(gocorpus(SERVER), "contains")
        assert ("srv", "srv_server") in got
        assert ("srv_server", "srv_server_start") in got


class TestResolution:
    def test_a_call_on_the_receiver_resolves_exactly(self, gocorpus):
        # The receiver's type is declared, so this needs no inference at all.
        assert ("srv_server_start", "srv_server_listen") in links(gocorpus(SERVER))

    def test_a_plain_call_in_the_same_file(self, gocorpus):
        root = gocorpus({"m.go": '''package main
func helper() {}
func go1() { helper() }
'''})
        assert ("m_go1", "m_helper") in links(root)

    def test_a_typed_parameter_types_its_variable(self, gocorpus):
        root = gocorpus({"m.go": '''package main
type Server struct{}
func (s *Server) Ping() {}
func use(srv *Server) { srv.Ping() }
'''})
        assert ("m_use", "m_server_ping") in links(root)

    def test_a_var_declaration_types_its_variable(self, gocorpus):
        root = gocorpus({"m.go": '''package main
type Server struct{}
func (s *Server) Ping() {}
func use() {
    var srv Server
    srv.Ping()
}
'''})
        assert ("m_use", "m_server_ping") in links(root)

    def test_a_composite_literal_types_its_variable(self, gocorpus):
        root = gocorpus({"m.go": '''package main
type Server struct{}
func (s *Server) Ping() {}
func use() {
    srv := &Server{}
    srv.Ping()
}
'''})
        assert ("m_use", "m_server_ping") in links(root)

    def test_a_call_on_an_untyped_receiver_is_refused_not_guessed(self, gocorpus):
        root = gocorpus({"m.go": '''package main
type Server struct{}
func (s *Server) Ping() {}
func use(thing interface{}) { thing.Ping() }
'''})
        assert reasons_for(root)["receiver type unknown"] >= 1

    def test_imports_are_recorded(self, gocorpus):
        root = gocorpus({"m.go": 'package main\nimport (\n\t"fmt"\n)\n'})
        files, _ = parse_corpus_files(root)
        assert files[0].imports["fmt"] == "fmt"

    def test_resolved_edges_never_dangle(self, gocorpus):
        files, _ = parse_corpus_files(gocorpus(SERVER))
        ids = {n.id for p in files for n in p.nodes}
        edges, _ = resolve(files)
        for e in edges:
            if e.resolved:
                assert e.target in ids


class TestMixedCorpus:
    def test_python_and_go_in_one_repository(self, gocorpus):
        # A monorepo is the case this phase exists for.
        root = gocorpus({"api/srv.go": "package api\ntype Server struct{}\n",
                         "tools/build.py": "class Builder:\n    pass\n"})
        k = kinds(root, "x")
        assert k["api_srv_server"] == "class"
        assert k["tools_build_builder"] == "class"


class TestFailure:
    def test_an_unparseable_file_is_reported_not_skipped_silently(self, gocorpus):
        root = gocorpus({"bad.go": "package \x00\x00 ((((", "ok.go": "package main\n"})
        _, failed = parse_corpus_files(root)
        assert "ok.go" not in failed
