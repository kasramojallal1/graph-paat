"""PHP, read through tree-sitter.

The point is not that PHP works, but that it produces the SAME shapes: a class,
an interface, a trait and an enum are all `class` nodes, `extends`,
`implements` and a used trait are all `inherits`, and a `/** */` block is a
`rationale`. If any of that needed a new word, everything downstream would have
to learn PHP too.

PHP's own advantages are tested here as well -- a property's type is declared,
a static call names its class, and `use Foo\\Bar\\Baz;` names the file Baz lives
in -- along with the case where a namespaced reference is refused rather than
flattened to its last segment.
"""
import pytest

from graphpaat.parse import parse_corpus_files, parse_file
from graphpaat.resolve import resolve

pytest.importorskip("tree_sitter_php")


@pytest.fixture
def phpcorpus(tmp_path):
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


SERVER = {"Server.php": '''<?php

namespace App;

/**
 * Accepts connections.
 */
class Server
{
    protected Logger $logger;

    /**
     * Begins serving.
     */
    public function start(): void
    {
        $this->listen();
        $this->logger->write("up");
    }

    private function listen(): void
    {
    }
}
'''}


class TestSameShapesAsPython:
    def test_a_class_is_a_class_node(self, phpcorpus):
        assert kinds(phpcorpus(SERVER))["server_server"] == "class"

    def test_an_interface_is_also_a_class_node(self, phpcorpus):
        root = phpcorpus({"H.php": "<?php\ninterface Handler { public function go(): void; }\n"})
        assert kinds(root)["h_handler"] == "class"

    def test_a_trait_is_also_a_class_node(self, phpcorpus):
        root = phpcorpus({"T.php": "<?php\ntrait Loggy { public function log($m) {} }\n"})
        assert kinds(root)["t_loggy"] == "class"

    def test_an_enum_is_also_a_class_node(self, phpcorpus):
        root = phpcorpus({"S.php": '''<?php
enum Suit: string {
    case Hearts = 'H';
    public function colour(): string { return "red"; }
}
'''})
        k = kinds(root)
        assert k["s_suit"] == "class"
        assert k["s_suit_colour"] == "method"

    def test_methods_are_qualified_by_their_class(self, phpcorpus):
        k = kinds(phpcorpus(SERVER))
        assert k["server_server_start"] == "method"
        assert k["server_server_listen"] == "method"

    def test_a_free_function_is_a_function_node(self, phpcorpus):
        root = phpcorpus({"F.php": "<?php\nfunction helper(): void {}\n"})
        assert kinds(root)["f_helper"] == "function"

    def test_two_classes_with_a_same_named_method_do_not_collide(self, phpcorpus):
        root = phpcorpus({"A.php": "<?php\nclass A { public function run() {} }\n",
                          "B.php": "<?php\nclass B { public function run() {} }\n"})
        k = kinds(root)
        assert "a_a_run" in k and "b_b_run" in k

    def test_a_docblock_becomes_a_rationale_node(self, phpcorpus):
        found = docs(phpcorpus(SERVER))
        assert found["server_server#doc"] == "Accepts connections."
        assert found["server_server_start#doc"] == "Begins serving."

    def test_a_line_comment_is_not_documentation(self, phpcorpus):
        # `//` above a declaration is a remark about the next line far more
        # often than it is documentation; only a docblock counts.
        root = phpcorpus({"M.php": "<?php\n// just a note\nclass Thing {}\n"})
        files, _ = parse_corpus_files(root)
        assert not any(n.kind == "rationale" for p in files for n in p.nodes)

    def test_a_docblock_separated_by_a_blank_line_is_not_documentation(self, phpcorpus):
        root = phpcorpus({"M.php": "<?php\n/** Unrelated. */\n\nclass Thing {}\n"})
        files, _ = parse_corpus_files(root)
        assert not any(n.kind == "rationale" for p in files for n in p.nodes)

    def test_containment_matches_the_python_shape(self, phpcorpus):
        got = links(phpcorpus(SERVER), "contains")
        assert ("server", "server_server") in got
        assert ("server_server", "server_server_start") in got

    def test_a_declaration_inside_a_braced_namespace_is_still_found(self, phpcorpus):
        # `namespace A\B { ... }` is the one wrapper PHP puts declarations in;
        # without looking through it the whole file reads as empty.
        root = phpcorpus({"N.php": "<?php\nnamespace A\\B {\n    class Boxed {}\n}\n"})
        assert kinds(root)["n_boxed"] == "class"

    def test_an_attributed_class_is_filed_at_its_own_line(self, phpcorpus):
        # `#[Attr]` is part of the class node, so the declaration starts a line
        # early; a reader following the line number must land on `class`.
        root = phpcorpus({"M.php": "<?php\n/** Doc. */\n#[Attr]\nclass Marked {}\n"})
        files, _ = parse_corpus_files(root)
        node = [n for p in files for n in p.nodes if n.id == "m_marked"][0]
        assert node.line == 4
        assert docs(root)["m_marked#doc"] == "Doc."


class TestInheritance:
    def test_extends_is_inheritance(self, phpcorpus):
        root = phpcorpus({"Base.php": "<?php\nclass Base {}\n",
                          "Sub.php": "<?php\nclass Sub extends Base {}\n"})
        assert ("sub_sub", "base_base") in links(root, "inherits")

    def test_implements_is_inheritance_too(self, phpcorpus):
        # PHP separates the two; the graph does not, because both mean the type
        # gains the other's shape.
        root = phpcorpus({"H.php": "<?php\ninterface Handler {}\n",
                          "S.php": "<?php\nclass Srv implements Handler {}\n"})
        assert ("s_srv", "h_handler") in links(root, "inherits")

    def test_a_used_trait_is_inheritance(self, phpcorpus):
        # A trait is copied into the class wholesale, so leaving it out hides
        # where half a trait-heavy class's methods come from.
        root = phpcorpus({"T.php": "<?php\ntrait Loggy { public function log($m) {} }\n",
                          "S.php": "<?php\nclass Srv { use Loggy; }\n"})
        assert ("s_srv", "t_loggy") in links(root, "inherits")

    def test_an_interface_extending_two_interfaces_records_both(self, phpcorpus):
        root = phpcorpus({"A.php": "<?php\ninterface A {}\n",
                          "B.php": "<?php\ninterface B {}\n",
                          "C.php": "<?php\ninterface C extends A, B {}\n"})
        got = links(root, "inherits")
        assert ("c_c", "a_a") in got and ("c_c", "b_b") in got


class TestResolution:
    def test_a_call_on_this_resolves_to_the_same_class(self, phpcorpus):
        assert ("server_server_start", "server_server_listen") in links(phpcorpus(SERVER))

    def test_a_typed_property_resolves_a_call_on_this(self, phpcorpus):
        # `protected Logger $logger;` is declared, not inferred -- the single
        # biggest reason PHP resolves better than Python here.
        root = phpcorpus(dict(SERVER,
                              **{"Logger.php": "<?php\nclass Logger { public function write($m) {} }\n"}))
        assert ("server_server_start", "logger_logger_write") in links(root)

    def test_a_promoted_constructor_parameter_is_a_property(self, phpcorpus):
        root = phpcorpus({"Logger.php": "<?php\nclass Logger { public function write($m) {} }\n",
                          "Srv.php": '''<?php
class Srv {
    public function __construct(private Logger $logger) {}
    public function go() { $this->logger->write("x"); }
}
'''})
        assert ("srv_srv_go", "logger_logger_write") in links(root)

    def test_a_typed_parameter_types_its_variable(self, phpcorpus):
        root = phpcorpus({"Logger.php": "<?php\nclass Logger { public function write($m) {} }\n",
                          "F.php": '<?php\nfunction use_it(Logger $l) { $l->write("x"); }\n'})
        assert ("f_use_it", "logger_logger_write") in links(root)

    def test_a_constructor_call_types_its_variable(self, phpcorpus):
        root = phpcorpus({"Logger.php": "<?php\nclass Logger { public function write($m) {} }\n",
                          "F.php": '<?php\nfunction go() { $l = new Logger(); $l->write("x"); }\n'})
        assert ("f_go", "logger_logger_write") in links(root)

    def test_new_is_a_call_to_the_class(self, phpcorpus):
        # Often the only edge tying a factory to what it builds.
        root = phpcorpus({"Logger.php": "<?php\nclass Logger {}\n",
                          "F.php": "<?php\nfunction make() { return new Logger(); }\n"})
        assert ("f_make", "logger_logger") in links(root)

    def test_a_static_call_names_its_class(self, phpcorpus):
        root = phpcorpus({"Utils.php": "<?php\nclass Utils { public static function slug($s) {} }\n",
                          "F.php": '<?php\nfunction go() { Utils::slug("a"); }\n'})
        assert ("f_go", "utils_utils_slug") in links(root)

    def test_self_and_static_resolve_to_the_same_class(self, phpcorpus):
        root = phpcorpus({"U.php": '''<?php
class U {
    public static function outer() { self::inner(); static::other(); }
    private static function inner() {}
    private static function other() {}
}
'''})
        got = links(root)
        assert ("u_u_outer", "u_u_inner") in got
        assert ("u_u_outer", "u_u_other") in got

    def test_parent_resolves_to_the_base_class(self, phpcorpus):
        root = phpcorpus({"Base.php": "<?php\nclass Base { public function __construct() {} }\n",
                          "Sub.php": '''<?php
class Sub extends Base {
    public function __construct() { parent::__construct(); }
}
'''})
        assert ("sub_sub___construct", "base_base___construct") in links(root)

    def test_an_import_names_the_file_the_class_lives_in(self, phpcorpus):
        # `use App\Routing\Resolver;` is App/Routing/Resolver.php, so the whole
        # path names a module and the import edge points at that file.
        root = phpcorpus({"Routing/Resolver.php": "<?php\nnamespace App\\Routing;\nclass Resolver {}\n",
                          "App.php": '''<?php
namespace App;

use App\\Routing\\Resolver;

class App
{
    public function go() { return new Resolver(); }
}
'''})
        assert ("app", "routing_resolver") in links(root, "imports")
        assert ("app_app_go", "routing_resolver_resolver") in links(root)

    def test_an_interface_method_is_a_node_a_call_can_land_on(self, phpcorpus):
        root = phpcorpus({"HandlerI.php": '''<?php
interface HandlerI { public function handle(): void; }
''',
                          "Srv.php": '''<?php
class Srv {
    protected HandlerI $h;
    public function go() { $this->h->handle(); }
}
'''})
        assert ("srv_srv_go", "handleri_handleri_handle") in links(root)

    def test_a_call_on_an_untyped_property_is_refused_not_guessed(self, phpcorpus):
        root = phpcorpus({"Srv.php": '''<?php
class Srv {
    protected $h;
    public function go() { $this->h->handle(); }
}
'''})
        assert reasons_for(root)["receiver is a self attribute of unknown type"] >= 1

    def test_a_call_on_a_chained_expression_is_refused_not_guessed(self, phpcorpus):
        # `$x->body()->read()` must not be mistaken for a plain `read()`; the
        # resolver would bind that to any corpus function of the name.
        root = phpcorpus({"R.php": "<?php\nfunction read() {}\n",
                          "F.php": "<?php\nfunction go($x) { $x->body()->read(); }\n"})
        assert reasons_for(root)["receiver type unknown"] >= 1
        assert ("f_go", "r_read") not in links(root)

    def test_a_namespaced_static_call_is_refused_not_flattened(self, phpcorpus):
        # `Psr7\Utils::` is not `Utils::`. Taking the last segment aims the call
        # at an unrelated class that happens to share a short name.
        root = phpcorpus({"Utils.php": "<?php\nclass Utils { public static function slug($s) {} }\n",
                          "F.php": '<?php\nfunction go() { Psr7\\Utils::slug("a"); }\n'})
        assert ("f_go", "utils_utils_slug") not in links(root)
        assert reasons_for(root)["receiver type unknown"] >= 1

    def test_a_union_typed_property_is_refused_not_guessed(self, phpcorpus):
        root = phpcorpus({"A.php": "<?php\nclass A { public function go() {} }\n",
                          "B.php": "<?php\nclass B { public function go() {} }\n",
                          "S.php": '''<?php
class S {
    protected A|B $x;
    public function run() { $this->x->go(); }
}
'''})
        assert reasons_for(root)["receiver is a self attribute of unknown type"] >= 1
        assert ("s_s_run", "a_a_go") not in links(root)

    def test_a_global_function_import_names_no_module(self, phpcorpus):
        # `use function sprintf;` names a global function, not a file. Recording
        # it as a module would add a gap that can never close, and would match a
        # same-named file in the corpus if one existed.
        root = phpcorpus({"F.php": "<?php\nuse function sprintf;\nfunction go() {}\n"})
        files, _ = parse_corpus_files(root)
        assert files[0].import_sites == []

    def test_imports_are_recorded(self, phpcorpus):
        root = phpcorpus({"F.php": "<?php\nuse App\\Routing\\Resolver;\n"})
        files, _ = parse_corpus_files(root)
        assert files[0].import_sites == [("App.Routing.Resolver", 2)]

    def test_a_grouped_import_records_every_name(self, phpcorpus):
        root = phpcorpus({"F.php": "<?php\nuse App\\{Alpha, Beta};\n"})
        files, _ = parse_corpus_files(root)
        assert sorted(files[0].import_sites) == [("App.Alpha", 2), ("App.Beta", 2)]

    def test_resolved_edges_never_dangle(self, phpcorpus):
        root = phpcorpus(dict(SERVER,
                              **{"Logger.php": "<?php\nclass Logger { public function write($m) {} }\n"}))
        files, _ = parse_corpus_files(root)
        ids = {n.id for p in files for n in p.nodes}
        edges, _ = resolve(files)
        for e in edges:
            if e.resolved:
                assert e.target in ids


class TestMixedCorpus:
    def test_python_and_php_in_one_repository(self, phpcorpus):
        root = phpcorpus({"api/Srv.php": "<?php\nnamespace Api;\nclass Srv {}\n",
                          "tools/build.py": "class Builder:\n    pass\n"})
        k = kinds(root)
        assert k["api_srv_srv"] == "class"
        assert k["tools_build_builder"] == "class"


class TestFailure:
    def test_a_file_of_junk_costs_only_itself_and_stays_visible(self, phpcorpus):
        """PHP never reports a whole-file failure, and this records why.

        Everything outside `<?php` is text, so the grammar wraps bytes it
        cannot read in an ERROR node instead of refusing the file: a file of
        junk parses to a root that HAS children, which is the condition for
        counting a failure. The loss is still visible rather than silent --
        the file keeps its own node and simply contains nothing -- and the rest
        of the corpus is untouched. If that guard is ever widened, this test is
        the one that should change first.
        """
        root = phpcorpus({"bad.php": "\x00\x00\x00", "ok.php": "<?php\nclass Ok {}\n"})
        files, failed = parse_corpus_files(root)
        assert failed == []
        junk = [p for p in files if p.path == "bad.php"][0]
        assert [n.kind for n in junk.nodes] == ["file"]
        assert kinds(root)["ok_ok"] == "class"

    def test_one_broken_function_does_not_lose_the_rest_of_the_file(self, phpcorpus):
        # A syntax error inside one body must not cost the whole file; that is
        # the difference between a gap and a blackout.
        root = phpcorpus({"M.php": '''<?php
class M {
    public function good() {}
    public function broken() { $x = = ; }
}
'''})
        _, failed = parse_corpus_files(root)
        assert failed == []
        assert kinds(root)["m_m_good"] == "method"
