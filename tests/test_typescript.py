"""TypeScript, read through tree-sitter.

Same test as Go: does a third language produce the SAME shapes, without
anything downstream learning it exists? TypeScript stresses that harder than Go
did, because it has three ways to name a type and two ways to declare a
function.
"""
import pytest

from graphpaat.parse import parse_corpus_files
from graphpaat.resolve import resolve

pytest.importorskip("tree_sitter_typescript")


@pytest.fixture
def ts(tmp_path):
    def build(files: dict[str, str]):
        root = tmp_path / "repo"
        for name, source in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
        root.mkdir(parents=True, exist_ok=True)
        return root
    return build


def nodes_of(root):
    files, _ = parse_corpus_files(root)
    return {n.id: n for p in files for n in p.nodes}


def links(root, relation="calls"):
    files, _ = parse_corpus_files(root)
    edges, _ = resolve(files)
    edges = list(edges) + [e for p in files for e in p.edges]
    return {(e.source, e.target) for e in edges
            if e.relation == relation and e.resolved}


SERVER = {"srv.ts": '''import { Base } from "./base";

/** A server that accepts connections. */
export class Server extends Base implements Handler {
  private addr: string;
  constructor(addr: string) { super(); this.addr = addr; }
  start(): void { this.listen(); }
  listen(): void {}
}

export interface Handler { handle(): void }

export type Options = { port: number };

export function helper(x: number): string { return ""; }

export const make = (addr: string) => new Server(addr);
''', "base.ts": "export class Base {}\n"}


class TestSameShapesAsPythonAndGo:
    def test_class_interface_and_type_alias_are_all_class_nodes(self, ts):
        n = nodes_of(ts(SERVER))
        assert n["srv_server"].kind == "class"
        assert n["srv_handler"].kind == "class"
        assert n["srv_options"].kind == "class"

    def test_methods_are_qualified_by_their_class(self, ts):
        n = nodes_of(ts(SERVER))
        assert n["srv_server_start"].kind == "method"
        assert n["srv_server_listen"].kind == "method"

    def test_two_classes_with_a_same_named_method_do_not_collide(self, ts):
        root = ts({"m.ts": "class A { run() {} }\nclass B { run() {} }\n"})
        n = nodes_of(root)
        assert "m_a_run" in n and "m_b_run" in n

    def test_an_exported_declaration_is_unwrapped(self, ts):
        # `export class Foo` wraps the declaration; not unwrapping it made
        # every exported symbol in a modern file invisible.
        assert "srv_helper" in nodes_of(ts(SERVER))

    def test_a_const_arrow_function_is_a_function(self, ts):
        # Most functions in modern TypeScript are written this way.
        assert nodes_of(ts(SERVER))["srv_make"].kind == "function"

    def test_extends_and_implements_are_both_inheritance(self, ts):
        got = links(ts(SERVER), "inherits")
        assert ("srv_server", "base_base") in got
        assert ("srv_server", "srv_handler") in got

    def test_a_jsdoc_block_becomes_a_claim(self, ts):
        n = nodes_of(ts(SERVER))
        assert "accepts connections" in n["srv_server#doc"].text

    def test_a_line_comment_is_not_documentation(self, ts):
        # `//` usually annotates a line, not the declaration. Treating every
        # one as documentation buried the real ones.
        root = ts({"m.ts": "// bump this later\nexport class Thing {}\n"})
        assert not any(n.kind == "rationale" for n in nodes_of(root).values())

    def test_an_overload_signature_does_not_duplicate_its_implementation(self, ts):
        # Bodyless overloads parse as function_signature; only the
        # implementation is a declaration, and it is the one real definition.
        root = ts({"m.ts": '''export function f(a: string): void;
export function f(a: number): void;
export function f(a: any): void { }
'''})
        assert len([n for n in nodes_of(root).values() if n.label == "f"]) == 1


class TestResolution:
    def test_a_call_on_this_resolves_within_the_class(self, ts):
        assert ("srv_server_start", "srv_server_listen") in links(ts(SERVER))

    def test_an_annotated_parameter_types_its_variable(self, ts):
        root = ts({"m.ts": '''class Server { ping(): void {} }
export function use(s: Server) { s.ping(); }
'''})
        assert ("m_use", "m_server_ping") in links(root)

    def test_a_new_expression_types_its_variable(self, ts):
        root = ts({"m.ts": '''class Server { ping(): void {} }
export function use() {
  const s = new Server();
  s.ping();
}
'''})
        assert ("m_use", "m_server_ping") in links(root)

    def test_constructing_a_class_is_a_call_to_it(self, ts):
        # Often the only edge tying a factory to what it builds.
        assert ("srv_make", "srv_server") in links(ts(SERVER))

    def test_an_untyped_receiver_is_refused_not_guessed(self, ts):
        root = ts({"m.ts": '''class Server { ping(): void {} }
export function use(thing: any) { thing.ping(); }
'''})
        files, _ = parse_corpus_files(root)
        assert resolve(files)[1]["receiver type unknown"] >= 1

    def test_a_relative_import_resolves_to_its_file(self, ts):
        assert ("srv", "base") in links(ts(SERVER), "imports")

    def test_resolved_edges_never_dangle(self, ts):
        files, _ = parse_corpus_files(ts(SERVER))
        ids = {n.id for p in files for n in p.nodes}
        edges, _ = resolve(files)
        for e in edges:
            if e.resolved:
                assert e.target in ids


class TestDeclarationFiles:
    def test_a_d_ts_file_is_never_parsed(self, ts):
        # It restates the module beside it and mints identical ids, exactly as
        # Python's .pyi stubs did.
        root = ts({"m.ts": "export class Thing {}\n",
                   "m.d.ts": "export declare class Thing {}\n"})
        files, _ = parse_corpus_files(root)
        assert [p.path for p in files] == ["m.ts"]


class TestTsx:
    def test_a_tsx_file_parses_with_the_jsx_grammar(self, ts):
        # In .tsx, `<T>` is JSX rather than a cast, so it needs its own grammar.
        root = ts({"c.tsx": '''export function Button() {
  return <div className="x">hi</div>;
}
'''})
        assert "c_button" in nodes_of(root)


class TestThreeLanguagesTogether:
    def test_one_repository_can_hold_all_three(self, ts):
        root = ts({"web/app.ts": "export class App {}\n",
                   "api/srv.go": "package api\ntype Server struct{}\n",
                   "tools/build.py": "class Builder:\n    pass\n"})
        n = nodes_of(root)
        assert n["web_app_app"].kind == "class"
        assert n["api_srv_server"].kind == "class"
        assert n["tools_build_builder"].kind == "class"
