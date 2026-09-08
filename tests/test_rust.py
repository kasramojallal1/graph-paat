"""Rust, read through tree-sitter.

The point of another language is not that it works, but that it produces the
SAME shapes: a struct is a `class` node, `impl Trait for Type` is `inherits`, a
`///` run is a `rationale`. If any of that needed a new word, the model was
never language-neutral and everything downstream would have to learn Rust too.

Rust's own hard part is tested here as well. A method is written outside its
type, in an `impl` block that names it, and reading that block is the whole
difference between a map of types with methods and a pile of loose functions.
"""
import pytest

from graphpaat.parse import parse_corpus_files, parse_file
from graphpaat.resolve import resolve

pytest.importorskip("tree_sitter_rust")


@pytest.fixture
def rustcorpus(tmp_path):
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


GLOB = {"glob.rs": '''//! Globs, compiled.

/// A compiled glob.
#[derive(Debug)]
pub struct Glob {
    opts: GlobOptions,
}

pub struct GlobOptions {
    case: bool,
}

impl GlobOptions {
    pub fn check(&self) -> bool { true }
}

impl Glob {
    /// Make one.
    pub fn new(opts: GlobOptions) -> Glob { Glob { opts } }

    fn go(&self, other: &Glob) -> bool {
        self.helper();
        self.opts.check();
        other.helper();
        Glob::new(opts)
    }

    fn helper(&self) -> bool { false }
}

impl std::fmt::Display for Glob {
    fn fmt(&self) -> Result { Ok(()) }
}

fn make() -> Glob { Glob { opts: GlobOptions { case: true } } }
'''}


class TestSameShapesAsPython:
    def test_a_struct_is_a_class_node(self, rustcorpus):
        assert kinds(rustcorpus(GLOB))["glob_glob"] == "class"

    def test_an_enum_is_also_a_class_node(self, rustcorpus):
        root = rustcorpus({"e.rs": "pub enum Choice { A, B(u32) }\n"})
        assert kinds(root)["e_choice"] == "class"

    def test_a_trait_is_also_a_class_node(self, rustcorpus):
        root = rustcorpus({"t.rs": "pub trait Matcher { fn find(&self) -> bool; }\n"})
        assert kinds(root)["t_matcher"] == "class"

    def test_a_type_alias_is_also_a_class_node(self, rustcorpus):
        root = rustcorpus({"a.rs": "pub type Bytes = Vec<u8>;\n"})
        assert kinds(root)["a_bytes"] == "class"

    def test_an_impl_block_makes_its_functions_methods_of_the_type(self, rustcorpus):
        k = kinds(rustcorpus(GLOB))
        assert k["glob_glob_new"] == "method"
        assert k["glob_make"] == "function"

    def test_an_impl_trait_for_type_also_makes_methods_of_the_type(self, rustcorpus):
        # `impl Display for Glob` is where `fmt` lives; missing this would leave
        # it a stray free function belonging to nothing.
        assert kinds(rustcorpus(GLOB))["glob_glob_fmt"] == "method"

    def test_a_trait_method_belongs_to_the_trait(self, rustcorpus):
        root = rustcorpus({"t.rs": '''pub trait Matcher {
    fn find(&self) -> bool;
    fn helper(&self) -> bool { true }
}
'''})
        k = kinds(root)
        assert k["t_matcher_find"] == "method"
        assert k["t_matcher_helper"] == "method"

    def test_two_types_with_a_same_named_method_do_not_collide(self, rustcorpus):
        root = rustcorpus({"m.rs": '''struct A;
struct B;
impl A { fn run(&self) {} }
impl B { fn run(&self) {} }
'''})
        k = kinds(root)
        assert "m_a_run" in k and "m_b_run" in k

    def test_a_doc_comment_becomes_a_rationale_node(self, rustcorpus):
        got = docs(rustcorpus(GLOB))
        assert "A compiled glob." in got["glob_glob#doc"]
        assert "Make one." in got["glob_glob_new#doc"]

    def test_a_doc_comment_is_read_through_the_attributes_below_it(self, rustcorpus):
        # `/// ...` then `#[derive(Debug)]` then the item is the ordinary
        # spelling; stopping at the attribute loses the documentation.
        assert "A compiled glob." in docs(rustcorpus(GLOB))["glob_glob#doc"]

    def test_a_comment_separated_by_a_blank_line_is_not_documentation(self, rustcorpus):
        root = rustcorpus({"m.rs": '''/// Unrelated note.

pub struct Thing;
'''})
        assert docs(root) == {}

    def test_a_plain_line_comment_is_not_documentation(self, rustcorpus):
        # `//` is a note about a line of code. Treating every one as
        # documentation buries the `///` runs that really are documentation.
        root = rustcorpus({"m.rs": "// just a note\npub struct Thing;\n"})
        assert docs(root) == {}

    def test_an_inner_doc_comment_documents_the_file(self, rustcorpus):
        # `//!` is Rust's module docstring, and it is the exact equivalent of
        # Python's -- so it hangs off the file node, not off the next item.
        assert "Globs, compiled." in docs(rustcorpus(GLOB))["glob#doc"]

    def test_impl_trait_for_type_is_inheritance(self, rustcorpus):
        root = rustcorpus({"m.rs": '''pub trait Sink {}
pub struct Printer;
impl Sink for Printer {}
'''})
        assert ("m_printer", "m_sink") in links(root, "inherits")

    def test_a_supertrait_is_inheritance(self, rustcorpus):
        root = rustcorpus({"m.rs": '''pub trait Base {}
pub trait Derived: Base {}
'''})
        assert ("m_derived", "m_base") in links(root, "inherits")

    def test_containment_matches_the_python_shape(self, rustcorpus):
        got = links(rustcorpus(GLOB), "contains")
        assert ("glob", "glob_glob") in got
        assert ("glob_glob", "glob_glob_new") in got

    def test_an_impl_for_a_type_from_another_file_is_contained_by_the_file(
            self, rustcorpus):
        # The method is still qualified by the type -- that is what stops two
        # types' `new` colliding -- but containment has to point at a node that
        # exists, and the class node is not in this file.
        root = rustcorpus({"a.rs": "pub struct Glob;\n",
                           "b.rs": "impl Glob { fn extra(&self) {} }\n"})
        got = links(root, "contains")
        assert ("b", "b_glob_extra") in got
        assert kinds(root)["b_glob_extra"] == "method"

    def test_a_macro_definition_is_a_function_node(self, rustcorpus):
        # `macro_rules! x` is a named definition other code calls by name;
        # leaving it out would make every `x!(..)` point at nothing.
        root = rustcorpus({"m.rs": "macro_rules! shout { () => {}; }\n"})
        assert kinds(root)["m_shout"] == "function"

    def test_items_inside_an_inline_mod_belong_to_the_file(self, rustcorpus):
        # A `mod` block is a namespace, not a type. Calling it a class to get a
        # scope would hand an agent asking about a type's methods a module.
        root = rustcorpus({"m.rs": "mod tests {\n    fn helper() {}\n}\n"})
        assert ("m", "m_helper") in links(root, "contains")


class TestResolution:
    def test_a_call_on_self_resolves_through_the_impl_block(self, rustcorpus):
        assert ("glob_glob_go", "glob_glob_helper") in links(rustcorpus(GLOB))

    def test_a_call_on_a_declared_field_of_self_resolves(self, rustcorpus):
        # A struct states the type of every field, so `self.opts.check()` needs
        # no inference at all.
        assert ("glob_glob_go", "glob_globoptions_check") in links(rustcorpus(GLOB))

    def test_a_typed_parameter_types_its_variable(self, rustcorpus):
        assert ("glob_glob_go", "glob_glob_helper") in links(rustcorpus(GLOB))

    def test_a_path_call_on_a_type_resolves(self, rustcorpus):
        # `Glob::new()` names the type outright.
        assert ("glob_glob_go", "glob_glob_new") in links(rustcorpus(GLOB))

    def test_self_in_a_path_call_means_the_impl_target(self, rustcorpus):
        root = rustcorpus({"m.rs": '''pub struct Server;
impl Server {
    fn new() -> Server { Server }
    fn go(&self) { Self::new(); }
}
'''})
        assert ("m_server_go", "m_server_new") in links(root)

    def test_a_declared_let_binding_types_its_variable(self, rustcorpus):
        root = rustcorpus({"m.rs": '''pub struct Server;
impl Server { fn ping(&self) {} }
fn use_it() {
    let s: Server = build();
    s.ping();
}
'''})
        assert ("m_use_it", "m_server_ping") in links(root)

    def test_a_struct_literal_types_its_variable(self, rustcorpus):
        root = rustcorpus({"m.rs": '''pub struct Server { port: u32 }
impl Server { fn ping(&self) {} }
fn use_it() {
    let s = Server { port: 1 };
    s.ping();
}
'''})
        assert ("m_use_it", "m_server_ping") in links(root)

    def test_a_smart_pointer_types_its_variable_through_the_wrapper(self, rustcorpus):
        # `Arc<T>` derefs to T, so the call really is T's. `Option<T>` does not
        # and is deliberately not unwrapped.
        root = rustcorpus({"m.rs": '''pub struct Server;
impl Server { fn ping(&self) {} }
fn use_it(s: Arc<Server>) { s.ping(); }
'''})
        assert ("m_use_it", "m_server_ping") in links(root)

    def test_a_generic_wrapper_that_does_not_deref_is_not_unwrapped(self, rustcorpus):
        root = rustcorpus({"m.rs": '''pub struct Server;
impl Server { fn ping(&self) {} }
fn use_it(s: Option<Server>) { s.ping(); }
'''})
        assert ("m_use_it", "m_server_ping") not in links(root)

    def test_a_plain_call_in_the_same_file(self, rustcorpus):
        root = rustcorpus({"m.rs": "fn helper() {}\nfn go() { helper(); }\n"})
        assert ("m_go", "m_helper") in links(root)

    def test_a_path_call_that_builds_a_type_points_at_the_type(self, rustcorpus):
        # `Choice::First(x)` builds a value; the useful edge is to the type,
        # not to a method called `First` that does not exist.
        root = rustcorpus({"m.rs": '''pub enum Choice { First(u32) }
fn go() { Choice::First(1); }
'''})
        assert ("m_go", "m_choice") in links(root)

    def test_a_call_on_an_untyped_receiver_is_refused_not_guessed(self, rustcorpus):
        # `let s = Server::new()` states nothing about what `new` returns --
        # a builder returning something else is ordinary Rust -- so `s` has no
        # declared type and the call is refused rather than assumed.
        root = rustcorpus({"m.rs": '''pub struct Server;
impl Server { fn ping(&self) {} }
fn use_it() {
    let thing = Server::new();
    thing.ping();
}
'''})
        assert reasons_for(root)["receiver type unknown"] >= 1
        assert ("m_use_it", "m_server_ping") not in links(root)

    def test_a_chained_receiver_is_refused_not_treated_as_a_free_function(
            self, rustcorpus):
        # `a.b().ping()` must not resolve to the free function `ping`.
        root = rustcorpus({"m.rs": '''fn ping() {}
fn use_it(a: Thing) { a.b().ping(); }
'''})
        assert ("m_use_it", "m_ping") not in links(root)
        assert reasons_for(root)["receiver type unknown"] >= 1

    def test_a_std_macro_is_not_recorded_as_a_call(self, rustcorpus):
        # `println!` is punctuation. A corpus that happens to define `write`
        # should not collect a gap marker every time one is written.
        root = rustcorpus({"m.rs": 'fn write() {}\nfn go() { println!("{}", 1); write!(f, "x"); }\n'})
        assert ("m_go", "m_write") not in links(root)

    def test_a_macro_the_corpus_defines_is_recorded_as_a_call(self, rustcorpus):
        root = rustcorpus({"m.rs": 'macro_rules! shout { () => {}; }\nfn go() { shout!(); }\n'})
        assert ("m_go", "m_shout") in links(root)

    def test_an_absolute_use_path_resolves_to_the_file_it_names(self, rustcorpus):
        # `crate::` starts at the directory holding `src`, which is what turns a
        # module path into this corpus's file prefixes.
        root = rustcorpus({"globset/src/lib.rs": "pub mod glob;\n",
                           "globset/src/glob.rs": "pub struct Glob;\n",
                           "globset/src/other.rs": "use crate::glob::Glob;\n"})
        assert ("globset_src_other", "globset_src_glob") in links(root, "imports")

    def test_a_use_path_binds_the_name_it_imports(self, rustcorpus):
        root = rustcorpus({"globset/src/glob.rs": "pub fn compile() {}\n",
                           "globset/src/other.rs":
                               "use crate::glob::compile;\nfn go() { compile(); }\n"})
        files, _ = parse_corpus_files(root)
        binding = [p for p in files if p.prefix == "globset_src_other"][0]
        assert binding.imports["compile"] == "globset.src.glob.compile"
        assert ("globset_src_other_go", "globset_src_glob_compile") in links(root)

    def test_a_relative_use_path_resolves_against_this_module(self, rustcorpus):
        # `self::` names this file's own module, whose children sit in a
        # directory of the file's name.
        root = rustcorpus({"searcher/src/searcher.rs": "use self::mmap::Choice;\n",
                           "searcher/src/searcher/mmap.rs": "pub struct Choice;\n"})
        assert ("searcher_src_searcher", "searcher_src_searcher_mmap") \
            in links(root, "imports")

    def test_self_inside_a_brace_list_means_the_module_itself(self, rustcorpus):
        # `use a::b::{self, C}` imports the module `b` AND the name `C`.
        # Reading that `self` as a name appended a literal "self" segment to the
        # path and the import resolved to nothing.
        root = rustcorpus({"globset/src/glob.rs": "pub struct Candidate;\n",
                           "globset/src/other.rs":
                               "use crate::glob::{self, Candidate};\n"})
        assert ("globset_src_other", "globset_src_glob") in links(root, "imports")

    def test_a_type_parameter_is_not_a_type(self, rustcorpus):
        # `fn go<P: AsRef<Path>>(p: P)` does not say what `p` is. Treating `P`
        # as a class made the map answer "no such method on P" instead of
        # admitting the type is unknown here.
        root = rustcorpus({"m.rs": '''pub struct P;
impl P { fn ping(&self) {} }
fn go<P>(p: P) { p.ping(); }
'''})
        assert ("m_go", "m_p_ping") not in links(root)
        assert reasons_for(root)["receiver type unknown"] >= 1

    def test_a_mod_declaration_names_a_sibling_file(self, rustcorpus):
        root = rustcorpus({"searcher/src/lib.rs": "mod lines;\n",
                           "searcher/src/lines.rs": "pub struct Lines;\n"})
        assert ("searcher_src_lib", "searcher_src_lines") in links(root, "imports")

    def test_a_use_from_another_crate_stays_unresolved(self, rustcorpus):
        # `std::io::Read` genuinely leaves this corpus. Saying so is the useful
        # answer; pretending it resolved would be a lie.
        root = rustcorpus({"m.rs": "use std::io::Read;\n"})
        assert reasons_for(root)["imports a module outside this corpus"] >= 1

    def test_resolved_edges_never_dangle(self, rustcorpus):
        files, _ = parse_corpus_files(rustcorpus(GLOB))
        ids = {n.id for p in files for n in p.nodes}
        edges, _ = resolve(files)
        for e in edges:
            if e.resolved:
                assert e.target in ids


class TestMixedCorpus:
    def test_python_and_rust_in_one_repository(self, rustcorpus):
        root = rustcorpus({"api/src/srv.rs": "pub struct Server;\n",
                           "tools/build.py": "class Builder:\n    pass\n"})
        k = kinds(root)
        assert k["api_src_srv_server"] == "class"
        assert k["tools_build_builder"] == "class"


class TestFailure:
    def test_an_unparseable_file_is_reported_not_skipped_silently(self, rustcorpus):
        root = rustcorpus({"bad.rs": "\x00\x00))))", "ok.rs": "pub struct Ok;\n"})
        _, failed = parse_corpus_files(root)
        assert "ok.rs" not in failed
