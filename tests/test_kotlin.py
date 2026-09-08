"""Kotlin, read through tree-sitter.

The point of another language is not that it works, but that it produces the
SAME shapes: an `object` is a `class` node, a supertype is `inherits`, a KDoc
block is a `rationale`. If any of that needed a new word, the model was never
language-neutral and everything downstream would have to learn Kotlin too.

Kotlin's own awkwardnesses are tested here as well, one test per rule: a
companion object folded into the type around it, an extension function scoped
by the type it extends, and an unqualified call that is only a call on self
when the type actually declares that name.

Every source in this file is written with real line breaks. The grammar uses
them as statement separators, and a class body squeezed onto one line does not
parse -- which would make a test pass or fail for a reason that has nothing to
do with the rule under test.
"""
import pytest

from graphpaat.parse import parse_corpus_files
from graphpaat.resolve import resolve

pytest.importorskip("tree_sitter_kotlin")


@pytest.fixture
def ktcorpus(tmp_path):
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


SERVER = {"demo/Srv.kt": '''package demo

/** Serves connections. */
class Server(val addr: String) {
  private val pool: Pool = Pool()

  /** Starts it. */
  fun start(): Int {
    pool.close()
    return listen()
  }

  private fun listen(): Int = 0
}

class Pool {
  fun close() {}
}

fun make(addr: String): Server = Server(addr)
'''}


class TestSameShapesAsPython:
    def test_a_class_is_a_class_node(self, ktcorpus):
        assert kinds(ktcorpus(SERVER))["demo_srv_server"] == "class"

    def test_an_interface_is_also_a_class_node(self, ktcorpus):
        root = ktcorpus({"I.kt": "package demo\n\ninterface Handler {\n  fun handle()\n}\n"})
        assert kinds(root)["i_handler"] == "class"

    def test_an_object_is_also_a_class_node(self, ktcorpus):
        # A singleton is still a type: it has members and it is named.
        root = ktcorpus({"R.kt": "package demo\n\nobject Registry {\n  fun add() {}\n}\n"})
        k = kinds(root)
        assert k["r_registry"] == "class"
        assert k["r_registry_add"] == "method"

    def test_an_enum_class_is_also_a_class_node(self, ktcorpus):
        root = ktcorpus({"K.kt": '''package demo

enum class Protocol {
  HTTP_1_1,
  HTTP_2,
}
'''})
        assert kinds(root)["k_protocol"] == "class"

    def test_a_type_alias_is_also_a_class_node(self, ktcorpus):
        root = ktcorpus({"A.kt": "package demo\n\ntypealias Handler = (Int) -> Unit\n"})
        assert kinds(root)["a_handler"] == "class"

    def test_a_top_level_fun_is_a_function_and_a_member_is_a_method(self, ktcorpus):
        k = kinds(ktcorpus(SERVER))
        assert k["demo_srv_make"] == "function"
        assert k["demo_srv_server_start"] == "method"

    def test_methods_are_qualified_by_their_type(self, ktcorpus):
        root = ktcorpus({"M.kt": '''package demo

class A {
  fun run() {}
}

class B {
  fun run() {}
}
'''})
        k = kinds(root)
        assert "m_a_run" in k and "m_b_run" in k

    def test_a_kdoc_block_becomes_a_rationale_node(self, ktcorpus):
        got = docs(ktcorpus(SERVER))
        assert got["demo_srv_server#doc"] == "Serves connections."
        assert got["demo_srv_server_start#doc"] == "Starts it."

    def test_a_kdoc_separated_by_a_blank_line_is_not_documentation(self, ktcorpus):
        root = ktcorpus({"M.kt": '''package demo

/** Unrelated note. */

class Thing
'''})
        assert not docs(root)

    def test_a_line_comment_is_not_documentation(self, ktcorpus):
        # Kotlin has a documentation form of its own; a `//` note above a
        # declaration is usually about the line below it, not about the type.
        root = ktcorpus({"M.kt": "package demo\n\n// just a note\nclass Thing\n"})
        assert not docs(root)

    def test_a_licence_header_is_not_documentation(self, ktcorpus):
        # Every file in a real Kotlin repository opens with one, and it opens
        # `/*` rather than `/**`, so it is skipped without a special case.
        root = ktcorpus({"M.kt": "/*\n * Copyright.\n */\npackage demo\n\nclass Thing\n"})
        assert not docs(root)

    def test_a_kdoc_survives_a_line_comment_between_it_and_the_declaration(self, ktcorpus):
        root = ktcorpus({"M.kt": '''package demo

/** The real description. */
// TODO: revisit
class Thing
'''})
        assert docs(root)["m_thing#doc"] == "The real description."

    def test_a_kdoc_above_the_package_line_documents_the_file(self, ktcorpus):
        root = ktcorpus({"M.kt": "/** What this file is for. */\npackage demo\n\nclass Thing\n"})
        assert docs(root)["m#doc"] == "What this file is for."

    def test_containment_matches_the_python_shape(self, ktcorpus):
        got = links(ktcorpus(SERVER), "contains")
        assert ("demo_srv", "demo_srv_server") in got
        assert ("demo_srv_server", "demo_srv_server_start") in got

    def test_a_supertype_is_inheritance(self, ktcorpus):
        root = ktcorpus({"M.kt": '''package demo

open class Base

class Derived : Base()
'''})
        assert ("m_derived", "m_base") in links(root, "inherits")

    def test_an_implemented_interface_is_inheritance_too(self, ktcorpus):
        root = ktcorpus({"M.kt": '''package demo

interface Handler

class Server : Handler
'''})
        assert ("m_server", "m_handler") in links(root, "inherits")

    def test_delegation_is_inheritance_too(self, ktcorpus):
        # `by` means the class gains the interface's whole shape, which is what
        # a reader asking "what is a Handler here" wants to see.
        root = ktcorpus({"M.kt": '''package demo

interface Handler

class Server(impl: Handler) : Handler by impl
'''})
        assert ("m_server", "m_handler") in links(root, "inherits")

    def test_a_nested_type_is_minted_flat_and_contained_by_its_parent(self, ktcorpus):
        # Flat, because every receiver-typed call on a `Listener` and every
        # import of one looks for `mint(file prefix, "Listener")`.
        root = ktcorpus({"M.kt": '''package demo

class Connection {
  interface Listener {
    fun onSettings()
  }
}
'''})
        k = kinds(root)
        assert k["m_listener"] == "class"
        assert k["m_listener_onsettings"] == "method"
        assert ("m_connection", "m_listener") in links(root, "contains")

    def test_a_property_is_not_a_node(self, ktcorpus):
        # In Kotlin every field is a property, so a member list is mostly
        # private state -- the thing no language module here emits.
        root = ktcorpus({"M.kt": '''package demo

class Server {
  val addr: String = ""
}
'''})
        assert "m_server_addr" not in kinds(ktcorpus({"M.kt": ""}))
        assert "m_server_addr" not in kinds(root)

    def test_an_anonymous_object_gets_no_node(self, ktcorpus):
        # It declares a type nobody can refer to, so there is no name to mint.
        # The calls inside it are still attributed to the enclosing method.
        root = ktcorpus({"M.kt": '''package demo

class Server {
  fun handler(): Any =
    object : Runnable {
      override fun run() {}
    }
}
'''})
        assert set(kinds(root)) == {"m", "m_server", "m_server_handler"}


class TestKotlinShapes:
    def test_a_companion_member_belongs_to_the_type_around_it(self, ktcorpus):
        # A companion object is where Kotlin keeps statics, and the call site
        # writes `Server.create()`. Folding it into Server is what makes that
        # resolve; a separate `Companion` node would resolve nothing.
        root = ktcorpus({"M.kt": '''package demo

class Server {
  companion object {
    fun create(): Server = Server()
  }
}
'''})
        k = kinds(root)
        assert k["m_server_create"] == "method"
        assert "m_companion" not in k
        assert ("m_server", "m_server_create") in links(root, "contains")

    def test_a_top_level_extension_is_a_method_of_the_type_it_extends(self, ktcorpus):
        root = ktcorpus({"M.kt": "package demo\n\nfun String.toSlug(): String = lowercase()\n"})
        k = kinds(root)
        assert k["m_string_toslug"] == "method"
        # It is contained by the FILE, because it is not inside the class.
        assert ("m", "m_string_toslug") in links(root, "contains")

    def test_two_extensions_of_one_name_on_different_types_do_not_collide(self, ktcorpus):
        # Kotlin overloads an extension by changing the receiver, not the name.
        # A flat id would mint one node for both and lose one outright.
        root = ktcorpus({"M.kt": '''package demo

fun String.toBody(): Int = 1

fun ByteArray.toBody(): Int = 2
'''})
        k = kinds(root)
        assert "m_string_tobody" in k and "m_bytearray_tobody" in k

    def test_a_member_extension_carries_both_names(self, ktcorpus):
        # `fun URL.parse()` inside `HttpUrl` needs an HttpUrl AND a URL, so
        # both are in the id -- otherwise the same function written inside two
        # classes mints one id.
        root = ktcorpus({"M.kt": '''package demo

class HttpUrl {
  companion object {
    fun String.parse(): Int = 1

    fun ByteArray.parse(): Int = 2
  }
}
'''})
        k = kinds(root)
        assert k["m_httpurl_string_parse"] == "method"
        assert k["m_httpurl_bytearray_parse"] == "method"

    def test_a_secondary_constructor_is_a_method_not_a_second_class_name(self, ktcorpus):
        # Kotlin spells construction `Server(addr)`, which is a call to the
        # class. Naming the constructor `Server` too would give the resolver
        # two candidates and it would refuse every construction.
        root = ktcorpus({"M.kt": '''package demo

class Server {
  constructor(addr: String) {
    setUp()
  }

  fun setUp() {}
}
'''})
        k = kinds(root)
        assert k["m_server_constructor"] == "method"
        assert ("m_server_constructor", "m_server_setup") in links(root)

    def test_an_enum_entry_body_contributes_methods_to_the_enum(self, ktcorpus):
        # An entry is a value, not a class -- but an enum whose whole behaviour
        # lives in its entries would otherwise have no methods at all.
        root = ktcorpus({"M.kt": '''package demo

enum class Policy {
  LAZY {
    override fun read(): Int = 1
  },
  EAGER {
    override fun read(): Int = 2
  },
  ;

  abstract fun read(): Int
}
'''})
        assert kinds(root)["m_policy_read"] == "method"

    def test_overloads_are_one_node(self, ktcorpus):
        # A caller writes one name for both; they are one entry point.
        root = ktcorpus({"M.kt": '''package demo

class Reader {
  fun read(n: Int): Int = n

  fun read(n: Int, m: Int): Int = n + m
}
'''})
        files, _ = parse_corpus_files(root)
        assert [n.id for p in files for n in p.nodes].count("m_reader_read") == 1

    def test_a_factory_function_named_after_its_type_is_kept_not_dropped(self, ktcorpus):
        # Kotlin writes a constructor-shaped function beside the type it makes:
        # `interface Job` and `fun Job(parent)` in one file. Both mint one id,
        # which the shared collision report is there to surface -- letting
        # whichever came first silently swallow the other hid the loss.
        root = ktcorpus({"M.kt": '''package demo

interface Job

fun Job(parent: Int): Job = TODO()
'''})
        files, _ = parse_corpus_files(root)
        claimed = [(n.kind, n.line) for p in files for n in p.nodes if n.id == "m_job"]
        assert claimed == [("class", 3), ("function", 5)]

    def test_a_call_written_in_a_class_body_is_attributed_to_the_class(self, ktcorpus):
        # An `init { }` block and a property initializer run real code, and the
        # type is the only honest caller for it.
        root = ktcorpus({"M.kt": '''package demo

fun setUp() {}

class Server {
  init {
    setUp()
  }
}
'''})
        assert ("m_server", "m_setup") in links(root)


class TestResolution:
    def test_a_call_on_a_declared_parameter_resolves(self, ktcorpus):
        root = ktcorpus({"M.kt": '''package demo

class Server {
  fun ping() {}
}

fun use(srv: Server) {
  srv.ping()
}
'''})
        assert ("m_use", "m_server_ping") in links(root)

    def test_a_declared_property_type_resolves_a_call_on_it(self, ktcorpus):
        assert ("demo_srv_server_start", "demo_srv_pool_close") in links(ktcorpus(SERVER))

    def test_an_initializer_types_its_variable(self, ktcorpus):
        # `val srv = Server()` states the type as plainly as an annotation.
        root = ktcorpus({"M.kt": '''package demo

class Server {
  fun ping() {}
}

fun use() {
  val srv = Server()
  srv.ping()
}
'''})
        assert ("m_use", "m_server_ping") in links(root)

    def test_a_bare_call_to_a_method_the_class_declares_is_a_call_on_self(self, ktcorpus):
        assert ("demo_srv_server_start", "demo_srv_server_listen") in links(ktcorpus(SERVER))

    def test_a_bare_call_to_a_top_level_function_is_not_a_call_on_self(self, ktcorpus):
        # The rule that matters most in Kotlin: treating every unqualified call
        # as a member sends this one to the self-method path, which refuses it
        # and never reaches the same-file rule.
        root = ktcorpus({"M.kt": '''package demo

fun helper() {}

class Server {
  fun start() {
    helper()
  }
}
'''})
        assert ("m_server_start", "m_helper") in links(root)

    def test_a_call_through_this_resolves(self, ktcorpus):
        root = ktcorpus({"M.kt": '''package demo

class Server {
  fun start() {
    this.listen()
  }

  fun listen() {}
}
'''})
        assert ("m_server_start", "m_server_listen") in links(root)

    def test_a_call_on_a_property_of_this_resolves(self, ktcorpus):
        root = ktcorpus({"M.kt": '''package demo

class Pool {
  fun close() {}
}

class Server {
  private val pool: Pool = Pool()

  fun stop() {
    this.pool.close()
  }
}
'''})
        assert ("m_server_stop", "m_pool_close") in links(root)

    def test_a_companion_member_is_reached_through_the_type_name(self, ktcorpus):
        root = ktcorpus({"M.kt": '''package demo

class Server {
  companion object {
    fun create(): Int = 1
  }
}

fun build(): Int = Server.create()
'''})
        assert ("m_build", "m_server_create") in links(root)

    def test_a_constructor_call_points_at_the_class(self, ktcorpus):
        root = ktcorpus({"M.kt": '''package demo

class Server

fun build(): Server = Server()
'''})
        assert ("m_build", "m_server") in links(root)

    def test_a_super_call_names_the_constructed_supertype(self, ktcorpus):
        # Kotlin only lets the superclass be constructed in the supertype list,
        # so the parent is named by the grammar and not by a convention.
        root = ktcorpus({"M.kt": '''package demo

open class Base {
  open fun close() {}
}

class Child : Base() {
  override fun close() {
    super.close()
  }
}
'''})
        assert ("m_child_close", "m_base_close") in links(root)

    def test_a_call_on_an_untyped_receiver_is_refused_not_guessed(self, ktcorpus):
        # `val thing = compute()` names no type: `compute` is not capitalised,
        # so it is a function call and not a construction. Nothing in the file
        # says what it returns, and the map says so rather than guessing.
        root = ktcorpus({"M.kt": '''package demo

class Server {
  fun ping() {}
}

fun compute(): Any = Any()

fun use() {
  val thing = compute()
  thing.ping()
}
'''})
        assert reasons_for(root)["receiver type unknown"] >= 1

    def test_a_bare_call_on_self_with_no_such_method_is_refused(self, ktcorpus):
        root = ktcorpus({"M.kt": '''package demo

class Server {
  fun start() {
    this.missing()
  }
}
'''})
        assert reasons_for(root)["called on self, but no such method on the class"] >= 1

    def test_imports_are_recorded_and_bind_a_local_name(self, ktcorpus):
        root = ktcorpus({"demo/M.kt": "package demo\n\nimport okio.ByteString\n"})
        files, _ = parse_corpus_files(root)
        assert files[0].imports["ByteString"] == "okio.ByteString.ByteString"
        assert files[0].import_sites == [("okio.ByteString", 3)]

    def test_an_import_is_cut_at_the_type_so_a_nested_one_names_its_file(self, ktcorpus):
        # `Http2Connection.Listener` lives in Http2Connection.kt; without the
        # cut this points at a directory and resolves to nothing.
        root = ktcorpus({"demo/M.kt":
                         "package demo\n\nimport demo.http2.Http2Connection.Listener\n"})
        files, _ = parse_corpus_files(root)
        assert files[0].import_sites == [("demo.http2.Http2Connection", 3)]

    def test_a_wildcard_import_binds_no_name(self, ktcorpus):
        root = ktcorpus({"demo/M.kt": "package demo\n\nimport okio.*\n"})
        files, _ = parse_corpus_files(root)
        assert files[0].imports == {}

    def test_an_import_is_rewritten_through_the_source_root(self, ktcorpus):
        # The file states its package and its path; the tail they share is the
        # mirrored part, and what is left over is the source root. Without this
        # not one import in either real corpus names a file.
        root = ktcorpus({
            "jvmMain/kotlin/demo/Srv.kt": '''package demo

import demo.util.Helper

fun use(): Int = Helper.go()
''',
            "jvmMain/kotlin/demo/util/Helper.kt": '''package demo.util

object Helper {
  fun go(): Int = 1
}
''',
        })
        got = links(root, "imports")
        assert ("jvmmain_kotlin_demo_srv",
                "jvmmain_kotlin_demo_util_helper") in got

    def test_an_import_that_leaves_the_corpus_keeps_the_name_the_source_wrote(self, ktcorpus):
        # Prefixing it with the source root would report a module nobody wrote.
        root = ktcorpus({"jvmMain/kotlin/demo/Srv.kt":
                         "package demo\n\nimport java.io.IOException\n"})
        files, _ = parse_corpus_files(root)
        assert files[0].import_sites == [("java.io.IOException", 3)]

    def test_an_imported_symbol_resolves_a_bare_call(self, ktcorpus):
        root = ktcorpus({
            "demo/Srv.kt": '''package demo

import demo.util.Helper

fun use(): Int = Helper()
''',
            "demo/util/Helper.kt": '''package demo.util

class Helper
''',
        })
        assert ("demo_srv_use", "demo_util_helper_helper") in links(root)

    def test_resolved_edges_never_dangle(self, ktcorpus):
        files, _ = parse_corpus_files(ktcorpus(SERVER))
        ids = {n.id for p in files for n in p.nodes}
        edges, _ = resolve(files)
        for e in edges:
            if e.resolved:
                assert e.target in ids


class TestMixedCorpus:
    def test_python_and_kotlin_in_one_repository(self, ktcorpus):
        root = ktcorpus({"api/Srv.kt": "package api\n\nclass Server\n",
                         "tools/build.py": "class Builder:\n    pass\n"})
        k = kinds(root)
        assert k["api_srv_server"] == "class"
        assert k["tools_build_builder"] == "class"


class TestFailure:
    def test_a_kotlin_script_is_read_too(self, ktcorpus):
        # A build script is Kotlin and its declarations are real symbols. The
        # id keeps the inner dot of `build.gradle.kts` because the shared
        # prefix rule drops only the last extension; that is not this module's
        # to decide and it collides with nothing.
        root = ktcorpus({"build.gradle.kts": "fun configure() {}\n"})
        assert kinds(root)["build.gradle_configure"] == "function"

    def test_an_unparseable_file_is_reported_not_skipped_silently(self, ktcorpus):
        root = ktcorpus({"Bad.kt": "\x00\x00 ((((", "Ok.kt": "package demo\n"})
        _, failed = parse_corpus_files(root)
        assert "Ok.kt" not in failed

    def test_a_file_with_one_broken_declaration_still_yields_the_rest(self, ktcorpus):
        # A syntax error in one function is not a reason to lose the file; the
        # grammar recovers and the other declarations are still there.
        root = ktcorpus({"M.kt": '''package demo

class Server {
  fun works() {
    val x = ]
  }

  fun other() {}
}
'''})
        _, failed = parse_corpus_files(root)
        assert failed == []
        assert {"m_server", "m_server_works", "m_server_other"} <= set(kinds(root))
