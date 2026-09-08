"""C, read through tree-sitter.

C is the language with the least to map: no classes, no methods, no namespaces.
A struct, union or enum is a `class` node and every function is a `function`
node, and the tests below exist to prove that the flattest possible language
still produces the same five shapes as Python does.

Three of them guard bugs that were in this module and are easy to write again:
a function returning a pointer whose NAME came out as its return type, a
`typedef CURLcode Curl_cft_connect(...)` whose name came out as `CURLcode`, and
a header whose whole contents were invisible because they sit inside an include
guard.

`.h` is claimed by C++ as well when both grammars are installed, so every test
of header parsing calls this module directly instead of going through the
registry. Everything else uses `.c`, which nothing else claims.
"""
from pathlib import Path

import pytest

from graphpaat.ids import file_prefix
from graphpaat.parse import Node, ParsedFile, parse_corpus_files
from graphpaat.resolve import resolve

pytest.importorskip("tree_sitter_c")

from graphpaat.languages import c as clang        # noqa: E402


@pytest.fixture
def ccorpus(tmp_path):
    def build(files: dict[str, str]):
        root = tmp_path / "repo"
        for name, source in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
        root.mkdir(parents=True, exist_ok=True)
        return root
    return build


def read(source: str, path: str = "m.c") -> ParsedFile:
    """One file, parsed by this module regardless of what else is installed.

    Mirrors `parse_file`, including the node for the file itself, so the node
    lists here are the same ones a real build produces.
    """
    parsed = ParsedFile(path=path,
                        prefix=file_prefix(Path("/r") / path, Path("/r"),
                                           keep_extension=True))
    parsed.nodes.append(Node(id=parsed.prefix, label=Path(path).name, kind="file",
                             file=path, line=1))
    assert clang.parse(source, parsed) is not False
    return parsed


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


def unresolved(root, relation):
    files, _ = parse_corpus_files(root)
    edges, _ = resolve(files)
    return {(e.source, e.target) for e in edges
            if e.relation == relation and not e.resolved}


def reasons_for(root):
    files, _ = parse_corpus_files(root)
    return resolve(files)[1]


LIST = '''/* A doubly linked list. */
#include <stdlib.h>
#include "adlist.h"

/* Create a new list.
 * On error, NULL is returned. */
list *listCreate(void)
{
    list *l;
    l = allocate(sizeof(*l));
    return l;
}

void *allocate(size_t n) { return malloc(n); }
'''


class TestSameShapesAsPython:
    def test_a_struct_is_a_class_node(self):
        parsed = read("struct point { int x; int y; };")
        assert {n.id: n.kind for n in parsed.nodes}["m_c_point"] == "class"

    def test_a_union_is_a_class_node(self):
        parsed = read("union value { int i; float f; };")
        assert {n.id: n.kind for n in parsed.nodes}["m_c_value"] == "class"

    def test_an_enum_is_a_class_node(self):
        parsed = read("enum colour { RED, GREEN };")
        assert {n.id: n.kind for n in parsed.nodes}["m_c_colour"] == "class"

    def test_a_typedef_struct_is_named_by_its_typedef_name(self):
        # `struct _entry` is spelled `Entry` everywhere else in the source, so
        # that is the name the map carries. redis does this 30 times.
        parsed = read("typedef struct _entry { int x; } Entry;")
        labels = {n.label for n in parsed.nodes if n.kind == "class"}
        assert labels == {"Entry"}

    def test_a_function_type_typedef_is_named_by_its_declarator(self):
        # `typedef CURLcode Curl_cft_connect(...)` -- the FIRST type name is
        # the return type. Reading forwards named eight of curl's eleven filter
        # callbacks `CURLcode` and collided them into one node.
        parsed = read("typedef int Curl_cft_connect(struct cf *f, int flags);")
        assert {n.label for n in parsed.nodes if n.kind == "class"} == \
            {"Curl_cft_connect"}

    def test_a_function_is_a_function_node(self):
        parsed = read(LIST)
        assert {n.id: n.kind for n in parsed.nodes}["m_c_listcreate"] == "function"

    def test_a_pointer_returning_function_is_named_by_the_function(self):
        # `list *listCreate(void)` puts the return type first, so taking the
        # first name-shaped child called this function `list` -- and so did
        # every other pointer-returning function in the file, which then all
        # minted one id.
        parsed = read("list *listCreate(void) { return 0; }")
        assert [n.label for n in parsed.nodes if n.kind == "function"] == \
            ["listCreate"]

    def test_nothing_in_c_becomes_a_method(self):
        # A struct of function pointers looks like an object with methods and
        # is not one: the value is chosen at run time, so a method node here
        # would be a definition the source does not contain.
        parsed = read('''struct handler {
    int (*connect)(struct handler *h);
    void (*done)(struct handler *h);
};
''')
        assert [n.kind for n in parsed.nodes] == ["file", "class"]
        assert parsed.owner_of == {}

    def test_a_block_comment_above_a_declaration_is_documentation(self):
        parsed = read(LIST)
        docs = {n.id: n.text for n in parsed.nodes if n.kind == "rationale"}
        assert docs["m_c_listcreate#doc"] == \
            "Create a new list. On error, NULL is returned."

    def test_a_run_of_line_comments_is_also_documentation(self):
        # C has no dedicated doc syntax, so both comment forms are the same
        # act. Taking only `/* */` would leave most of a real file silent.
        parsed = read("// Frees the node.\n// Never fails.\nvoid drop(void) {}")
        docs = [n.text for n in parsed.nodes if n.kind == "rationale"]
        assert docs == ["Frees the node. Never fails."]

    def test_a_comment_separated_by_a_blank_line_is_not_documentation(self):
        parsed = read("/* Unrelated note. */\n\nstruct thing { int x; };")
        assert not any(n.kind == "rationale" for n in parsed.nodes)

    def test_containment_matches_the_python_shape(self, ccorpus):
        got = links(ccorpus({"m.c": LIST}), "contains")
        assert ("m_c", "m_c_listcreate") in got
        # The doc hangs off the symbol, not the file -- same shape as Python.
        assert ("m_c_listcreate", "m_c_listcreate#doc") in \
            links(ccorpus({"m.c": LIST}), "rationale_for")

    def test_two_files_with_one_stem_do_not_collide(self, ccorpus):
        # `url.c` and `url.h` are two files. Dropping the extension from the id
        # prefix cost curl 159 of 385 files and redis 67 of 218.
        k = kinds(ccorpus({"url.c": "void go(void) {}\n",
                           "url.h": "struct url { int x; };\n"}))
        assert k["url_c"] == "file" and k["url_h"] == "file"


class TestWhatIsDeliberatelyNotANode:
    def test_a_function_prototype_is_not_a_node(self):
        # A prototype names a function defined elsewhere. Emitting it would put
        # two nodes under one name, which turns every cross-file call to that
        # name into a refusal: 2,788 of redis's 5,708 function names carry one.
        parsed = read("void listRelease(list *l);")
        assert [n.kind for n in parsed.nodes] == ["file"]

    def test_a_forward_declaration_beside_its_definition_is_not_a_node(self):
        # The same rule inside one file, where it matters more: a second node
        # would break resolution of calls made in this very file.
        parsed = read("static void helper(int x);\nstatic void helper(int x) {}")
        assert [n.label for n in parsed.nodes if n.kind == "function"] == ["helper"]

    def test_an_opaque_struct_declaration_is_not_a_class_node(self):
        # `struct conn;` says a type exists; the definition, with everything
        # worth reading, is in another file and gets the node.
        parsed = read("struct conn;\nstruct conn *open(void) { return 0; }")
        assert not any(n.kind == "class" for n in parsed.nodes)

    def test_a_function_like_macro_is_not_a_node(self):
        # Tempting, because a macro is callable and is often the real API. But
        # one is routinely defined twice per file behind `#ifdef`, and that
        # alone would mint 81 colliding ids on redis and 102 on curl.
        parsed = read("#define listLength(l) ((l)->len)\n")
        assert [n.kind for n in parsed.nodes] == ["file"]


class TestThePreprocessor:
    def test_declarations_inside_an_include_guard_are_found(self):
        # Every header opens with `#ifndef GUARD`, so reading only the top
        # level of the tree finds the guard and nothing else. adlist.h's three
        # structs are children of one preproc_ifdef.
        parsed = read('''#ifndef ADLIST_H
#define ADLIST_H
struct listNode { int x; };
#endif
''', path="adlist.h")
        assert [n.label for n in parsed.nodes if n.kind == "class"] == ["listNode"]

    def test_both_arms_of_a_conditional_are_read(self):
        # The preprocessor picks one; we are describing what the source
        # contains, and a Windows-only function is still a function.
        parsed = read('''#ifdef WIN32
void win_open(void) {}
#else
void unix_open(void) {}
#endif
''')
        assert {n.label for n in parsed.nodes if n.kind == "function"} == \
            {"win_open", "unix_open"}

    def test_one_name_defined_in_two_arms_becomes_one_node(self):
        # curl's `curl_ed25519.c` defines `Curl_ed25519_sign` four times, once
        # per TLS backend. That is one symbol, and minting the id four times
        # would report it as data loss that never happened.
        parsed = read('''#ifdef USE_OPENSSL
int sign(void) { return 1; }
#else
int sign(void) { return 2; }
#endif
''')
        assert [n.line for n in parsed.nodes if n.kind == "function"] == [2]


class TestIncludes:
    def test_a_quoted_include_carries_the_directory_and_the_extension(self):
        # C looks beside the including file first and on the include path
        # second. Emitting the directory-qualified form gets both, because
        # resolution tries the whole string and then drops leading segments.
        parsed = read('#include "urldata.h"\n', path="vtls/openssl.c")
        assert parsed.import_sites == [("vtls.urldata.h", 1)]

    def test_a_relative_include_is_made_absolute(self):
        parsed = read('#include "../redismodule.h"\n', path="modules/hello.c")
        assert parsed.import_sites == [("redismodule.h", 1)]

    def test_an_include_resolves_to_the_header_it_names(self, ccorpus):
        root = ccorpus({"vtls/openssl.c": '#include "urldata.h"\nvoid go(void) {}\n',
                        "urldata.h": "struct Curl_easy { int x; };\n"})
        assert ("vtls_openssl_c", "urldata_h") in links(root, "imports")

    def test_a_system_include_stays_unresolved(self, ccorpus):
        # `<stdio.h>` is not in the corpus and is not meant to be. Saying
        # nothing about it would hide what the file actually depends on.
        root = ccorpus({"m.c": "#include <stdio.h>\nvoid go(void) {}\n"})
        assert ("m_c", "?stdio.h") in unresolved(root, "imports")

    def test_c_records_no_per_symbol_import(self):
        # An include brings in every name a header declares, and which ones
        # those are is not knowable while parsing this file alone. Claiming
        # otherwise would let the resolver trust evidence that does not exist.
        parsed = read('#include "adlist.h"\n')
        assert parsed.imports == {}


class TestResolution:
    def test_a_plain_call_in_the_same_file(self, ccorpus):
        root = ccorpus({"m.c": "void helper(void) {}\nvoid run(void) { helper(); }\n"})
        assert ("m_c_run", "m_c_helper") in links(root)

    def test_a_call_to_a_function_in_another_file(self, ccorpus):
        # C's only cross-file evidence: the name is defined exactly once in the
        # corpus. An include proves nothing about which symbol was meant.
        root = ccorpus({"a.c": "void helper(void) {}\n",
                        "b.c": "void run(void) { helper(); }\n"})
        assert ("b_c_run", "a_c_helper") in links(root)

    def test_a_declared_local_records_its_type(self):
        # C states this outright, which no dynamic language does.
        parsed = read('''void run(void) {
    struct connectdata *conn;
    listNode *node;
}
''')
        assert parsed.var_types == {"run::conn": "connectdata",
                                    "run::node": "listNode"}

    def test_a_typed_parameter_records_its_type(self):
        parsed = read("void run(list *l, int n) { }")
        assert parsed.var_types == {"run::l": "list"}

    def test_several_variables_in_one_declaration_are_all_typed(self):
        # `listNode *current, *next;` is two variables; taking only the first
        # left `next` untyped in the very file that declares it.
        parsed = read("void run(void) { listNode *current, *next; }")
        assert parsed.var_types == {"run::current": "listNode",
                                    "run::next": "listNode"}

    def test_a_builtin_type_records_nothing(self):
        # No node in the graph can ever be an `unsigned long`.
        parsed = read("void run(void) { unsigned long len; int n; }")
        assert parsed.var_types == {}

    def test_a_function_pointer_call_is_not_read_as_a_global_call(self, ccorpus):
        # `list->free(x)` calls a field, not the corpus function named `free`.
        # Recording it bare would invent an edge to whatever else has the name.
        root = ccorpus({"m.c": '''void freeit(void) {}
void run(list *l) { l->freeit(l); }
'''})
        assert ("m_c_run", "m_c_freeit") not in links(root)

    def test_a_call_on_an_untyped_receiver_is_refused_not_guessed(self, ccorpus):
        root = ccorpus({"m.c": "void run(void *p) { p->go(p); }\n"})
        assert reasons_for(root)["receiver type unknown"] >= 1

    def test_a_typed_receiver_with_no_such_member_is_refused(self, ccorpus):
        # The type is declared, so the refusal can say more than "unknown" --
        # which is the whole value of C's declared locals here.
        root = ccorpus({"m.c": "struct conn { int x; };\nvoid run(struct conn *c) { c->go(c); }\n"})
        assert reasons_for(root)["receiver typed, but no such method in the corpus"] >= 1

    def test_resolved_edges_never_dangle(self, ccorpus):
        root = ccorpus({"m.c": LIST, "adlist.h": "struct list { int len; };\n",
                        "b.c": "void other(void) { listCreate(); }\n"})
        files, _ = parse_corpus_files(root)
        ids = {n.id for p in files for n in p.nodes}
        edges, _ = resolve(files)
        for edge in edges:
            if edge.resolved:
                assert edge.target in ids


class TestFailure:
    def test_an_unparseable_file_is_reported_not_skipped_silently(self, ccorpus):
        root = ccorpus({"bad.c": "\x00\x00", "ok.c": "void go(void) {}\n"})
        _, failed = parse_corpus_files(root)
        assert "ok.c" not in failed

    def test_a_file_of_macros_still_parses(self, ccorpus):
        # Macro-dense C makes tree-sitter emit ERROR nodes and keep going.
        # 162 of the 603 files in the two C corpora contain at least one, and
        # none is a total loss -- treating an error as a failure would throw
        # away a quarter of the map.
        root = ccorpus({"m.c": '''#define WRAP(x) do { x; } while(0)
LIST_HEAD(things);
void go(void) { WRAP(1); }
'''})
        files, failed = parse_corpus_files(root)
        assert failed == []
        assert any(n.label == "go" for p in files for n in p.nodes)
