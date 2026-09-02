"""Resolution: what we prove, and what we refuse.

The refusal tests matter as much as the resolution tests. A resolver that emits
a plausible-looking wrong edge is worse than one that emits nothing, because a
graph only records successes and a reader cannot tell a missing edge from a
nonexistent relationship. Every rule here has a test for what it accepts AND
what it declines.
"""
from graphpaat.parse import parse_corpus_files
from graphpaat.resolve import resolve


def edges_of(root, relation="calls", resolved=True):
    files, _ = parse_corpus_files(root)
    edges, reasons = resolve(files)
    return ([(e.source, e.target, e.reason) for e in edges
             if e.relation == relation and e.resolved is resolved], reasons)


def links(root, relation="calls"):
    got, _ = edges_of(root, relation)
    return {(s, t) for s, t, _ in got}


class TestBareCalls:
    def test_same_file(self, corpus):
        root = corpus({"m.py": "def helper():\n    pass\ndef go():\n    helper()\n"})
        assert ("m_go", "m_helper") in links(root)

    def test_imported(self, corpus):
        root = corpus({
            "lib.py": "def helper():\n    pass\n",
            "app.py": "from lib import helper\ndef go():\n    helper()\n"})
        assert ("app_go", "lib_helper") in links(root)

    def test_unique_name_anywhere_in_corpus(self, corpus):
        root = corpus({
            "lib.py": "def only_one():\n    pass\n",
            "app.py": "def go():\n    only_one()\n"})
        assert ("app_go", "lib_only_one") in links(root)

    def test_ambiguous_name_is_refused_not_guessed(self, corpus):
        root = corpus({
            "a.py": "def dup():\n    pass\n",
            "b.py": "def dup():\n    pass\n",
            "app.py": "def go():\n    dup()\n"})
        got, reasons = edges_of(root)
        assert not any(s == "app_go" for s, _, _ in got)
        assert any("cannot choose" in r for r in reasons)

    def test_builtin_is_never_resolved_to_a_same_named_symbol(self, corpus):
        # Shipped bug: a plain set() was resolved to a function named set in a
        # vendored fixture. 307 wrong edges on one corpus.
        root = corpus({
            "vendored.py": "def set():\n    pass\n",
            "app.py": "def go():\n    set()\n"})
        got, reasons = edges_of(root)
        assert not any(s == "app_go" for s, _, _ in got)
        assert reasons["python builtin"] == 1

    def test_a_local_definition_still_shadows_a_builtin(self, corpus):
        # The builtin filter applies only to the weakest rule. A definition in
        # the same file is real evidence of shadowing.
        root = corpus({"m.py": "def set():\n    pass\ndef go():\n    set()\n"})
        assert ("m_go", "m_set") in links(root)

    def test_unknown_name_is_refused(self, corpus):
        root = corpus({"m.py": "def go():\n    third_party_thing()\n"})
        _, reasons = edges_of(root)
        assert reasons["not defined in this corpus (builtin or third party)"] == 1


class TestMethodCalls:
    def test_self_method(self, corpus):
        root = corpus({"m.py": (
            "class S:\n"
            "    def go(self):\n"
            "        self.work()\n"
            "    def work(self):\n"
            "        pass\n")})
        assert ("m_s_go", "m_s_work") in links(root)

    def test_self_method_that_does_not_exist_is_refused(self, corpus):
        root = corpus({"m.py": "class S:\n    def go(self):\n        self.missing()\n"})
        _, reasons = edges_of(root)
        assert reasons["called on self, but no such method on the class"] == 1

    def test_receiver_typed_by_assignment(self, corpus):
        root = corpus({
            "db.py": "class Database:\n    def connect(self):\n        pass\n",
            "app.py": ("from db import Database\n"
                       "def go():\n"
                       "    conn = Database()\n"
                       "    conn.connect()\n")})
        assert ("app_go", "db_database_connect") in links(root)

    def test_receiver_typed_by_annotation(self, corpus):
        root = corpus({
            "db.py": "class Database:\n    def connect(self):\n        pass\n",
            "app.py": "def go(conn: Database):\n    conn.connect()\n"})
        assert ("app_go", "db_database_connect") in links(root)

    def test_self_attribute_typed_in_init(self, corpus):
        root = corpus({
            "db.py": "class Database:\n    def connect(self):\n        pass\n",
            "app.py": ("from db import Database\n"
                       "class Service:\n"
                       "    def __init__(self):\n"
                       "        self.db = Database()\n"
                       "    def go(self):\n"
                       "        self.db.connect()\n")})
        assert ("app_service_go", "db_database_connect") in links(root)

    def test_typed_receiver_without_that_method_is_refused(self, corpus):
        # Knowing conn is a Database proves nothing if Database has no such
        # method. This is the line between an inference and a guess.
        root = corpus({
            "db.py": "class Database:\n    def connect(self):\n        pass\n",
            "app.py": ("from db import Database\n"
                       "def go():\n"
                       "    conn = Database()\n"
                       "    conn.explode()\n")})
        _, reasons = edges_of(root)
        assert reasons["receiver typed, but no such method in the corpus"] == 1

    def test_untyped_receiver_is_refused(self, corpus):
        root = corpus({"m.py": "def go(thing):\n    thing.run()\n"})
        _, reasons = edges_of(root)
        assert reasons["receiver type unknown"] == 1


class TestUnresolvedEdges:
    def test_drawn_when_a_symbol_of_that_name_exists(self, corpus):
        # We know a plausible target and failed to prove the link -- worth an
        # agent's attention.
        root = corpus({
            "lib.py": "class Other:\n    def run(self):\n        pass\n",
            "app.py": "def go(thing):\n    thing.run()\n"})
        got, _ = edges_of(root, resolved=False)
        assert ("app_go", "?run", "receiver type unknown") in got

    def test_not_drawn_for_names_that_exist_nowhere(self, corpus):
        # data.get() and lines.append() point at nothing in the corpus. Drawing
        # them would drown the map the query budget has to fit.
        root = corpus({"app.py": "def go(data):\n    data.get('k')\n"})
        got, reasons = edges_of(root, resolved=False)
        assert got == []
        assert reasons["receiver type unknown"] == 1


class TestImports:
    def test_module_inside_the_corpus_resolves_to_its_file(self, corpus):
        root = corpus({"lib.py": "x = 1\n", "app.py": "from lib import x\n"})
        assert ("app", "lib") in links(root, "imports")

    def test_module_outside_the_corpus_is_drawn_unresolved(self, corpus):
        # Which third-party libraries a file depends on is a real question, and
        # answering "none" would be a lie.
        root = corpus({"app.py": "import json\n"})
        got, _ = edges_of(root, "imports", resolved=False)
        assert ("app", "?json", "module outside this corpus (stdlib or third party)") in got

    def test_one_edge_per_statement_not_per_name(self, corpus):
        root = corpus({"lib.py": "a = 1\nb = 2\n",
                       "app.py": "from lib import a, b\n"})
        got, _ = edges_of(root, "imports")
        assert len([e for e in got if e[0] == "app"]) == 1


class TestInvariants:
    """Properties that must hold for any corpus, not facts about one."""

    def test_a_resolved_edge_never_points_at_a_missing_node(self, corpus):
        # Shipped bug: the same-file rule rebuilt an id from prefix + name, but
        # a nested function's real id carries its enclosing chain, so the edge
        # pointed at an id nobody owned.
        root = corpus({"m.py": (
            "def outer():\n"
            "    def helper():\n"
            "        pass\n"
            "    helper()\n")})
        files, _ = parse_corpus_files(root)
        ids = {n.id for p in files for n in p.nodes}
        edges, _ = resolve(files)
        for edge in edges:
            if edge.resolved:
                assert edge.target in ids, f"{edge.source} -> {edge.target} points nowhere"

    def test_nested_call_resolves_to_the_nested_function(self, corpus):
        root = corpus({"m.py": (
            "def outer():\n"
            "    def helper():\n"
            "        pass\n"
            "    helper()\n")})
        assert ("m_outer", "m_outer_helper") in links(root)

    def test_same_name_in_two_scopes_of_one_file_is_refused(self, corpus):
        root = corpus({"m.py": (
            "def a():\n"
            "    def helper():\n"
            "        pass\n"
            "def b():\n"
            "    def helper():\n"
            "        pass\n"
            "    helper()\n")})
        _, reasons = edges_of(root)
        assert any("scopes of this file" in r for r in reasons)
