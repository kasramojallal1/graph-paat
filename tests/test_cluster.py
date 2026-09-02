"""Grouping and ranking.

Clustering is easy to make *run* and hard to make *useful*. These tests are
mostly about usefulness: that groups are stable between runs, that they are
named after something a person can look up, and that a node with no connections
does not become a group of one.
"""
import pytest

from graphpaat.cluster import communities, god_nodes

nx = pytest.importorskip("networkx")


def graph(corpus_root):
    from dataclasses import asdict
    from graphpaat.parse import parse_corpus_files
    from graphpaat.resolve import resolve
    files, _ = parse_corpus_files(corpus_root)
    nodes = [asdict(n) for p in files for n in p.nodes]
    edges = [asdict(e) for p in files for e in p.edges]
    call_edges, _ = resolve(files)
    edges += [asdict(e) for e in call_edges]
    return nodes, edges


TWO_CLUSTERS = {
    "auth/login.py": (
        "def verify(u):\n    pass\n"
        "def login(u):\n    verify(u)\n"
        "def logout(u):\n    verify(u)\n"),
    "render/html.py": (
        "def escape(s):\n    pass\n"
        "def paragraph(s):\n    escape(s)\n"
        "def heading(s):\n    escape(s)\n"),
}


class TestCommunities:
    def test_finds_the_separate_parts(self, corpus):
        nodes, edges = graph(corpus(TWO_CLUSTERS))
        membership, _ = communities(nodes, edges)
        assert membership["auth_login_login"] == membership["auth_login_verify"]
        assert membership["render_html_heading"] == membership["render_html_escape"]
        assert membership["auth_login_login"] != membership["render_html_escape"]

    def test_is_stable_between_runs(self, corpus):
        # An agent asking twice must not get different group names. Sorted
        # insertion and a fixed seed; without them the partition can shift.
        nodes, edges = graph(corpus(TWO_CLUSTERS))
        first, summary_a = communities(nodes, edges)
        second, summary_b = communities(nodes, edges)
        assert first == second
        assert [g["name"] for g in summary_a["groups"]] == [g["name"] for g in summary_b["groups"]]

    def test_groups_are_named_after_something_lookupable(self, corpus):
        nodes, edges = graph(corpus(TWO_CLUSTERS))
        _, summary = communities(nodes, edges)
        names = " ".join(g["name"] for g in summary["groups"])
        assert "verify" in names or "escape" in names
        assert not any(g["name"].strip().isdigit() for g in summary["groups"])

    def test_a_hub_is_a_symbol_not_a_file(self, corpus):
        # A file connects to everything it contains, so it wins on degree while
        # naming nothing useful.
        nodes, edges = graph(corpus(TWO_CLUSTERS))
        _, summary = communities(nodes, edges)
        assert all(not (g["hub"] or "").endswith(".py") for g in summary["groups"])

    def test_unconnected_nodes_are_ungrouped_not_groups_of_one(self, corpus):
        # An empty file is genuinely isolated: it defines nothing to contain and
        # imports nothing. A file with a function in it is NOT isolated -- the
        # containment edge joins them, which is why the first version of this
        # test was wrong rather than the code.
        root = corpus(dict(TWO_CLUSTERS, **{"empty.py": "\n"}))
        nodes, edges = graph(root)
        _, summary = communities(nodes, edges)
        assert all(g["size"] > 1 for g in summary["groups"])
        assert summary["ungrouped"] >= 1

    def test_an_edgeless_corpus_does_not_crash(self, corpus):
        nodes, edges = graph(corpus({"m.py": "x = 1\n"}))
        membership, summary = communities(nodes, [e for e in edges if False])
        assert membership == {} and summary["groups"] == []


class TestGodNodes:
    def test_ranks_by_connections(self, corpus):
        nodes, edges = graph(corpus(TWO_CLUSTERS))
        top = god_nodes(nodes, edges, top=2)
        assert top[0]["connections"] >= top[1]["connections"]
        assert {n["label"] for n in top} <= {"verify", "escape", "login", "logout",
                                             "paragraph", "heading"}

    def test_excludes_files_which_touch_everything_they_hold(self, corpus):
        nodes, edges = graph(corpus(TWO_CLUSTERS))
        assert all(not n["label"].endswith(".py") for n in god_nodes(nodes, edges))

    def test_unresolved_edges_do_not_count_as_connections(self, corpus):
        # A gap is not evidence that two things are connected.
        root = corpus({"m.py": "def go(thing):\n    thing.connect()\n",
                       "o.py": "class C:\n    def connect(self):\n        pass\n"})
        nodes, edges = graph(root)
        ranked = {n["label"]: n["connections"] for n in god_nodes(nodes, edges)}
        assert ranked.get("go", 0) <= 1


class TestOrdering:
    def test_groups_are_listed_in_the_order_of_the_size_shown(self, corpus):
        # Sorting by raw membership while reporting a different count printed a
        # group of 370 above one of 394.
        nodes, edges = graph(corpus(TWO_CLUSTERS))
        _, summary = communities(nodes, edges)
        sizes = [g["size"] for g in summary["groups"]]
        assert sizes == sorted(sizes, reverse=True)


class TestGroupNaming:
    def test_an_exception_never_names_a_group(self, corpus):
        # An exception is connected to everything that raises it, so it wins on
        # degree while describing no subsystem. Django's largest group was
        # named ValidationError, then ImproperlyConfigured -- which does not
        # end in Error, so the name alone was not enough to spot it.
        root = corpus({"m.py": (
            "class ImproperlyConfigured(Exception):\n    pass\n"
            "class Engine:\n"
            "    def run(self):\n        boom()\n"
            "def boom():\n    raise ImproperlyConfigured()\n"
            "def a():\n    boom()\n"
            "def b():\n    boom()\n")})
        nodes, edges = graph(root)
        _, summary = communities(nodes, edges)
        assert all("ImproperlyConfigured" not in g["name"] for g in summary["groups"])

    def test_a_class_names_a_group_before_a_function(self, corpus):
        root = corpus({"m.py": (
            "class Engine:\n"
            "    def run(self):\n        helper()\n"
            "def helper():\n    pass\n"
            "def one():\n    helper()\n"
            "def two():\n    helper()\n")})
        nodes, edges = graph(root)
        _, summary = communities(nodes, edges)
        assert any("Engine" in g["name"] for g in summary["groups"])
