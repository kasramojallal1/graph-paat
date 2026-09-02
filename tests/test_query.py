"""Answering: matching a term, walking outward, and staying inside the budget."""
from graphpaat import store
from graphpaat.ids import Collisions
from graphpaat.parse import parse_corpus_files
from graphpaat.query import estimate_tokens, match, neighbourhood, render, vocabulary
from graphpaat.resolve import resolve


def graph_of(root, out):
    files, failed = parse_corpus_files(root)
    nodes, edges, collisions = [], [], Collisions()
    for parsed in files:
        for node in parsed.nodes:
            collisions.claim(node.id, f"{node.file}:L{node.line}")
            nodes.append(node)
        edges.extend(parsed.edges)
    call_edges, _ = resolve(files)
    edges.extend(call_edges)
    store.write(root, nodes, edges, collisions, failed, out=out)
    return store.read(root, out=out)


SAMPLE = {"m.py": (
    "class Widget:\n"
    "    '''A widget.'''\n"
    "    def render(self):\n"
    "        self.paint()\n"
    "    def paint(self):\n"
    "        pass\n"
    "def helper():\n"
    "    pass\n")}


class TestVocabulary:
    def test_publishes_real_names_only(self, corpus, tmp_path):
        graph = graph_of(corpus(SAMPLE), tmp_path / "out")
        names = vocabulary(graph)
        assert "Widget" in names and "render" in names
        # A docstring's label is bookkeeping, not a name an agent should pick.
        assert not any(n.startswith("docstring of") for n in names)


class TestMatch:
    def test_exact_before_substring(self, corpus, tmp_path):
        graph = graph_of(corpus({"m.py": "def load():\n    pass\ndef load_cached():\n    pass\n"}),
                         tmp_path / "out")
        assert match(graph, ["load"])[0] == "m_load"

    def test_case_insensitive(self, corpus, tmp_path):
        graph = graph_of(corpus(SAMPLE), tmp_path / "out")
        assert "m_widget" in match(graph, ["widget"])

    def test_substring_when_nothing_exact(self, corpus, tmp_path):
        graph = graph_of(corpus(SAMPLE), tmp_path / "out")
        assert "m_widget_render" in match(graph, ["rend"])

    def test_unknown_term_matches_nothing(self, corpus, tmp_path):
        graph = graph_of(corpus(SAMPLE), tmp_path / "out")
        assert match(graph, ["nonexistent"]) == []


class TestWalk:
    def test_an_edge_is_collected_once_not_once_per_end(self, corpus, tmp_path):
        # Shipped bug: edges were reachable from both ends and printed twice.
        graph = graph_of(corpus(SAMPLE), tmp_path / "out")
        _, travelled = neighbourhood(graph, ["m_widget"], depth=2)
        keys = [(e["source"], e["target"], e["relation"]) for e in travelled]
        assert len(keys) == len(set(keys))

    def test_depth_limits_the_walk(self, corpus, tmp_path):
        graph = graph_of(corpus(SAMPLE), tmp_path / "out")
        near, _ = neighbourhood(graph, ["m_widget"], depth=1)
        far, _ = neighbourhood(graph, ["m_widget"], depth=3)
        assert len(near) < len(far)


class TestRender:
    def test_shows_both_directions(self, corpus, tmp_path):
        # Shipped bug: only outgoing edges were shown, so nothing said which
        # file a function lived in.
        graph = graph_of(corpus(SAMPLE), tmp_path / "out")
        out = render(graph, match(graph, ["render"]), budget=2000)
        assert "part of" in out or "called by" in out

    def test_docstrings_are_marked_as_claims_not_facts(self, corpus, tmp_path):
        graph = graph_of(corpus(SAMPLE), tmp_path / "out")
        out = render(graph, match(graph, ["Widget"]), budget=2000)
        assert "[claim]" in out

    def test_never_prints_source_code(self, corpus, tmp_path):
        graph = graph_of(corpus(SAMPLE), tmp_path / "out")
        out = render(graph, match(graph, ["Widget", "render", "helper"]), budget=2000)
        assert "def render" not in out and "pass" not in out

    def test_budget_truncates_and_says_so(self, corpus, tmp_path):
        graph = graph_of(corpus(SAMPLE), tmp_path / "out")
        seeds = match(graph, ["Widget", "render", "paint", "helper"])
        out = render(graph, seeds, budget=1)
        assert "omitted" in out and "budget" in out

    def test_empty_result_points_at_vocab(self, corpus, tmp_path):
        graph = graph_of(corpus(SAMPLE), tmp_path / "out")
        assert "vocab" in render(graph, [], budget=2000)

    def test_stays_within_the_budget(self, corpus, tmp_path):
        graph = graph_of(corpus(SAMPLE), tmp_path / "out")
        seeds = match(graph, ["Widget", "render", "paint", "helper"])
        out = render(graph, seeds, budget=60)
        reported = int(out.split("~")[1].split(" ")[0])
        assert reported <= 60


class TestTokenEstimate:
    def test_roughly_four_characters_per_token(self):
        assert estimate_tokens("a" * 400) == 100

    def test_never_zero(self):
        assert estimate_tokens("") == 1
