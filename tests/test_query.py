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
        seeds, _ = match(graph, ["load"])
        assert seeds[0] == "m_load"

    def test_case_insensitive(self, corpus, tmp_path):
        graph = graph_of(corpus(SAMPLE), tmp_path / "out")
        seeds, _ = match(graph, ["widget"])
        assert "m_widget" in seeds

    def test_substring_when_nothing_exact(self, corpus, tmp_path):
        graph = graph_of(corpus(SAMPLE), tmp_path / "out")
        seeds, _ = match(graph, ["rend"])
        assert "m_widget_render" in seeds

    def test_unknown_term_matches_nothing(self, corpus, tmp_path):
        graph = graph_of(corpus(SAMPLE), tmp_path / "out")
        assert match(graph, ["nonexistent"]) == ([], 0)


class TestWalk:
    def test_an_edge_is_collected_once_not_once_per_end(self, corpus, tmp_path):
        # Shipped bug: edges were reachable from both ends and printed twice.
        graph = graph_of(corpus(SAMPLE), tmp_path / "out")
        _, travelled, _ = neighbourhood(graph, ["m_widget"], depth=2)
        keys = [(e["source"], e["target"], e["relation"]) for e in travelled]
        assert len(keys) == len(set(keys))

    def test_depth_limits_the_walk(self, corpus, tmp_path):
        graph = graph_of(corpus(SAMPLE), tmp_path / "out")
        near, _, _ = neighbourhood(graph, ["m_widget"], depth=1)
        far, _, _ = neighbourhood(graph, ["m_widget"], depth=3)
        assert len(near) < len(far)


class TestRender:
    def test_shows_both_directions(self, corpus, tmp_path):
        # Shipped bug: only outgoing edges were shown, so nothing said which
        # file a function lived in.
        graph = graph_of(corpus(SAMPLE), tmp_path / "out")
        out = render(graph, match(graph, ["render"])[0], budget=2000)
        assert "part of" in out or "called by" in out

    def test_docstrings_are_marked_as_claims_not_facts(self, corpus, tmp_path):
        graph = graph_of(corpus(SAMPLE), tmp_path / "out")
        out = render(graph, match(graph, ["Widget"])[0], budget=2000)
        assert "[claim]" in out

    def test_never_prints_source_code(self, corpus, tmp_path):
        graph = graph_of(corpus(SAMPLE), tmp_path / "out")
        out = render(graph, match(graph, ["Widget", "render", "helper"])[0], budget=2000)
        assert "def render" not in out and "pass" not in out

    def test_budget_truncates_and_says_so(self, corpus, tmp_path):
        graph = graph_of(corpus(SAMPLE), tmp_path / "out")
        seeds, _ = match(graph, ["Widget", "render", "paint", "helper"])
        out = render(graph, seeds, budget=1)
        assert "omitted" in out and "budget" in out

    def test_empty_result_points_at_vocab(self, corpus, tmp_path):
        graph = graph_of(corpus(SAMPLE), tmp_path / "out")
        assert "vocab" in render(graph, [], budget=2000)

    def test_stays_within_the_budget(self, corpus, tmp_path):
        graph = graph_of(corpus(SAMPLE), tmp_path / "out")
        seeds, _ = match(graph, ["Widget", "render", "paint", "helper"])
        out = render(graph, seeds, budget=60)
        reported = int(out.split("~")[1].split(" ")[0])
        assert reported <= 60


class TestTokenEstimate:
    def test_roughly_four_characters_per_token(self):
        assert estimate_tokens("a" * 400) == 100

    def test_never_zero(self):
        assert estimate_tokens("") == 1


class TestRanking:
    """The differences between a map and a dump."""

    def test_better_connected_symbol_leads_when_a_name_is_ambiguous(self, corpus, tmp_path):
        # Django has 30 methods named `get`. Returning them in discovery order
        # spends the whole budget on an arbitrary one.
        graph = graph_of(corpus({
            "hot.py": ("def handle():\n    pass\n"
                       "def a():\n    handle()\n"
                       "def b():\n    handle()\n"
                       "def c():\n    handle()\n"),
            "cold.py": "def handle():\n    pass\n"}), tmp_path / "out")
        seeds, _ = match(graph, ["handle"])
        assert seeds[0] == "hot_handle"

    def test_reports_how_many_more_matched(self, corpus, tmp_path):
        graph = graph_of(corpus({f"f{i}.py": "def thing():\n    pass\n" for i in range(9)}),
                         tmp_path / "out")
        seeds, more = match(graph, ["thing"], limit=3)
        assert len(seeds) == 3 and more == 6

    def test_relation_words_do_not_seed(self, corpus, tmp_path):
        # "what calls login" must not seat a root on `calls`.
        graph = graph_of(corpus({"m.py": "def calls():\n    pass\ndef login():\n    pass\n"}),
                         tmp_path / "out")
        seeds, _ = match(graph, ["what", "calls", "login"])
        assert seeds[0] == "m_login"

    def test_behaviour_is_shown_before_containment(self, corpus, tmp_path):
        graph = graph_of(corpus({"m.py": (
            "class W:\n"
            "    def go(self):\n"
            "        self.helper()\n"
            "    def helper(self):\n"
            "        pass\n")}), tmp_path / "out")
        out = render(graph, ["m_w_go"], budget=2000)
        body = out.split("\n")
        calls_at = next(i for i, l in enumerate(body) if "calls" in l)
        part_at = next(i for i, l in enumerate(body) if "part of" in l)
        assert calls_at < part_at

    def test_dunder_methods_rank_below_real_ones(self, corpus, tmp_path):
        graph = graph_of(corpus({"m.py": (
            "class W:\n"
            "    def __repr__(self):\n        pass\n"
            "    def __len__(self):\n        pass\n"
            "    def render(self):\n        pass\n")}), tmp_path / "out")
        out = render(graph, ["m_w"], budget=2000, per_node=2)
        assert "render" in out

    def test_a_hub_is_shown_but_not_expanded_through(self, corpus, tmp_path):
        # Two hops through a heavily used symbol reaches the whole repo.
        from graphpaat.query import HUB_DEGREE
        callers = "".join(f"def c{i}():\n    hub()\n" for i in range(HUB_DEGREE + 5))
        graph = graph_of(corpus({
            "m.py": "def hub():\n    pass\n" + callers,
            "far.py": "def start():\n    hub()\n"}), tmp_path / "out")
        reached, _, hubs = neighbourhood(graph, ["far_start"], depth=2)
        assert "m_hub" in hubs
        assert "m_c30" not in reached

    def test_group_context_is_shown_when_known(self, corpus, tmp_path):
        graph = graph_of(corpus(SAMPLE), tmp_path / "out")
        graph["overview"] = {"groups": [{"group": 0, "name": "rendering \u00b7 Widget"}]}
        for n in graph["nodes"]:
            n["group"] = 0
        assert "part of: rendering" in render(graph, match(graph, ["Widget"])[0], budget=2000)
