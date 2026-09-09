"""Answering: matching a term, walking outward, and staying inside the budget."""
from graphpaat import store
from graphpaat.ids import Collisions
from graphpaat.parse import parse_corpus_files
from graphpaat.query import (_query_terms, estimate_tokens, forms, match,
                             neighbourhood, render, stem, vocabulary, word_vocabulary,
                             words)
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
        """A docstring is a claim -- and D17 says where the claim came from.

        A docstring is prose, so it is a claim; but a parser read it out of the
        source file this build, which is a stronger statement than a sentence
        in a README, and the tag has to say which.
        """
        graph = graph_of(corpus(SAMPLE), tmp_path / "out")
        out = render(graph, match(graph, ["Widget"])[0], budget=2000)
        assert "[claim \u00b7 read from code]" in out
        assert "[class \u00b7 read from code]" in out

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
        graph = graph_of(corpus({f"f{i}.py": f"def thing_{i}():\n    pass\n"
                                 for i in range(9)}), tmp_path / "out")
        seeds, more = match(graph, ["thing"], limit=3)
        assert len(seeds) == 3 and more == 6

    def test_one_name_cannot_fill_the_whole_answer(self, corpus, tmp_path):
        # zod ships forty locale files each defining `error`; three of them took
        # the entire answer and said nothing the first one had not.
        graph = graph_of(corpus({f"loc{i}.py": "def error():\n    pass\n"
                                 for i in range(9)}), tmp_path / "out")
        seeds, more = match(graph, ["error"], limit=3)
        assert len(seeds) == 2, "two seats per name, so the ambiguity is still visible"
        assert more == 7

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
        graph["overview"] = {"groups": [
            {"group": 0, "name": "rendering \u00b7 Widget", "size": 3}]}
        for n in graph["nodes"]:
            n["group"] = 0
        assert "part of: rendering" in render(graph, match(graph, ["Widget"])[0], budget=2000)


class TestGroupLabels:
    def test_a_group_covering_most_of_the_corpus_is_not_used_as_a_label(self, corpus, tmp_path):
        # Django's largest group holds 2,811 nodes and is named after
        # ValidationError. Telling the reader of an admin view that it is "part
        # of ValidationError" is worse than saying nothing.
        graph = graph_of(corpus(SAMPLE), tmp_path / "out")
        # Above both the absolute floor and the share threshold.
        graph["overview"] = {"groups": [{"group": 0, "name": "everything", "size": 5000}]}
        for n in graph["nodes"]:
            n["group"] = 0
        assert "part of: everything" not in render(graph, match(graph, ["Widget"])[0],
                                                  budget=2000)


class TestTermAgreement:
    def test_terms_in_one_question_reinforce_each_other(self, corpus, tmp_path):
        # Two symbols share a name; the one beside the other term wins, even
        # though the unrelated one is better connected.
        graph = graph_of(corpus({
            "orm/query.py": ("class QuerySet:\n    pass\n"
                             "def filter(qs):\n    pass\n"),
            "web/tags.py": ("def filter(x):\n    pass\n"
                            "def a():\n    filter(1)\n"
                            "def b():\n    filter(2)\n"
                            "def c():\n    filter(3)\n")}), tmp_path / "out")
        alone, _ = match(graph, ["filter"])
        assert alone[0] == "web_tags_filter"          # better connected wins alone
        together, _ = match(graph, ["QuerySet", "filter"])
        assert "orm_query_filter" in together[:2]     # the neighbour wins together


class TestGroupLabelHonesty:
    def test_group_label_hidden_when_the_node_lives_elsewhere(self, corpus, tmp_path):
        # A group is named after where most of it lives. For a member living
        # somewhere else that name is simply false.
        graph = graph_of(corpus(SAMPLE), tmp_path / "out")
        graph["overview"] = {"groups": [
            {"group": 0, "name": "tests \u00b7 TestCase", "size": 5,
             "named_folder": "tests"}]}
        for n in graph["nodes"]:
            n["group"] = 0
        assert "part of: tests" not in render(graph, match(graph, ["Widget"])[0], budget=2000)

    def test_group_label_shown_when_the_node_lives_there(self, corpus, tmp_path):
        graph = graph_of(corpus(SAMPLE), tmp_path / "out")
        folder = graph["nodes"][0]["file"].rsplit("/", 1)[0] if "/" in graph["nodes"][0]["file"] else "."
        graph["overview"] = {"groups": [
            {"group": 0, "name": "core \u00b7 Widget", "size": 5,
             "named_folder": folder}]}
        for n in graph["nodes"]:
            n["group"] = 0
        assert "part of: core" in render(graph, match(graph, ["Widget"])[0], budget=2000)


class TestWordsAndSpellings:
    """A question is English; a codebase is named in stems. These meet them."""

    def test_a_name_splits_on_underscores(self):
        assert words("classify_file") == ["classify", "file"]

    def test_a_name_splits_on_case_changes(self):
        assert words("NewClient") == ["new", "client"]

    def test_an_acronym_stays_whole(self):
        assert words("HTTPAdapter") == ["http", "adapter"]

    def test_a_past_participle_reaches_its_verb(self):
        assert stem("classified") == "classify"

    def test_queried_reaches_query(self):
        # This one is why the rule exists: `QuerySet` must answer a question
        # about how rows are queried.
        assert stem("queried") == "query"

    def test_a_gentle_stem_leaves_a_real_word_alone(self):
        # An earlier version stripped trailing vowels and merged `serve`,
        # `server` and `service` into one term.
        assert stem("database") == "database"
        assert stem("service") == "service"

    def test_a_plural_carries_both_spellings(self):
        # `cookies` stems to `cooky`, but the symbol is spelled `Cookie`.
        # Keeping every spelling is what lets the two meet.
        assert "cookie" in forms("cookies")
        assert "cooky" in forms("cookies")


class TestQueryTerms:

    def test_ordinary_english_is_dropped(self):
        terms, _ = _query_terms("what is the base class for a database model".split())
        assert "a" not in terms and "is" not in terms and "the" not in terms

    def test_a_code_verb_is_not_dropped(self):
        # `get` and `send` are filler in English and method names everywhere.
        terms, _ = _query_terms("how do I send a get request".split())
        assert "send" in terms and "get" in terms

    def test_a_question_of_pure_filler_still_searches_something(self):
        terms, _ = _query_terms("how does it work".split())
        assert terms

    def test_the_kind_asked_for_is_recognised_and_removed(self):
        terms, kind = _query_terms("what class holds the response body".split())
        assert kind == "class" and "class" not in terms


class TestRankingRules:
    """Each rule below cost questions when it was removed. One test each."""

    def test_an_article_cannot_win_on_an_exact_match(self, corpus, tmp_path):
        # Django has a class called `A`. "a database model" must not seed on it.
        graph = graph_of(corpus({"fmt.py": "class A:\n    pass\n",
                                 "db.py": "class Model:\n    pass\n"}), tmp_path / "out")
        seeds, _ = match(graph, "what is the base class for a model".split())
        assert seeds[0] == "db_model"

    def test_matching_more_of_the_question_beats_being_better_connected(self, corpus, tmp_path):
        # The failure that started the rewrite: `classify_file` matches both
        # words, `_file_stem` matches one and has far more callers.
        graph = graph_of(corpus({
            "detect.py": "def classify_file(p):\n    pass\n",
            "base.py": ("def _file_stem(p):\n    pass\n"
                        "def a():\n    _file_stem(1)\n"
                        "def b():\n    _file_stem(2)\n"
                        "def c():\n    _file_stem(3)\n")}), tmp_path / "out")
        seeds, _ = match(graph, ["classify", "file"])
        assert seeds[0] == "detect_classify_file"

    def test_an_english_question_reaches_a_stemmed_name(self, corpus, tmp_path):
        graph = graph_of(corpus({"detect.py": "def classify_file(p):\n    pass\n"}),
                         tmp_path / "out")
        seeds, _ = match(graph, "how are files classified".split())
        assert seeds[0] == "detect_classify_file"

    def test_a_short_name_beats_a_long_one_matching_the_same_words(self, corpus, tmp_path):
        graph = graph_of(corpus({
            "q.py": "class QuerySet:\n    pass\n",
            "ops.py": "def fetch_returned_insert_rows_query(x):\n    pass\n"}),
            tmp_path / "out")
        seeds, _ = match(graph, ["query"])
        assert seeds[0] == "q_queryset"

    def test_a_test_file_is_not_the_answer(self, corpus, tmp_path):
        # Test names are English sentences built from the words a question uses.
        graph = graph_of(corpus({
            "server.go_test.py": "def test_client_connects_to_server():\n    pass\n",
            "conn.py": "def connect_client():\n    pass\n"}), tmp_path / "out")
        seeds, _ = match(graph, ["client", "connect"])
        assert seeds[0] == "conn_connect_client"

    def test_a_docstring_answers_when_no_name_does(self, corpus, tmp_path):
        # "how are connections pooled" shares no word with `HTTPAdapter`; its
        # docstring is where "connection pooling" is written down.
        graph = graph_of(corpus({"a.py": (
            "class HTTPAdapter:\n"
            "    '''Handles connection pooling and reuse.'''\n"
            "    pass\n"
            "class Session:\n    '''A session.'''\n    pass\n")}), tmp_path / "out")
        seeds, _ = match(graph, ["connection", "pooling"])
        assert seeds[0] == "a_httpadapter"

    def test_a_rarer_word_carries_more_weight(self, corpus, tmp_path):
        # `file` is everywhere and says little; `queryset` says everything.
        graph = graph_of(corpus({
            "a.py": "def file_one():\n    pass\ndef file_two():\n    pass\n"
                    "def file_three():\n    pass\ndef queryset_file():\n    pass\n"}),
            tmp_path / "out")
        seeds, _ = match(graph, ["file", "queryset"])
        assert seeds[0] == "a_queryset_file"


class TestWordVocabulary:
    """The list an agent reads before choosing search terms."""

    def test_it_gives_words_not_names(self, corpus, tmp_path):
        graph = graph_of(corpus({"m.py": "def classify_file(p):\n    pass\n"}),
                         tmp_path / "out")
        found = word_vocabulary(graph)
        assert "classify" in found and "file" in found
        assert "classify_file" not in found

    def test_it_splits_camel_case_too(self, corpus, tmp_path):
        graph = graph_of(corpus({"m.py": "class NewClient:\n    pass\n"}),
                         tmp_path / "out")
        assert {"new", "client"} <= set(word_vocabulary(graph))

    def test_it_drops_words_too_short_to_search_with(self, corpus, tmp_path):
        graph = graph_of(corpus({"m.py": "def a_b_thing():\n    pass\n"}),
                         tmp_path / "out")
        found = word_vocabulary(graph)
        assert "thing" in found and "a" not in found and "b" not in found

    def test_each_word_appears_once(self, corpus, tmp_path):
        graph = graph_of(corpus({"m.py": ("def read_file():\n    pass\n"
                                          "def write_file():\n    pass\n")}),
                         tmp_path / "out")
        found = word_vocabulary(graph)
        assert found.count("file") == 1

    def test_a_docstring_is_not_vocabulary(self, corpus, tmp_path):
        # Docstrings are searched, but their bookkeeping labels are not names.
        graph = graph_of(corpus({"m.py": "def go():\n    '''hello'''\n"}),
                         tmp_path / "out")
        assert not any("docstring" in w for w in word_vocabulary(graph))
