"""The question set, as a test gate, plus tests for the scoring itself.

The gate compares every question against its recorded rank. A total is not
enough: one question improving while another breaks leaves the percentage flat,
and the percentage is what a person looks at.
"""
import json

import pytest

from tests.question_runner import (BASELINE, QUESTIONS, TOP_N, _compare, ask,
                                   load_questions, main, where)
from tests.corpus_runner import CONFIG


def graph(nodes):
    """A graph dict shaped like `Built.payload()`, small enough to reason about."""
    return {"nodes": nodes, "edges": [], "overview": {"groups": [], "god_nodes": []}}


def node(nid, label, file, kind="function", group=None):
    return {"id": nid, "label": label, "file": file, "line": 1,
            "kind": kind, "group": group}


class TestTheQuestionsThemselves:
    """The file is hand-written, so it is worth checking it is well formed."""

    def test_every_corpus_has_ten_questions(self):
        for name, questions in load_questions().items():
            assert len(questions) == 10, f"{name} has {len(questions)}"

    def test_every_question_names_at_least_one_answer_and_says_why(self):
        for name, questions in load_questions().items():
            for q in questions:
                assert q["answers"], f"{name}: {q['q']} names no answer"
                assert q["why"].strip(), f"{name}: {q['q']} says no why"

    def test_answers_are_written_as_file_colon_label(self):
        for name, questions in load_questions().items():
            for q in questions:
                for answer in q["answers"]:
                    assert ":" in answer, f"{name}: {answer} is not file:Label"

    def test_no_question_is_asked_twice_in_one_corpus(self):
        for name, questions in load_questions().items():
            asked = [q["q"] for q in questions]
            assert len(set(asked)) == len(asked), f"{name} repeats a question"

    def test_questions_are_asked_in_english_not_in_identifiers(self):
        """The gap between a question and an identifier is the thing being
        measured. A question written as `classify_file` measures nothing."""
        for name, questions in load_questions().items():
            for q in questions:
                assert "_" not in q["q"], f"{name}: {q['q']} is an identifier"
                assert len(q["q"].split()) >= 3, f"{name}: {q['q']} is too short"


class TestScoring:
    """The scorer decides whether the tool is improving, so it gets its own tests."""

    def test_the_named_answer_first_is_rank_zero(self):
        g = graph([node("a", "classify_file", "detect.py")])
        rank, got = ask(g, {"q": "classify file", "answers": ["detect.py:classify_file"]},
                        {n["id"]: n for n in g["nodes"]})
        assert rank == 0
        assert got == ["detect.py:classify_file"]

    def test_an_answer_further_down_keeps_its_rank(self):
        g = graph([node("a", "file", "one.py"), node("b", "classify", "two.py")])
        rank, _ = ask(g, {"q": "file classify", "answers": ["two.py:classify"]},
                      {n["id"]: n for n in g["nodes"]})
        assert rank >= 1

    def test_an_answer_that_never_appears_is_a_miss(self):
        g = graph([node("a", "unrelated", "one.py")])
        rank, got = ask(g, {"q": "classify the file", "answers": ["two.py:missing"]},
                        {n["id"]: n for n in g["nodes"]})
        assert rank == -1
        assert "two.py:missing" not in got

    def test_any_listed_answer_counts(self):
        """Two answers are listed only when both are genuinely right -- a
        library shipping two major versions side by side. Either one wins."""
        g = graph([node("a", "ZodString", "v4/schemas.ts")])
        rank, _ = ask(g, {"q": "how is a string schema defined",
                          "answers": ["v3/types.ts:ZodString", "v4/schemas.ts:ZodString"]},
                      {n["id"]: n for n in g["nodes"]})
        assert rank == 0

    def test_only_the_top_three_are_looked_at(self):
        assert TOP_N == 3


class TestRegressionRules:

    def test_a_question_that_drops_a_place_fails(self):
        assert _compare({"c": {"q": 1}}, {"c": {"q": 0}}) == 1

    def test_a_question_that_becomes_a_miss_fails(self):
        assert _compare({"c": {"q": -1}}, {"c": {"q": 2}}) == 1

    def test_a_question_that_improves_passes(self):
        assert _compare({"c": {"q": 0}}, {"c": {"q": 2}}) == 0

    def test_a_miss_that_becomes_an_answer_passes(self):
        assert _compare({"c": {"q": 1}}, {"c": {"q": -1}}) == 0

    def test_an_unchanged_set_passes(self):
        assert _compare({"c": {"q": 0, "r": -1}}, {"c": {"q": 0, "r": -1}}) == 0

    def test_a_question_not_in_the_baseline_is_not_a_regression(self):
        """A newly written question has no recorded rank yet."""
        assert _compare({"c": {"new": -1}}, {"c": {}}) == 0


class TestTheRecordedAnswers:

    def test_the_baseline_covers_every_question_that_could_be_asked(self):
        """A question with no recorded rank is unguarded: it can break without
        failing anything."""
        if not BASELINE.exists() or not CONFIG.exists():
            pytest.skip("no baseline or corpora configured")
        recorded = json.loads(BASELINE.read_text())
        configured = set(json.loads(CONFIG.read_text()))
        for name, questions in load_questions().items():
            if name not in configured or name not in recorded:
                continue
            for q in questions:
                assert q["q"] in recorded[name], f"{name}: {q['q']} is not recorded"


@pytest.mark.skipif(not CONFIG.exists() or not QUESTIONS.exists(),
                    reason="tests/corpora.json not configured")
def test_no_question_is_answered_worse_than_it_was():
    assert main([]) == 0, "a question regressed; see the diff above"
