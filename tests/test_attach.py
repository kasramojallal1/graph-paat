"""Attaching prose to code, and refusing to invent anything.

The rule under test is the one the whole design rests on: **the model picks from
a list we hand it, and may never type an identifier.** The two lanes join on
exact id string match, so an id that is nearly right does not make a nearly
right graph -- it makes a second node that nothing will ever reconcile with the
first.

The other rule under test is that a varying answer produces an unvarying graph.
A model does not return byte-identical text twice, so if the ingest did not
normalise order, repeats and stray whitespace away, `--deep` would not be
shippable at all.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphpaat import attach, documents
from graphpaat.build import assemble, with_documents

CODE = '''\
class Console:
    """Writes to the terminal."""

    def print(self, text):
        return self._render(text)

    def _render(self, text):
        return text


class Parser:
    """Reads a file."""

    def parse(self, source):
        return source
'''

PROSE = """\
# Toy

## Output

`Console` is the entry point for all output. Everything that reaches a terminal
goes through it, and nothing else writes to the screen directly.

## Reading

The `Parser` turns a file into a tree, and does nothing else at all.

## Installing

Run the installer and you are finished; there is nothing here to configure.
"""


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "console.py").write_text(CODE, encoding="utf-8")
    (tmp_path / "README.md").write_text(PROSE, encoding="utf-8")
    return tmp_path


@pytest.fixture
def prepared(repo):
    """A built code graph, the documents read, and the ask made of the model."""
    built = assemble(repo, cluster=False)
    reading = documents.read(repo)
    ask = attach.build_ask(reading, built.payload())
    return built, reading, ask


def answer(ask, picks):
    return {"ask": ask.digest, "picks": picks}


def only(ask, fragment):
    """The one claim id whose passage mentions `fragment`."""
    hits = [cid for cid, e in ask.manifest.items() if fragment in e["title"]]
    assert len(hits) == 1, f"expected one passage titled {fragment!r}, got {hits}"
    return hits[0]


# --------------------------------------------------------------------------
# building the closed list


def test_a_passage_naming_a_symbol_gets_it_as_a_candidate(prepared):
    _, _, ask = prepared
    cid = only(ask, "Output")
    labels = ask.manifest[cid]["candidates"]
    assert any("console" in c for c in labels)


def test_a_passage_naming_nothing_is_never_put_to_the_model(prepared):
    """Most prose describes no symbol. Asking anyway invites a wrong pick."""
    _, _, ask = prepared
    assert not any("Installing" in e["title"] for e in ask.manifest.values())


def test_a_reference_table_is_dropped_but_a_busy_passage_is_only_trimmed():
    """An API index lists symbols; it explains none of them. A page that merely
    names a dozen is still explaining something, and gets a ranked menu.

    One threshold used to do both jobs and could not: at twelve it dropped 8% of
    one project's passages and 44% of sympy's, because sympy documents mathematics
    and names far more symbols per page. Dropping nearly half of a repository's
    prose to save tokens would have thrown away real explanations.
    """
    index = {f"symbol_{i}": [f"mod_symbol_{i}"] for i in range(80)}

    few = " ".join(f"`symbol_{i}`" for i in range(4))
    assert len(attach.candidates(few, index)) == 4

    # Busy, but not an index: ranked and cut to the menu size, never dropped.
    busy = " ".join(f"`symbol_{i}`" for i in range(attach.MAX_CANDIDATES + 5))
    assert len(attach.candidates(busy, index)) == attach.MAX_CANDIDATES

    # A genuine reference table never reaches the model.
    table = " ".join(f"`symbol_{i}`" for i in range(attach.INDEX_MENTIONS + 1))
    assert attach.candidates(table, index) == []


def test_the_menu_is_ranked_by_how_strongly_a_passage_names_a_symbol():
    """Named in the heading beats named in prose beats used in an example."""
    index = {"alpha": ["mod_alpha"], "beta": ["mod_beta"], "gamma": ["mod_gamma"]}
    text = "```\ngamma()\n```\n\nThe `beta` helper does the work."
    assert attach.candidates(text, index, title="alpha") == [
        "mod_alpha", "mod_beta", "mod_gamma"]


def test_a_plain_english_word_in_prose_is_not_a_mention():
    """`changes` is a word; `changes()` and `Changes` are symbols.

    An earlier version took every four-letter identifier outside a code block,
    which made "documentation" and "provides" into mentions and gave every
    passage in both test repositories a strong mention of something.
    """
    found = attach.mentioned("This documentation provides changes to the parser.")
    assert "documentation" not in found
    assert "provides" not in found
    assert "changes" not in found and "parser" not in found

    # The same words, written the way an author writes a symbol.
    assert set(attach.mentioned("Call changes() on the Parser.")) >= {"changes", "parser"}

    # A capital at the start of a sentence is noise, and it is harmless: a
    # mention only becomes a candidate if a symbol really carries that name.
    assert attach.candidates("This provides changes.", {"parser": ["mod_parser"]}) == []


def test_a_name_inside_a_fenced_block_is_found():
    """The passage that describes a project's entire pipeline is a fenced block.

    Found by hand, not by any test here: a
    single-line backtick rule cannot see inside a fence, and a bare lowercase
    word is neither CamelCase nor snake_case, so `detect() -> extract() ->
    build()` produced no candidates at all.
    """
    found = attach.mentioned("```\ndetect() -> extract() -> build()\n```")
    assert set(found) == {"detect", "extract", "build"}


def test_the_candidate_list_is_the_same_on_two_runs(repo):
    built = assemble(repo, cluster=False)
    reading = documents.read(repo)
    first = attach.build_ask(reading, built.payload())
    second = attach.build_ask(reading, built.payload())
    assert first.manifest == second.manifest
    assert first.digest == second.digest
    assert first.text == second.text


# --------------------------------------------------------------------------
# The model picks from the list, or picks nothing


def test_an_id_outside_the_closed_list_is_rejected_not_created(prepared):
    """The rule the whole lane rests on.

    The easy thing to do here is mint a new node. Instead the pick is dropped,
    counted and reported, and no node is made.
    """
    _, reading, ask = prepared
    cid = only(ask, "Output")
    got = attach.ingest(answer(ask, {cid: ["console_Console_invented"]}),
                        ask.manifest, reading)
    assert got.attached == 0
    assert got.nodes == [] and got.edges == []
    assert got.rejected["id not on the candidate list"] == 1


def test_a_real_id_from_another_passage_is_still_rejected(prepared):
    """The list is closed *per passage*, not across the document.

    An id that exists in the graph is not thereby a legal answer here -- being
    real is not the same as being offered.
    """
    _, reading, ask = prepared
    output, reading_sec = only(ask, "Output"), only(ask, "Reading")
    stranger = ask.manifest[reading_sec]["candidates"][0]
    if stranger in ask.manifest[output]["candidates"]:
        pytest.skip("the two passages share this candidate")
    got = attach.ingest(answer(ask, {output: [stranger]}), ask.manifest, reading)
    assert got.attached == 0
    assert got.rejected["id not on the candidate list"] == 1


def test_picking_nothing_is_a_correct_answer(prepared):
    _, reading, ask = prepared
    cid = only(ask, "Output")
    got = attach.ingest(answer(ask, {cid: []}), ask.manifest, reading)
    assert got.attached == 0 and got.empty == 1
    assert got.rejected.get("id not on the candidate list") is None


def test_a_legal_pick_makes_exactly_one_claim_and_one_document(prepared):
    _, reading, ask = prepared
    cid = only(ask, "Output")
    pick = ask.manifest[cid]["candidates"][0]
    got = attach.ingest(answer(ask, {cid: [pick]}), ask.manifest, reading)
    kinds = sorted(n.kind for n in got.nodes)
    assert kinds == ["claim", "document"]
    assert got.attached == 1
    describes = [e for e in got.edges if e.relation == "describes"]
    assert len(describes) == 1 and describes[0].target == pick


def test_everything_minted_here_is_marked_as_coming_from_a_document(prepared):
    """A sentence someone wrote must never read as a parser fact."""
    _, reading, ask = prepared
    cid = only(ask, "Output")
    got = attach.ingest(answer(ask, {cid: ask.manifest[cid]["candidates"][:1]}),
                        ask.manifest, reading)
    assert {n.origin for n in got.nodes} == {"doc"}
    assert {e.origin for e in got.edges} == {"doc"}


def test_more_than_three_picks_are_capped(tmp_path):
    """A passage that fits four symbols is about a topic, not about a symbol."""
    code = "\n\n".join(f"class Widget{i}:\n    pass" for i in range(6))
    names = " ".join(f"`Widget{i}`" for i in range(6))
    (tmp_path / "w.py").write_text(code, encoding="utf-8")
    (tmp_path / "README.md").write_text(
        f"# W\n\n## All\n\nThe widgets {names} each do a different job entirely.\n",
        encoding="utf-8")

    built = assemble(tmp_path, cluster=False)
    reading = documents.read(tmp_path)
    ask = attach.build_ask(reading, built.payload())
    cid = max(ask.manifest, key=lambda c: len(ask.manifest[c]["candidates"]))
    options = ask.manifest[cid]["candidates"]
    assert len(options) >= 4, options
    got = attach.ingest(answer(ask, {cid: options}), ask.manifest, reading)
    assert len([e for e in got.edges if e.relation == "describes"]) == 3


def test_a_malformed_pick_is_counted_not_crashed(prepared):
    _, reading, ask = prepared
    cid = only(ask, "Output")
    got = attach.ingest(answer(ask, {cid: [42, None, {"id": "x"}]}),
                        ask.manifest, reading)
    assert got.attached == 0
    assert got.rejected["pick was not an id"] == 3


def test_an_unanswered_passage_is_reported(prepared):
    _, reading, ask = prepared
    got = attach.ingest(answer(ask, {}), ask.manifest, reading)
    assert got.unanswered == len(ask.manifest)


# --------------------------------------------------------------------------
# the same repo, built twice, gives the same graph


def test_reordering_the_picks_changes_nothing(prepared):
    _, reading, ask = prepared
    cid = max(ask.manifest, key=lambda c: len(ask.manifest[c]["candidates"]))
    options = ask.manifest[cid]["candidates"][:3]
    if len(options) < 2:
        pytest.skip("need two candidates to reorder")
    a = attach.ingest(answer(ask, {cid: options}), ask.manifest, reading)
    b = attach.ingest(answer(ask, {cid: list(reversed(options))}), ask.manifest, reading)
    assert [e.target for e in a.edges] == [e.target for e in b.edges]


def test_repeating_a_pick_attaches_it_once(prepared):
    _, reading, ask = prepared
    cid = only(ask, "Output")
    pick = ask.manifest[cid]["candidates"][0]
    got = attach.ingest(answer(ask, {cid: [pick, pick, pick]}), ask.manifest, reading)
    assert len([e for e in got.edges if e.relation == "describes"]) == 1


def test_stray_whitespace_around_an_id_is_normalised(prepared):
    _, reading, ask = prepared
    cid = only(ask, "Output")
    pick = ask.manifest[cid]["candidates"][0]
    got = attach.ingest(answer(ask, {cid: [f"  {pick}\n"]}), ask.manifest, reading)
    assert got.attached == 1


def test_two_deep_builds_of_one_repo_give_an_identical_graph(repo, tmp_path):
    """The gate. If the model's answer varies, the ingest normalises it away."""
    from dataclasses import asdict

    def build_once(picks_order):
        built = assemble(repo, cluster=False)
        reading = documents.read(repo)
        ask = attach.build_ask(reading, built.payload())
        picks = {cid: picks_order(e["candidates"][:2])
                 for cid, e in ask.manifest.items()}
        got = attach.ingest(answer(ask, picks), ask.manifest, reading)
        with_documents(built, reading, got)
        return built.payload()

    plain = build_once(lambda c: c)
    shuffled = build_once(lambda c: list(reversed(c)) + c)   # reordered and repeated
    assert json.dumps(plain, sort_keys=True) == json.dumps(shuffled, sort_keys=True)


def test_the_document_lane_never_duplicates_a_symbol(repo):
    """One symbol, one node, whichever lane found it."""
    built = assemble(repo, cluster=False)
    before = [n.id for n in built.nodes]
    reading = documents.read(repo)
    ask = attach.build_ask(reading, built.payload())
    picks = {cid: e["candidates"] for cid, e in ask.manifest.items()}
    got = attach.ingest(answer(ask, picks), ask.manifest, reading)
    with_documents(built, reading, got)

    after = [n.id for n in built.nodes]
    assert len(after) == len(set(after)), "the deep lane minted a duplicate id"
    # Every code node survives untouched, and nothing new shadows one.
    assert set(before) <= set(after)
    assert {n.id for n in got.nodes}.isdisjoint(set(before))


def test_every_describes_edge_lands_on_a_node_that_exists(repo):
    """A dangling edge is the failure mode this design exists to prevent."""
    built = assemble(repo, cluster=False)
    reading = documents.read(repo)
    ask = attach.build_ask(reading, built.payload())
    got = attach.ingest(
        answer(ask, {cid: e["candidates"] for cid, e in ask.manifest.items()}),
        ask.manifest, reading)
    with_documents(built, reading, got)
    ids = {n.id for n in built.nodes}
    assert all(e.source in ids and e.target in ids for e in built.edges)


# --------------------------------------------------------------------------
# a stale answer is caught rather than applied


def test_an_answer_for_a_different_ask_is_refused(prepared, tmp_path):
    _, _, ask = prepared
    path = tmp_path / "document-answers.json"
    path.write_text(json.dumps({"ask": "0000000000000000", "picks": {}}))
    got, why = attach.load_answer(path, ask.digest)
    assert got == {}
    assert "different set of documents" in why


def test_a_matching_answer_is_accepted(prepared, tmp_path):
    _, _, ask = prepared
    path = tmp_path / "document-answers.json"
    path.write_text(json.dumps({"ask": ask.digest, "picks": {}}))
    got, why = attach.load_answer(path, ask.digest)
    assert why == "" and got["ask"] == ask.digest


def test_a_broken_answer_file_says_what_to_do(prepared, tmp_path):
    _, _, ask = prepared
    path = tmp_path / "document-answers.json"
    path.write_text("{not json")
    got, why = attach.load_answer(path, ask.digest)
    assert got == {} and "not valid JSON" in why


def test_a_missing_answer_file_is_not_an_error(prepared, tmp_path):
    _, _, ask = prepared
    got, why = attach.load_answer(tmp_path / "nope.json", ask.digest)
    assert got == {} and "no answer yet" in why


# --------------------------------------------------------------------------
# what a claim id looks like


def test_a_claim_id_can_never_collide_with_a_parsed_symbol():
    """`#` cannot appear in an identifier, which is the whole point of it."""
    cid = attach.claim_id("docs/guide.md", 42)
    assert "#" in cid
    assert cid == "docs_guide.md#document#L42"
    assert attach.document_id("docs/guide.md") == "docs_guide.md#document"
