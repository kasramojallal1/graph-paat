"""The document reader: what it finds, in what order, and what it refuses.

The rules that regress silently are the ordering and the budget, because both
fail by producing *less* rather than by producing an error -- a smaller read
still looks like a successful one.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from graphpaat import documents as D


def write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# finding documents


def test_collect_finds_each_document_extension(tmp_path):
    for rel in ("a.md", "b.rst", "c.txt", "d.markdown"):
        write(tmp_path, rel, "x" * 200)
    write(tmp_path, "e.py", "x" * 200)
    write(tmp_path, "f.png", "x" * 200)
    found = {p.name for p in D.collect(tmp_path)}
    assert found == {"a.md", "b.rst", "c.txt", "d.markdown"}


def test_collect_skips_noise_directories(tmp_path):
    write(tmp_path, "keep.md", "x" * 200)
    write(tmp_path, "node_modules/pkg/readme.md", "x" * 200)
    write(tmp_path, ".git/hooks/note.md", "x" * 200)
    write(tmp_path, "build/out.md", "x" * 200)
    assert [p.name for p in D.collect(tmp_path)] == ["keep.md"]


def test_readme_first_then_docs_then_outward(tmp_path):
    write(tmp_path, "zzz/deep/other.md", "x" * 200)
    write(tmp_path, "docs/guide.md", "x" * 200)
    write(tmp_path, "README.md", "x" * 200)
    order = [str(p.relative_to(tmp_path)) for p in D.collect(tmp_path)]
    assert order[0] == "README.md"
    assert order[1] == "docs/guide.md"


def test_history_files_sort_last(tmp_path):
    """A changelog is the largest and least informative file in most repos.

    Measured on one project: `CHANGELOG.md` alone is 99,484 tokens, half the whole
    prose budget, spent on what changed rather than on what anything is.
    """
    write(tmp_path, "CHANGELOG.md", "x" * 200)
    write(tmp_path, "zzz/deep/nested/thing.md", "x" * 200)
    order = [str(p.relative_to(tmp_path)) for p in D.collect(tmp_path)]
    assert order[-1] == "CHANGELOG.md"


# --------------------------------------------------------------------------
# splitting into claims


def test_markdown_splits_on_headings_with_line_numbers():
    text = "\n".join([
        "# Title",                    # L1
        "",
        "Intro prose that is long enough to survive the minimum length rule.",
        "",
        "## Second",                  # L5
        "",
        "More prose here, also comfortably past the minimum section length.",
    ])
    sections = D.split(text, ".md")
    assert [s.title for s in sections] == ["Title", "Second"]
    assert [s.line for s in sections] == [1, 5]


def test_a_hash_inside_a_fenced_block_is_not_a_heading():
    """Every README's shell examples begin with `#`. Without the fence check
    each one starts a new section and the document shatters."""
    text = "\n".join([
        "# Real",
        "",
        "```bash",
        "# this is a shell comment, not a heading",
        "graph-paat build .",
        "```",
        "",
        "Prose after the block, long enough to keep the section alive.",
    ])
    sections = D.split(text, ".md")
    assert [s.title for s in sections] == ["Real"]


def test_rst_splits_on_underlined_headings():
    text = "\n".join([
        "Solvers",
        "=======",
        "",
        "Prose about solving equations, long enough to pass the length floor.",
        "",
        "Matrices",
        "--------",
        "",
        "Prose about matrices, also long enough to pass the length floor here.",
    ])
    sections = D.split(text, ".rst")
    assert [s.title for s in sections] == ["Solvers", "Matrices"]


def test_rst_table_rules_are_not_headings():
    """A row of dashes under a short cell looks exactly like an underline."""
    text = "\n".join([
        "==== ====",
        "a    b",
        "==== ====",
        "",
        "Real prose in this document, long enough to be kept as a section.",
    ])
    sections = D.split(text, ".rst")
    assert all(s.title != "a    b" for s in sections)


def test_text_before_the_first_heading_is_kept():
    text = "\n".join([
        "This project does a thing, and this line is the only summary of it.",
        "",
        "# Install",
        "",
        "Run the installer, in a sentence long enough to be a real section.",
    ])
    sections = D.split(text, ".md")
    assert sections[0].line == 1
    assert "only summary" in sections[0].text


def test_a_long_section_is_split_at_paragraph_boundaries():
    body = "\n\n".join(f"Paragraph number {i} " + "word " * 60 for i in range(8))
    sections = D.split(f"# Big\n\n{body}", ".md")
    assert len(sections) > 1
    assert all(len(s.text) <= D.MAX_SECTION_CHARS * 1.5 for s in sections)
    assert all(s.title == "Big" for s in sections)


def test_a_document_with_no_headings_still_yields_sections():
    sections = D.split("Just prose, no headings at all, but plenty of it here.", ".txt")
    assert len(sections) == 1
    assert sections[0].line == 1


def test_trivially_short_sections_are_dropped():
    assert D.split("# T\n\nok\n", ".md") == []


# --------------------------------------------------------------------------
# the budget, and saying what it excluded


def test_budget_stops_reading_and_records_what_it_skipped(tmp_path):
    write(tmp_path, "README.md", "readme prose. " * 200)
    for i in range(5):
        write(tmp_path, f"docs/big{i}.md", f"chapter {i} " + "prose " * 2000)
    reading = D.read(tmp_path, budget_tokens=1000)
    assert reading.tokens_read <= 1000
    assert reading.skipped, "a capped read must say what it left out"
    assert all(len(row) == 3 for row in reading.skipped)
    assert reading.tokens_skipped > 0
    # The README is the most informative file, so the budget buys it first.
    assert reading.documents[0].path == "README.md"


def test_an_uncapped_read_skips_nothing(tmp_path):
    write(tmp_path, "README.md", "prose enough to count as a real section here.")
    reading = D.read(tmp_path, budget_tokens=10_000_000)
    assert reading.skipped == []
    assert len(reading.documents) == 1


def test_near_duplicate_documents_are_skipped_with_a_reason(tmp_path):
    """One page translated thirty-one times is one page.

    Prose changes language; `pip install sympy` does not, so the fingerprint is
    the identifiers and commands rather than the words around them.
    """
    marks = "\n".join(f"Run `command_{i} --flag` and see https://example.com/p{i}"
                      for i in range(20))
    write(tmp_path, "README.md", f"# English\n\nThe English explanation.\n\n{marks}")
    write(tmp_path, "docs/README.fr.md", f"# Francais\n\nL'explication en francais.\n\n{marks}")
    reading = D.read(tmp_path)
    assert [d.path for d in reading.documents] == ["README.md"]
    assert reading.skipped[0][0] == "docs/README.fr.md"
    assert "near-duplicate" in reading.skipped[0][2]


def test_two_different_documents_are_not_called_duplicates(tmp_path):
    a = "\n".join(f"Call `alpha_{i}()` here." for i in range(20))
    b = "\n".join(f"Call `beta_{i}()` here." for i in range(20))
    write(tmp_path, "README.md", f"# A\n\nFirst document body.\n\n{a}")
    write(tmp_path, "docs/other.md", f"# B\n\nSecond document body.\n\n{b}")
    reading = D.read(tmp_path)
    assert len(reading.documents) == 2


def test_a_tiny_fingerprint_is_never_grounds_for_duplication(tmp_path):
    """Two short pages that both say `run` are not the same page."""
    write(tmp_path, "README.md",
          "# A\n\nThis first document mentions `run` exactly once, and nothing else.")
    write(tmp_path, "docs/b.md",
          "# B\n\nThis second document also mentions `run`, and is about something else.")
    reading = D.read(tmp_path)
    assert len(reading.documents) == 2


# --------------------------------------------------------------------------
# PDFs, and failing to read one


def test_a_missing_pdf_library_is_reported_not_swallowed(tmp_path, monkeypatch):
    """A PDF we cannot read is a gap, and a gap gets a line in the report."""
    import builtins
    real_import = builtins.__import__

    def no_pypdf(name, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("no pypdf")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pypdf)
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.4 not really a pdf")
    reading = D.read(tmp_path)
    assert reading.documents == []
    assert reading.unreadable == [("paper.pdf", "no PDF support: pip install 'graph-paat[pdf]'")]


def test_a_corrupt_pdf_does_not_take_the_build_down(tmp_path):
    pytest.importorskip("pypdf")
    (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.4\nthis is not a pdf body")
    reading = D.read(tmp_path)
    assert reading.documents == []
    assert len(reading.unreadable) == 1
    assert reading.unreadable[0][0] == "broken.pdf"


def test_reading_is_deterministic(tmp_path):
    """Two reads of one tree give the same documents in the same order."""
    for i in range(12):
        write(tmp_path, f"docs/page{i}.md", f"# Page {i}\n\n" + f"prose about topic {i} " * 20)
    write(tmp_path, "README.md", "# Top\n\n" + "the summary of this project " * 20)
    first, second = D.read(tmp_path), D.read(tmp_path)
    assert [(d.path, [(s.title, s.line) for s in d.sections]) for d in first.documents] == \
           [(d.path, [(s.title, s.line) for s in d.sections]) for d in second.documents]
