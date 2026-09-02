"""Shared fixtures: build a tiny corpus on disk from a dict of source strings.

Tests read better when the code under test is visible in the test, so each one
writes the exact Python it is about rather than pointing at a shared sample.
"""
import pytest


@pytest.fixture
def corpus(tmp_path):
    """Write a corpus into tmp_path/repo.

    Nested in a subdirectory on purpose, so a test can put the output directory
    beside it rather than inside it -- the output must never land in the tree
    being analysed, and a fixture that made that impossible to express would
    hide the very thing worth testing.
    """
    def build(files: dict[str, str]):
        root = tmp_path / "repo"
        for name, source in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
        root.mkdir(parents=True, exist_ok=True)
        return root
    return build
