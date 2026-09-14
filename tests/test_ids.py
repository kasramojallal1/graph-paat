"""Identity rules.

These are tested hardest because everything depends on them. A node's id is how
edges find their target, how a re-run recognises a symbol it saw before, and how
two extraction lanes agree they are talking about the same thing. An id rule that
regresses does not produce a slightly wrong graph -- it produces duplicate or
merged nodes that nothing downstream can detect.

Every case below is a rule we arrived at by measurement, and several are bugs we
already shipped once.
"""
from pathlib import Path

from graphpaat.ids import Collisions, file_prefix, mint, normalise, normalise_path_part


class TestNormalise:
    """Symbol names keep their underscores; file names do not."""

    def test_lowercases(self):
        assert normalise("MinHashLSH") == "minhashlsh"

    def test_keeps_leading_underscore(self):
        # Django writes a public method
        # delegating to a private one; stripping merged them and destroyed 197
        # symbols.
        assert normalise("_changeform_view") == "_changeform_view"
        assert normalise("changeform_view") == "changeform_view"

    def test_private_and_public_are_different_names(self):
        assert normalise("_add_edge") != normalise("add_edge")

    def test_dunder_survives(self):
        assert normalise("__init__") == "__init__"

    def test_path_parts_still_strip(self):
        # A file's underscores carry no such meaning: __init__.py is in every
        # package directory, and _minhash.py is one module, not the private twin
        # of a minhash.py.
        assert normalise_path_part("__init__") == "init"
        assert normalise_path_part("_minhash") == "minhash"
        assert normalise_path_part("Detect") == "detect"


class TestFilePrefix:
    """The prefix is the whole relative path, not the filename."""

    def test_uses_full_relative_path(self):
        root = Path("/repo")
        assert file_prefix(root / "extractors" / "models.py", root) == "extractors_models"

    def test_bare_file(self):
        root = Path("/repo")
        assert file_prefix(root / "affected.py", root) == "affected"

    def test_two_utils_in_different_folders_do_not_collide(self):
        # The reason the prefix is the whole path. Using the filename alone was
        # the single largest source of disagreement in the first rebuild.
        root = Path("/repo")
        a = file_prefix(root / "a" / "utils.py", root)
        b = file_prefix(root / "b" / "utils.py", root)
        assert a != b

    def test_dunder_and_private_filenames(self):
        root = Path("/repo")
        assert file_prefix(root / "__main__.py", root) == "main"
        assert file_prefix(root / "_minhash.py", root) == "minhash"


class TestMint:
    def test_plain_symbol(self):
        assert mint("cache", "file_hash") == "cache_file_hash"

    def test_method_carries_its_class(self):
        # Without the class segment, two classes in one file both defining
        # __init__ mint one id.
        assert mint("cli", "__init__", ["StageTimer"]) == "cli_stagetimer___init__"

    def test_two_classes_same_method_differ(self):
        assert mint("m", "__init__", ["A"]) != mint("m", "__init__", ["B"])

    def test_nested_function_carries_its_enclosing_chain(self):
        # extract.py had three nested add_edge/_add_edge definitions minting one
        # id before scope qualification.
        outer = mint("extract", "add_edge", ["build_graph"])
        other = mint("extract", "add_edge", ["merge_graph"])
        assert outer != other

    def test_private_and_public_methods_of_one_class_differ(self):
        pub = mint("options", "changeform_view", ["ModelAdmin"])
        priv = mint("options", "_changeform_view", ["ModelAdmin"])
        assert pub != priv


class TestCollisions:
    def test_single_claim_is_not_a_collision(self):
        c = Collisions()
        c.claim("a_b", "a.py:L1")
        assert c.collided() == {}
        assert c.report() == []

    def test_two_claims_collide(self):
        c = Collisions()
        c.claim("detect_nfc", "detect.py:L970")
        c.claim("detect_nfc", "detect.py:L1865")
        assert list(c.collided()) == ["detect_nfc"]

    def test_report_names_kept_and_lost(self):
        c = Collisions()
        c.claim("x", "a.py:L1")
        c.claim("x", "a.py:L2")
        line = c.report()[0]
        # The point of the report is that the loser is named, not just counted.
        assert "a.py:L1" in line and "a.py:L2" in line

    def test_three_way_collision_reports_all_losers(self):
        c = Collisions()
        for line in ("L85", "L109", "L119"):
            c.claim("locks_lock", f"locks.py:{line}")
        assert len(c.collided()["locks_lock"]) == 3
        assert "L109" in c.report()[0] and "L119" in c.report()[0]
