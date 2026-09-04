"""Installing the instructions.

This code writes into files a user owns and did not create for us. Most of
these tests are about restraint: touching only a marked region, never
clobbering, never creating files nobody asked for, and being reversible.
"""
from graphpaat.instructions import (BEGIN, END, INSTRUCTIONS, TARGETS, block,
                                    install, merge, uninstall)


class TestBlock:
    def test_is_marker_delimited(self):
        b = block()
        assert b.startswith(BEGIN) and b.rstrip().endswith(END)

    def test_explains_how_to_read_the_output_not_just_how_to_run_it(self):
        # A map that marks claims and gaps is only useful to a reader told what
        # the marks mean.
        for mark in ("[claim]", "?name", "(hub)", "vocab", "file:line"):
            assert mark in INSTRUCTIONS

    def test_stays_small_enough_to_carry_on_every_turn(self):
        # It is loaded into an assistant's context constantly.
        assert len(INSTRUCTIONS) < 2200, "instructions are getting expensive"


class TestMerge:
    def test_into_an_empty_file(self):
        assert merge("", block()) == block()

    def test_appends_without_touching_what_is_there(self):
        existing = "# My project\n\nAlways run the tests.\n"
        out = merge(existing, block())
        assert out.startswith("# My project")
        assert "Always run the tests." in out
        assert BEGIN in out

    def test_replaces_only_our_block_on_reinstall(self):
        first = merge("# Mine\n\nKeep me.\n", BEGIN + "\nOLD\n" + END + "\n")
        second = merge(first, block())
        assert "OLD" not in second
        assert "Keep me." in second and second.count(BEGIN) == 1

    def test_preserves_content_written_after_our_block(self):
        existing = merge("# Mine\n", block()) + "\n## Later section\n\nMine too.\n"
        out = merge(existing, block())
        assert "## Later section" in out and "Mine too." in out
        assert out.count(BEGIN) == 1

    def test_is_idempotent(self):
        once = merge("# Mine\n", block())
        assert merge(once, block()) == once


class TestInstall:
    def test_creates_nothing_by_default(self, tmp_path):
        # Running one command must not litter five files in someone's repo.
        results = install(tmp_path)
        assert all(w.startswith("skipped") for _, w in results)
        assert list(tmp_path.iterdir()) == []

    def test_updates_a_file_that_already_exists(self, tmp_path):
        claude = tmp_path / "CLAUDE.md"
        claude.write_text("# Rules\n\nBe careful.\n")
        install(tmp_path)
        text = claude.read_text()
        assert "Be careful." in text and BEGIN in text

    def test_named_host_creates_its_file(self, tmp_path):
        install(tmp_path, ["claude"])
        assert (tmp_path / "CLAUDE.md").exists()

    def test_creates_nested_paths_for_hosts_that_need_them(self, tmp_path):
        install(tmp_path, ["cursor"])
        assert (tmp_path / ".cursor/rules/graph-paat.mdc").exists()

    def test_reinstall_does_not_duplicate(self, tmp_path):
        install(tmp_path, ["claude"])
        install(tmp_path, ["claude"])
        assert (tmp_path / "CLAUDE.md").read_text().count(BEGIN) == 1

    def test_reports_already_current_on_a_second_run(self, tmp_path):
        install(tmp_path, ["claude"])
        assert install(tmp_path, ["claude"])[0][1] == "already current"

    def test_unknown_host_is_reported_not_crashed(self, tmp_path):
        assert install(tmp_path, ["nonsense"])[0][1] == "unknown host"

    def test_every_target_is_a_relative_path(self):
        # An absolute path here would write outside the repository.
        assert all(not t.startswith("/") for t in TARGETS.values())


class TestUninstall:
    def test_removes_our_block_and_keeps_the_rest(self, tmp_path):
        claude = tmp_path / "CLAUDE.md"
        claude.write_text("# Rules\n\nBe careful.\n")
        install(tmp_path)
        uninstall(tmp_path)
        text = claude.read_text()
        assert BEGIN not in text and END not in text
        assert "Be careful." in text and "# Rules" in text

    def test_leaves_untouched_files_alone(self, tmp_path):
        other = tmp_path / "AGENTS.md"
        other.write_text("# Not ours\n")
        uninstall(tmp_path)
        assert other.read_text() == "# Not ours\n"

    def test_round_trip_restores_the_original_exactly(self, tmp_path):
        # Not "close enough": removing our block must leave the file as it was.
        # Concatenating the two halves ran the following heading onto the
        # preceding paragraph, which changes how the markdown renders.
        claude = tmp_path / "CLAUDE.md"
        original = "# Rules\n\nOne.\n\n## Two\n\nMore.\n"
        claude.write_text(original)
        install(tmp_path)
        uninstall(tmp_path)
        assert claude.read_text() == original

    def test_round_trip_when_content_was_added_after_installing(self, tmp_path):
        claude = tmp_path / "CLAUDE.md"
        claude.write_text("# Rules\n\nOne.\n")
        install(tmp_path)
        claude.write_text(claude.read_text() + "\n## Later\n\nTwo.\n")
        uninstall(tmp_path)
        assert claude.read_text() == "# Rules\n\nOne.\n\n## Later\n\nTwo.\n"

    def test_a_file_containing_only_our_block_is_emptied(self, tmp_path):
        install(tmp_path, ["agents"])
        uninstall(tmp_path)
        assert (tmp_path / "AGENTS.md").read_text() == ""


class TestFollowingTheInstructionsLiterally:
    """The instructions are a contract. An agent will follow them exactly."""

    def test_every_command_named_is_a_real_command(self):
        from graphpaat.cli import USAGE
        import re
        named = set(re.findall(r"graph-paat (\w+)", INSTRUCTIONS))
        supported = set(re.findall(r"graph-paat (\w+)", USAGE))
        assert named <= supported, f"instructions promise {named - supported}"

    def test_every_flag_named_is_a_real_flag(self):
        from graphpaat.cli import USAGE
        import re
        named = set(re.findall(r"--[a-z-]+", INSTRUCTIONS))
        supported = set(re.findall(r"--[a-z-]+", USAGE))
        assert named <= supported, f"instructions promise {named - supported}"

    def test_querying_before_building_fails_with_a_message_not_a_traceback(self, tmp_path, monkeypatch):
        from graphpaat.cli import main
        monkeypatch.chdir(tmp_path)
        assert main(["query", "Session"]) == 1      # an exception would propagate
