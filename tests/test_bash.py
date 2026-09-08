"""Bash, read through tree-sitter.

Bash is the case that tests the vocabulary from the other end. Go and
TypeScript ask whether a rich language can be squeezed into five node kinds;
shell asks whether a language that fills only three of them still produces a
usable map. It has no classes, no methods and no types, so there is nothing
here to invent a `class` node out of -- and the tests below say so explicitly,
because "we emit no classes" is a decision, not an omission.

What shell has instead is one flat global function table and a call syntax that
cannot be told apart from running `ls`. So most of these tests are about what
the map REFUSES: an ordinary shell command, a name two files define, a command
whose name is a variable.
"""
import pytest

from graphpaat.parse import parse_corpus_files, parse_file
from graphpaat.resolve import resolve

pytest.importorskip("tree_sitter_bash")


@pytest.fixture
def shcorpus(tmp_path):
    def build(files: dict[str, str]):
        root = tmp_path / "repo"
        for name, source in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
        root.mkdir(parents=True, exist_ok=True)
        return root
    return build


def kinds(root):
    files, _ = parse_corpus_files(root)
    return {n.id: n.kind for p in files for n in p.nodes}


def links(root, relation="calls"):
    """Resolved edges of one relation.

    Containment comes from parsing and calls come from resolution, so both
    sources are needed -- looking only at the resolver's output made a correct
    containment edge look missing.
    """
    files, _ = parse_corpus_files(root)
    edges, _ = resolve(files)
    edges = list(edges) + [e for p in files for e in p.edges]
    return {(e.source, e.target) for e in edges
            if e.relation == relation and e.resolved}


def reasons_for(root):
    files, _ = parse_corpus_files(root)
    return resolve(files)[1]


def docs(root):
    files, _ = parse_corpus_files(root)
    return {n.id: n.text for p in files for n in p.nodes if n.kind == "rationale"}


LOG = {"lib/log.bash": '''#!/usr/bin/env bash

# Writes one line to the log file.
log_line() {
  printf '%s\\n' "$1" >> "$LOG_FILE"
}

# Writes a line and gives up.
function log_fatal {
  log_line "$1"
  exit 1
}
'''}


class TestSameShapesAsPython:
    def test_a_function_definition_is_a_function_node(self, shcorpus):
        assert kinds(shcorpus(LOG))["lib_log_log_line"] == "function"

    def test_the_function_keyword_form_is_also_a_function_node(self, shcorpus):
        # `function f { }` and `f() { }` are the same declaration written two
        # ways, and half of real shell uses each.
        assert kinds(shcorpus(LOG))["lib_log_log_fatal"] == "function"

    def test_no_class_nodes_are_invented(self, shcorpus):
        # Shell has no type to make one out of. A `class` node here would have
        # to come from a naming convention, which is a claim the source never
        # makes.
        assert "class" not in set(kinds(shcorpus(LOG)).values())

    def test_a_doc_comment_becomes_a_rationale_node(self, shcorpus):
        assert docs(shcorpus(LOG))["lib_log_log_line#doc"] == \
            "Writes one line to the log file."

    def test_a_run_of_comment_lines_is_one_doc(self, shcorpus):
        root = shcorpus({"m.sh": '''# Sorts the array.
# Intended for short lists.
sort_them() { :; }
'''})
        assert docs(root)["m_sort_them#doc"] == \
            "Sorts the array. Intended for short lists."

    def test_a_comment_separated_by_a_blank_line_is_not_documentation(self, shcorpus):
        root = shcorpus({"m.sh": '''# Unrelated note.

thing() { :; }
'''})
        assert not docs(root)

    def test_a_trailing_comment_belongs_to_the_line_it_sits_on(self, shcorpus):
        # `first() { :; }  # note` is a comment about `first`, on `first`'s own
        # line. Taking it as `second`'s documentation would attribute one
        # function's reason to another.
        root = shcorpus({"m.sh": '''first() { :; } # sets nothing
second() { :; }
'''})
        assert docs(root) == {}

    def test_a_shebang_is_not_documentation(self, shcorpus):
        root = shcorpus({"m.sh": '''#!/usr/bin/env bash
main() { :; }
'''})
        assert not docs(root)

    def test_a_shellcheck_directive_is_not_documentation(self, shcorpus):
        # Left in, the stated reason for this function is a linter setting --
        # a claim about the code that is not about the code.
        root = shcorpus({"m.sh": '''# shellcheck disable=SC2155
noop() { :; }
'''})
        assert not docs(root)

    def test_a_directive_under_a_sentence_keeps_the_sentence(self, shcorpus):
        root = shcorpus({"m.sh": '''# Extends the prompt.
# shellcheck disable=SC2329
extend() { :; }
'''})
        assert docs(root)["m_extend#doc"] == "Extends the prompt."

    def test_containment_matches_the_python_shape(self, shcorpus):
        got = links(shcorpus(LOG), "contains")
        assert ("lib_log", "lib_log_log_line") in got
        assert ("lib_log", "lib_log_log_fatal") in got

    def test_a_nested_function_is_qualified_by_the_one_it_sits_in(self, shcorpus):
        # Two helpers of the same name in one file are two pieces of code;
        # minting one id for both would lose one of them silently.
        root = shcorpus({"m.sh": '''outer() {
  helper() { :; }
}
other() {
  helper() { :; }
}
'''})
        k = kinds(root)
        assert k["m_outer_helper"] == "function"
        assert k["m_other_helper"] == "function"
        assert ("m_outer", "m_outer_helper") in links(root, "contains")

    def test_a_function_defined_in_a_branch_is_still_top_level(self, shcorpus):
        # An `if` is not a scope in shell: the definition lands in the same
        # global table as any other, so the branch must not enter its id.
        root = shcorpus({"m.sh": '''if [[ -n "$BASH" ]]; then
  portable_echo() { :; }
fi
'''})
        assert kinds(root)["m_portable_echo"] == "function"
        assert ("m", "m_portable_echo") in links(root, "contains")


class TestResolution:
    def test_a_call_in_the_same_file(self, shcorpus):
        assert ("lib_log_log_fatal", "lib_log_log_line") in links(shcorpus(LOG))

    def test_a_call_across_files_resolves_on_a_unique_name(self, shcorpus):
        # Shell has one global function table, so a name defined once in the
        # corpus is what the shell itself would run. This is the same rule the
        # resolver applies everywhere; it is simply stronger here.
        root = shcorpus({"lib/log.bash": "log_line() { :; }\n",
                         "bin/run.sh": "main() { log_line hello; }\n"})
        assert ("bin_run_main", "lib_log_log_line") in links(root)

    def test_a_call_at_the_top_level_is_owned_by_the_file(self, shcorpus):
        # A shell script does much of its work outside any function. Dropping
        # those calls would leave install.sh looking like it does nothing.
        root = shcorpus({"m.sh": '''setup() { :; }
setup
'''})
        assert ("m", "m_setup") in links(root)

    def test_a_name_two_files_define_is_refused_not_guessed(self, shcorpus):
        root = shcorpus({"a.sh": "noop() { :; }\n",
                         "b.sh": "noop() { :; }\n",
                         "c.sh": "run() { noop; }\n"})
        assert reasons_for(root)["name defined in 2 places, cannot choose"] >= 1
        assert ("c_run", "a_noop") not in links(root)

    def test_an_ordinary_shell_command_is_refused_not_guessed(self, shcorpus):
        # `grep` is not in the corpus and never will be. It is counted as a
        # refusal and left out of the map rather than pointed somewhere.
        root = shcorpus({"m.sh": "search() { grep -q x file; }\n"})
        assert reasons_for(root)[
            "not defined in this corpus (builtin or third party)"] >= 1

    def test_a_command_whose_name_is_a_variable_is_not_a_call(self, shcorpus):
        # `"$cmd" arg` names something only the running shell knows. Recording
        # it would invent a function called `$cmd`.
        root = shcorpus({"m.sh": '''run() {
  "$cmd" --flag
  ${runner} --flag
}
'''})
        files, _ = parse_corpus_files(root)
        assert [c.name for p in files for c in p.calls] == []

    def test_a_source_is_an_import_and_not_a_call(self, shcorpus):
        root = shcorpus({"m.sh": 'source lib/log.bash\n'})
        files, _ = parse_corpus_files(root)
        assert files[0].import_sites == [("lib.log", 1)]
        assert [c.name for c in files[0].calls] == []

    def test_the_dot_form_is_also_an_import(self, shcorpus):
        root = shcorpus({"m.sh": '. lib/log.bash\n'})
        files, _ = parse_corpus_files(root)
        assert files[0].import_sites == [("lib.log", 1)]

    def test_an_import_resolves_to_the_file_it_names(self, shcorpus):
        root = shcorpus({"lib/log.bash": "log_line() { :; }\n",
                         "run.sh": 'source lib/log.bash\n'})
        assert ("run", "lib_log") in links(root, "imports")

    def test_a_variable_standing_for_a_directory_keeps_the_literal_tail(self, shcorpus):
        # `"$BASH_IT/lib/log.bash"` is the normal way to write this: the root
        # is only known at run time, and the tail after it is a real path.
        root = shcorpus({"lib/log.bash": "log_line() { :; }\n",
                         "run.sh": 'source "$BASH_IT/lib/log.bash"\n'})
        assert ("run", "lib_log") in links(root, "imports")

    def test_a_variable_inside_a_filename_records_no_import(self, shcorpus):
        # The literal left over from `"$THEME/$THEME.theme.bash"` is
        # `.theme.bash`, a fragment of a name rather than a path. Emitting it
        # would name a file that does not exist.
        root = shcorpus({"run.sh": 'source "$THEME/$THEME.theme.bash"\n'})
        files, _ = parse_corpus_files(root)
        assert files[0].import_sites == []

    def test_a_wholly_dynamic_source_records_nothing(self, shcorpus):
        root = shcorpus({"run.sh": 'source "$1"\nsource <(gen bash)\n'})
        files, _ = parse_corpus_files(root)
        assert files[0].import_sites == []

    def test_an_import_that_leaves_the_corpus_is_drawn_unresolved(self, shcorpus):
        # Which system files a script pulls in is part of what it does, so
        # saying nothing would be a lie.
        root = shcorpus({"run.sh": 'source /etc/profile.d/x.sh\n'})
        assert reasons_for(root)["imports a module outside this corpus"] == 1

    def test_no_local_names_are_bound_by_a_source(self, shcorpus):
        # Sourcing creates no namespace -- the file's functions join the one
        # global table -- so an `imports` entry would be a guess about which
        # file defines which name.
        root = shcorpus({"run.sh": 'source lib/log.bash\n'})
        files, _ = parse_corpus_files(root)
        assert files[0].imports == {}

    def test_resolved_edges_never_dangle(self, shcorpus):
        files, _ = parse_corpus_files(shcorpus(LOG))
        ids = {n.id for p in files for n in p.nodes}
        edges, _ = resolve(files)
        for e in edges:
            if e.resolved:
                assert e.target in ids


class TestMixedCorpus:
    def test_python_and_bash_in_one_repository(self, shcorpus):
        root = shcorpus({"bin/deploy.sh": "deploy() { :; }\n",
                         "tools/build.py": "class Builder:\n    pass\n"})
        k = kinds(root)
        assert k["bin_deploy_deploy"] == "function"
        assert k["tools_build_builder"] == "class"


class TestFailure:
    def test_a_broken_file_does_not_take_the_corpus_down(self, shcorpus):
        root = shcorpus({"bad.sh": "\x00\x00 ((((", "ok.sh": "fine() { :; }\n"})
        files, failed = parse_corpus_files(root)
        assert "ok.sh" not in failed
        assert kinds(root)["ok_fine"] == "function"

    def test_a_very_long_command_chain_does_not_exhaust_the_stack(self, shcorpus):
        # Shell nests one level per `&&`, so a generated script joining a few
        # thousand commands makes a tree thousands deep. The parser handles it;
        # a recursive walk over it would not, and the file would be lost to the
        # tool for a reason that has nothing to do with the file.
        root = shcorpus({"deep.sh": " && ".join(["step"] * 3000) + "\n"})
        files, failed = parse_corpus_files(root)
        assert failed == []
        assert len(files[0].calls) == 3000

    def test_a_file_with_a_syntax_error_keeps_the_functions_around_it(self, shcorpus):
        # One unterminated block must not delete the rest of the file. A gap
        # the reader cannot see is worse than one they can.
        root = shcorpus({"m.sh": '''before() { :; }
broken() { if fi
after() { :; }
'''})
        k = kinds(root)
        assert k["m_before"] == "function"
        assert "m" in k
