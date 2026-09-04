"""The artifact: what gets written, and what happens when it is wrong."""
import json

import pytest

from graphpaat import store
from graphpaat.ids import Collisions
from graphpaat.parse import parse_corpus_files


def build(root, out):
    files, failed = parse_corpus_files(root)
    nodes, edges, collisions = [], [], Collisions()
    for parsed in files:
        for node in parsed.nodes:
            collisions.claim(node.id, f"{node.file}:L{node.line}")
            nodes.append(node)
        edges.extend(parsed.edges)
    return store.write(root, nodes, edges, collisions, failed, out=out), nodes, edges


class TestRoundTrip:
    def test_written_graph_reads_back_identically(self, corpus, tmp_path):
        root = corpus({"m.py": "class W:\n    def r(self):\n        pass\n"})
        out = tmp_path / "out"
        _, nodes, edges = build(root, out)
        data = store.read(root, out=out)
        assert len(data["nodes"]) == len(nodes)
        assert len(data["edges"]) == len(edges)

    def test_missing_graph_says_how_to_make_one(self, corpus, tmp_path):
        root = corpus({"m.py": ""})
        with pytest.raises(FileNotFoundError, match="build"):
            store.read(root, out=tmp_path / "nothing")


class TestOutputLocation:
    def test_never_written_inside_the_corpus(self, corpus, tmp_path):
        # The output is an artifact. Writing it into the tree being analysed
        # pollutes a repo the user may not own -- we did this on the first run.
        root = corpus({"m.py": ""})
        path, _, _ = build(root, tmp_path / "out")
        assert root not in path.parents

    def test_default_location_is_the_working_directory(self, corpus, tmp_path, monkeypatch):
        """With no --out, the graph lands in the cwd, never in the corpus."""
        root = corpus({"m.py": ""})
        elsewhere = tmp_path / "cwd"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        assert store.out_path(root).parent.parent == elsewhere.resolve()
        assert root not in store.out_path(root).parents


class TestCoverage:
    def test_records_what_failed_to_parse(self, corpus, tmp_path):
        root = corpus({"ok.py": "def f():\n    pass\n", "bad.py": "def (:\n"})
        out = tmp_path / "out"
        build(root, out)
        assert store.read(root, out=out)["coverage"]["files_failed"] == ["bad.py"]

    def test_records_collisions_with_both_locations(self, corpus, tmp_path):
        root = corpus({"m.py": "def f():\n    pass\ndef f():\n    pass\n"})
        out = tmp_path / "out"
        build(root, out)
        collisions = store.read(root, out=out)["coverage"]["collisions"]
        assert len(collisions["m_f"]) == 2

    def test_records_id_counts_so_loss_is_visible(self, corpus, tmp_path):
        root = corpus({"m.py": "def f():\n    pass\ndef f():\n    pass\n"})
        out = tmp_path / "out"
        build(root, out)
        cov = store.read(root, out=out)["coverage"]
        assert cov["ids_unique"] < cov["nodes_total"]


class TestFailureModes:
    def test_a_stale_schema_is_refused_not_misread(self, corpus, tmp_path):
        root = corpus({"m.py": ""})
        out = tmp_path / "out"
        path, _, _ = build(root, out)
        data = json.loads(path.read_text())
        data["schema"] = store.SCHEMA + 1
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="rebuild"):
            store.read(root, out=out)

    def test_a_truncated_file_says_what_to_do_about_it(self, corpus, tmp_path):
        # "Expecting value: line 1 column 1" is not actionable.
        root = corpus({"m.py": "def f():\n    pass\n"})
        out = tmp_path / "out"
        path, _, _ = build(root, out)
        path.write_text(path.read_text()[: len(path.read_text()) // 2])
        with pytest.raises(ValueError, match="rebuild it"):
            store.read(root, out=out)
