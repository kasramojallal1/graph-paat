"""Parsing: which definitions become nodes, and what holds what."""
from graphpaat.parse import DOC_SUFFIX, collect, parse_corpus_files, parse_file


def ids(nodes):
    return {n.id for n in nodes}


def kinds(nodes):
    return {n.id: n.kind for n in nodes}


class TestCollect:
    def test_finds_python_files(self, corpus):
        root = corpus({"a.py": "", "pkg/b.py": "", "notes.md": ""})
        assert {p.name for p in collect(root)} == {"a.py", "b.py"}

    def test_skips_noise_directories(self, corpus):
        root = corpus({"a.py": "", ".venv/lib/x.py": "", "__pycache__/y.py": "",
                       "node_modules/z.py": "", ".git/w.py": ""})
        assert {p.name for p in collect(root)} == {"a.py"}


class TestNodes:
    def test_file_class_function_method(self, corpus):
        root = corpus({"m.py": (
            "class Widget:\n"
            "    def render(self):\n"
            "        pass\n"
            "def helper():\n"
            "    pass\n")})
        parsed = parse_file(root / "m.py", root)
        k = kinds(parsed.nodes)
        assert k["m"] == "file"
        assert k["m_widget"] == "class"
        assert k["m_widget_render"] == "method"
        assert k["m_helper"] == "function"

    def test_nested_function_is_a_function_not_a_method(self, corpus):
        # A def inside a method is a nested function. Getting this wrong made
        # every inner helper look like a method of the surrounding class.
        root = corpus({"m.py": (
            "class W:\n"
            "    def outer(self):\n"
            "        def inner():\n"
            "            pass\n")})
        parsed = parse_file(root / "m.py", root)
        k = kinds(parsed.nodes)
        assert k["m_w_outer"] == "method"
        assert k["m_w_outer_inner"] == "function"

    def test_property_setter_is_not_a_separate_node(self, corpus):
        # Callers write obj.choices for both halves, so the pair is one thing.
        # Emitting both was 100 false collision reports on Django.
        root = corpus({"m.py": (
            "class W:\n"
            "    @property\n"
            "    def choices(self):\n"
            "        return 1\n"
            "    @choices.setter\n"
            "    def choices(self, v):\n"
            "        pass\n")})
        nodes, _, collisions, _ = _run(root)
        assert collisions.collided() == {}
        assert "m_w_choices" in ids(nodes)

    def test_two_same_named_functions_collide_and_are_reported(self, corpus):
        # The case graphify is silent about, and the reason we report at all.
        root = corpus({"m.py": "def f():\n    pass\ndef f():\n    pass\n"})
        _, _, collisions, _ = _run(root)
        assert "m_f" in collisions.collided()

    def test_unparseable_file_is_reported_not_skipped(self, corpus):
        root = corpus({"broken.py": "def (:\n", "fine.py": "def ok():\n    pass\n"})
        _, _, _, failed = _run(root)
        assert failed == ["broken.py"]


class TestDocstrings:
    def test_module_and_function_docstrings_become_nodes(self, corpus):
        root = corpus({"m.py": '"""Module doc."""\ndef f():\n    """Fn doc."""\n'})
        parsed = parse_file(root / "m.py", root)
        by_id = {n.id: n for n in parsed.nodes}
        assert by_id[f"m{DOC_SUFFIX}"].text == "Module doc."
        assert by_id[f"m_f{DOC_SUFFIX}"].text == "Fn doc."
        assert by_id[f"m_f{DOC_SUFFIX}"].kind == "rationale"

    def test_doc_suffix_cannot_clash_with_a_real_symbol(self, corpus):
        # Shipped bug: the suffix used to be "__doc", so a module docstring and
        # a real function named _doc minted the same id.
        root = corpus({"m.py": '"""Doc."""\ndef _doc():\n    pass\n'})
        _, _, collisions, _ = _run(root)
        assert collisions.collided() == {}

    def test_docstring_edge_is_rationale_for_not_contains(self, corpus):
        root = corpus({"m.py": "def f():\n    '''Why.'''\n"})
        parsed = parse_file(root / "m.py", root)
        rel = {(e.source, e.target): e.relation for e in parsed.edges}
        assert rel[("m_f", f"m_f{DOC_SUFFIX}")] == "rationale_for"


class TestContainment:
    def test_file_contains_class_contains_method(self, corpus):
        root = corpus({"m.py": "class W:\n    def r(self):\n        pass\n"})
        parsed = parse_file(root / "m.py", root)
        edges = {(e.source, e.target) for e in parsed.edges if e.relation == "contains"}
        assert ("m", "m_w") in edges
        assert ("m_w", "m_w_r") in edges

    def test_every_non_file_node_has_exactly_one_container(self, corpus):
        root = corpus({"m.py": "class W:\n    def r(self):\n        def i():\n            pass\n"})
        nodes, edges, _, _ = _run(root)
        contained = [e.target for e in edges if e.relation in ("contains", "rationale_for")]
        assert len(contained) == len(set(contained))
        assert len(contained) == len(nodes) - 1        # every node but the file


class TestEvidence:
    """What the parser collects for the resolver to use later."""

    def test_imports_recorded_both_forms(self, corpus):
        root = corpus({"m.py": "import os\nfrom pkg.mod import thing\n"})
        parsed = parse_file(root / "m.py", root)
        assert parsed.imports["os"] == "os"
        assert parsed.imports["thing"] == "pkg.mod.thing"

    def test_variable_typed_by_assignment(self, corpus):
        root = corpus({"m.py": "class Db:\n    pass\ndef f():\n    d = Db()\n"})
        parsed = parse_file(root / "m.py", root)
        assert any(v == "Db" for v in parsed.var_types.values())

    def test_parameter_typed_by_annotation(self, corpus):
        root = corpus({"m.py": "def f(gw: Gateway):\n    pass\n"})
        parsed = parse_file(root / "m.py", root)
        assert any(v == "Gateway" for v in parsed.var_types.values())

    def test_self_attribute_typed_in_init(self, corpus):
        root = corpus({"m.py": (
            "class Db:\n    pass\n"
            "class S:\n"
            "    def __init__(self):\n"
            "        self.db = Db()\n")})
        parsed = parse_file(root / "m.py", root)
        assert parsed.attr_types["S::db"] == "Db"

    def test_call_sites_distinguish_their_shapes(self, corpus):
        root = corpus({"m.py": (
            "class S:\n"
            "    def go(self):\n"
            "        plain()\n"
            "        self.other()\n"
            "        self.db.save()\n"
            "        thing.run()\n")})
        parsed = parse_file(root / "m.py", root)
        by_name = {c.name: c for c in parsed.calls}
        assert by_name["plain"].receiver is None
        assert by_name["other"].on_self
        assert by_name["save"].receiver_is_self and by_name["save"].receiver == "db"
        assert by_name["run"].receiver == "thing" and not by_name["run"].on_self


def _run(root):
    """parse the corpus the way build does, returning nodes/edges/collisions/failed."""
    from graphpaat.ids import Collisions
    files, failed = parse_corpus_files(root)
    nodes, edges, collisions = [], [], Collisions()
    for parsed in files:
        for node in parsed.nodes:
            collisions.claim(node.id, f"{node.file}:L{node.line}")
            nodes.append(node)
        edges.extend(parsed.edges)
    return nodes, edges, collisions, failed


class TestAFileThatCannotBeWalked:
    """One unreadable file must not take a whole repository with it."""

    def test_a_deeply_nested_expression_is_reported_not_fatal(self, corpus, tmp_path):
        # sympy ships a generated lookup table whose expressions nest deeply
        # enough to exhaust the interpreter stack during the walk. It killed a
        # 1,532-file build outright.
        deep = "x = " + "(" * 400 + "1" + ")" * 400 + "\n"
        root = corpus({"generated.py": deep, "real.py": "def works():\n    pass\n"})
        files, failed = parse_corpus_files(root)

        assert "generated.py" in failed, "the gap has to be visible"
        labels = {n.label for parsed in files for n in parsed.nodes}
        assert "works" in labels, "every other file still parses"
