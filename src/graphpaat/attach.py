"""Decide what a document is talking about -- without letting a model invent it.

The reader (`documents.py`) turns prose into claims. Something still has to say
which claim is about which symbol, and that is a judgement no parser makes: a
README saying *"the entry point for all output"* is about `Console`, and only
reading English gets you there.

**So a model does it, and the model is the one already running the tool.** This
holds no API key and calls no provider. The tool writes down what needs reading;
the assistant reads it and writes its answer back; the tool ingests the answer.
The vocabulary step works the same way, and so does graphify, which is why
neither needs a key.

**The model may not type an identifier.** It is handed a closed list of names
that already exist in the graph and must pick from it or pick nothing. This is
the single most important rule in this file. The two lanes join on exact id
string match, so an id that is *nearly* right does not produce a nearly-right
graph -- it produces a second node that nothing will ever reconcile with the
first. graphify mints duplicates in exactly this spot. A document about
something not in the graph attaches to nothing, and that is the correct answer,
not a failure.

**A claim never becomes a fact.** Everything minted here carries `origin="doc"`,
which survives into the store and into the rendered answer, so a sentence
someone wrote in 2019 is never read as something a parser verified today.

The candidate list is built from what a section literally mentions -- the
identifiers in backticks, the CamelCase and snake_case names in the prose.
Precision matters more than recall here: the list is a menu, and a menu full of
plausible wrong answers is how a model is talked into a wrong pick.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .documents import DOC_MARK, Reading, Section
from .ids import normalise_path_part
from .parse import Edge, Node

# How many names one section may choose from -- and, above it, the point at
# which a section stops being an explanation.
#
# Measured 2026-09-09 on sympy: the median section mentions 6 symbols already in
# the graph, but 47% mention more and the worst mentions 45. A passage naming 45
# symbols is an API index, and an index explains none of them -- the ask below
# already tells the model to answer `[]` when more than three fit, so putting
# those passages in front of it buys nothing and costs 140,000 tokens. They are
# skipped, and counted, rather than truncated into a menu of plausible wrong
# answers.
MAX_CANDIDATES = 12

# A name shorter than this matches too much prose to mean anything: `id`, `of`
# and `to` are all real symbol names somewhere.
MIN_NAME_CHARS = 4

# How much of a section the model is shown. Enough to judge what it is about;
# the whole thing would triple the cost of the ask for no extra signal.
ASK_SECTION_CHARS = 700

# Things that look like identifiers when they appear in prose.
_BACKTICKED = re.compile(r"`{1,3}([^`\n]{2,120})`{1,3}")
_IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\b")

ASK_FILE = "documents-to-read.md"
ANSWER_FILE = "document-answers.json"


@dataclass
class Ask:
    """What the model is being asked to read, and the key to check its answer."""
    text: str
    manifest: dict            # claim id -> {"candidates": [...], "path", "line"}
    digest: str
    sections: int
    tokens: int
    skipped_as_index: int = 0


def claim_id(doc_path: str, line: int) -> str:
    """The id of one claim.

    `#` cannot appear in an identifier, so no parsed symbol can ever mint one of
    these. That is the whole reason the marker is punctuation and not a suffix
    like `_doc`, which a function genuinely named `_doc` would collide with.
    """
    stem = "_".join(normalise_path_part(p) for p in PurePosixPath(doc_path).parts)
    return f"{stem}{DOC_MARK}#L{line}"


def document_id(doc_path: str) -> str:
    stem = "_".join(normalise_path_part(p) for p in PurePosixPath(doc_path).parts)
    return f"{stem}{DOC_MARK}"


def mentioned(text: str) -> set[str]:
    """Every name a section mentions, lowercased.

    Backticked spans first, because a project that writes `Console` in backticks
    is telling you it means the symbol. Bare identifiers in the prose count too:
    plenty of documentation writes CamelCase without any markup at all.
    """
    names: set[str] = set()
    for span in _BACKTICKED.findall(text):
        # `Console.print(x)` mentions Console and print, and `pip install x`
        # mentions nothing -- both fall out of splitting on non-identifier
        # characters and keeping what survives.
        for piece in _IDENTIFIER.findall(span):
            for part in piece.split("."):
                if len(part) >= MIN_NAME_CHARS:
                    names.add(part.lower())
    for piece in _IDENTIFIER.findall(text):
        # Bare prose is noisier than a code span, so only shapes that a person
        # would not write by accident count: CamelCase, or snake_case.
        if len(piece) < MIN_NAME_CHARS:
            continue
        for part in piece.split("."):
            if len(part) < MIN_NAME_CHARS:
                continue
            camel = any(c.isupper() for c in part[1:]) and not part.isupper()
            snake = "_" in part.strip("_")
            if camel or snake:
                names.add(part.lower())
    return names


def label_index(graph: dict) -> dict[str, list[str]]:
    """Lowercased name -> the node ids carrying it, best first.

    A file node's stem counts as one of its names, so a document saying
    "see console.py" can reach the file as well as the class inside it.
    """
    degree: dict[str, int] = defaultdict(int)
    for edge in graph["edges"]:
        if edge.get("resolved", True):
            degree[edge["source"]] += 1
            degree[edge["target"]] += 1

    index: dict[str, list[str]] = defaultdict(list)
    for node in graph["nodes"]:
        if node["kind"] in ("rationale", "claim", "document"):
            continue
        label = node["label"]
        keys = {label.lower()}
        if node["kind"] == "file":
            keys.add(PurePosixPath(label).stem.lower())
        for key in keys:
            if len(key) >= MIN_NAME_CHARS:
                index[key].append(node["id"])

    by_id = {n["id"]: n for n in graph["nodes"]}
    for key, ids in index.items():
        # Most connected first, then shortest id, then alphabetical -- fixed
        # order, so the same graph always produces the same menu.
        ids.sort(key=lambda i: (-degree.get(i, 0), len(i), i))
        index[key] = ids[:4]        # one name, at most four places it could be
    return {k: v for k, v in index.items() if v}


def candidates(section_text: str, index: dict[str, list[str]],
               limit: int = MAX_CANDIDATES) -> list[str]:
    """The closed list for one section. Deterministic, and possibly empty.

    Empty is a real answer, and it arrives two ways. A section mentioning
    nothing in the graph is prose about installation or licensing. A section
    mentioning *more than the limit* is an index or a reference table, which
    describes nothing in particular; both should stay away from the model.
    """
    picked: list[str] = []
    seen: set[str] = set()
    for name in sorted(mentioned(section_text)):
        for nid in index.get(name, []):
            if nid not in seen:
                seen.add(nid)
                picked.append(nid)
    return [] if len(picked) > limit else picked


def build_ask(reading: Reading, graph: dict) -> Ask:
    """The file the assistant is asked to read.

    Only sections that mention something already in the graph are included.
    That is not a saving so much as the closed-list rule applied one stage
    earlier: if nothing in a section is in the map, there is no correct pick,
    and asking anyway invites an incorrect one.
    """
    index = label_index(graph)
    by_id = {n["id"]: n for n in graph["nodes"]}

    manifest: dict[str, dict] = {}
    blocks: list[str] = []
    skipped_as_index = 0
    for doc in reading.documents:
        for section in doc.sections:
            options = candidates(section.text, index)
            if not options:
                if len(candidates(section.text, index, limit=10 ** 6)) > MAX_CANDIDATES:
                    skipped_as_index += 1
                continue
            cid = claim_id(doc.path, section.line)
            manifest[cid] = {"candidates": options, "path": doc.path,
                             "line": section.line, "title": section.title}
            body = " ".join(section.text.split())[:ASK_SECTION_CHARS]
            head = f"### {cid}"
            if section.title:
                head += f"  —  {section.title}"
            lines = [head, f"`{doc.path}:L{section.line}`", "", f"> {body}", "",
                     "Candidates:"]
            for nid in options:
                node = by_id[nid]
                lines.append(f"- `{nid}`  —  {node['label']}  "
                             f"({node['kind']}, {node['file']}:L{node['line']})")
            blocks.append("\n".join(lines))

    text = _ask_header(len(blocks)) + "\n\n" + "\n\n---\n\n".join(blocks) + "\n"
    digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    text = text.replace("__DIGEST__", digest)
    return Ask(text=text, manifest=manifest, digest=digest,
               sections=len(blocks), tokens=max(1, len(text) // 4),
               skipped_as_index=skipped_as_index)


def _ask_header(count: int) -> str:
    return f"""# Which symbol is each of these {count} passages about?

This file was written by `graph-paat build --deep`. Read it, then write your
answer to `{ANSWER_FILE}` beside it and run the same build command again.

## The rule

For each passage below, pick the symbols it is **actually describing** from that
passage's own `Candidates` list.

- **Copy an id from the list exactly. Never write one that is not there.**
  An id you invent is rejected, not created -- it cannot become a node.
- **Picking nothing is a correct and common answer.** Most prose is about
  installation, licensing or history and describes no symbol at all. Use `[]`.
- Pick the symbol the passage **explains**, not every symbol it happens to
  name. A passage listing ten functions in a table explains none of them.
- At most three picks per passage. If more than three fit, the passage is about
  a topic rather than about a symbol, and the answer is `[]`.

## The format

```json
{{
  "ask": "__DIGEST__",
  "picks": {{
    "readme#document#L12": ["console_Console"],
    "readme#document#L40": []
  }}
}}
```

`ask` must be copied exactly as it appears above; it says which version of this
file you answered, so a stale answer is caught rather than silently applied.
Every passage should appear in `picks`, with `[]` where nothing fits.
"""


@dataclass
class Ingested:
    nodes: list[Node]
    edges: list[Edge]
    attached: int = 0
    empty: int = 0
    rejected: dict[str, int] | None = None
    unanswered: int = 0


def ingest(answer: dict, manifest: dict, reading: Reading) -> Ingested:
    """Turn the model's answer into nodes and edges, refusing anything invented.

    Three normalisations, and each exists so that two runs of the same build
    produce byte-identical graphs even though the model does not produce
    byte-identical answers:

    **Order is discarded.** Picks are sorted, so `["a", "b"]` and `["b", "a"]`
    are the same answer.

    **Repeats are discarded.** A model that names one symbol twice attached it
    once.

    **Anything outside the closed list is discarded**, counted, and reported --
    never created. This is D18, and it is the whole point of the stage.
    """
    picks = answer.get("picks") or {}
    rejected: dict[str, int] = defaultdict(int)
    nodes: list[Node] = []
    edges: list[Edge] = []
    attached = empty = 0

    text_of: dict[str, tuple[Section, str]] = {}
    for doc in reading.documents:
        for section in doc.sections:
            text_of[claim_id(doc.path, section.line)] = (section, doc.path)

    seen_documents: set[str] = set()
    for cid in sorted(manifest):
        entry = manifest[cid]
        allowed = set(entry["candidates"])
        raw = picks.get(cid)
        if raw is None:
            rejected["passage not answered"] += 1
            continue
        if not isinstance(raw, list):
            rejected["answer was not a list"] += 1
            continue

        chosen: list[str] = []
        for item in raw:
            if not isinstance(item, str):
                rejected["pick was not an id"] += 1
                continue
            pick = item.strip()
            if pick not in allowed:
                # The one refusal that matters. An id the model typed rather
                # than picked would become a node nothing else can ever reach.
                rejected["id not on the candidate list"] += 1
                continue
            if pick not in chosen:
                chosen.append(pick)
        chosen = sorted(chosen)[:3]
        if not chosen:
            empty += 1
            continue

        section, doc_path = text_of[cid]
        did = document_id(doc_path)
        if did not in seen_documents:
            seen_documents.add(did)
            nodes.append(Node(id=did, label=PurePosixPath(doc_path).name,
                              kind="document", file=doc_path, line=1,
                              origin="doc"))
        nodes.append(Node(
            id=cid, label=section.title or PurePosixPath(doc_path).name,
            kind="claim", file=doc_path, line=section.line, origin="doc",
            text=" ".join(section.text.split()),
        ))
        edges.append(Edge(source=did, target=cid, relation="contains",
                          file=doc_path, line=section.line, origin="doc"))
        for pick in chosen:
            edges.append(Edge(source=cid, target=pick, relation="describes",
                              file=doc_path, line=section.line, origin="doc"))
        attached += 1

    return Ingested(nodes=nodes, edges=edges, attached=attached, empty=empty,
                    rejected=dict(rejected),
                    unanswered=rejected.get("passage not answered", 0))


def load_answer(path: Path, digest: str) -> tuple[dict, str]:
    """Read the answer file, refusing one written against a different ask.

    Content-addressing rather than invalidation, which is the one idea worth
    taking from graphify's cache: a stale answer is unrepresentable instead of
    being something a later stage has to notice.
    """
    if not path.exists():
        return {}, f"no answer yet at {path}"
    try:
        answer = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, f"the answer at {path} is not valid JSON ({exc.msg} at line {exc.lineno})"
    if not isinstance(answer, dict):
        return {}, f"the answer at {path} should be a JSON object"
    given = str(answer.get("ask", ""))
    if given != digest:
        return {}, (f"the answer at {path} was written for a different set of "
                    f"documents (says '{given}', this build asks '{digest}') - "
                    f"re-read {ASK_FILE} and answer it again")
    return answer, ""
