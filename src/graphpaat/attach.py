"""Decide what a document is talking about -- without letting a model invent it.

The reader (`documents.py`) turns prose into claims. Something still has to say
which claim is about which symbol, and that is a judgement no parser makes: a
README saying *"the entry point for all output"* is about `Console`, and only
reading English gets you there.

**So a model does it, and the model is the one already running the tool.** This
holds no API key and calls no provider. The tool writes down what needs reading;
the assistant reads it and writes its answer back; the tool ingests the answer.
The vocabulary step works the same way, which is why neither needs a key.

**The model may not type an identifier.** It is handed a closed list of names
that already exist in the graph and must pick from it or pick nothing. This is
the single most important rule in this file. The two lanes join on exact id
string match, so an id that is *nearly* right does not produce a nearly-right
graph -- it produces a second node that nothing will ever reconcile with the
first. A document about something not in the graph attaches to nothing, and
that is the correct answer, not a failure.

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

# How many names one section may choose from.
MAX_CANDIDATES = 12

# Above this many distinct real symbols, a passage is an API index or a
# reference table, and an index explains none of what it lists. Those never
# reach the model: the ask already tells it to answer `[]` when more than three
# symbols fit, so showing them costs tokens to be told nothing.
#
# **Why this is not simply MAX_CANDIDATES.** It was, and one threshold behaved
# completely differently on two real repositories: at 12 it dropped 8% of one
# project's passages and 44% of sympy's, because sympy documents mathematics
# and names far more symbols per page. Dropping 44% would have thrown away real
# explanations. So the menu is *ranked and truncated* in the ordinary case, and
# only a genuine index is dropped.
INDEX_MENTIONS = 30

# A name shorter than this matches too much prose to mean anything: `id`, `of`
# and `to` are all real symbol names somewhere.
MIN_NAME_CHARS = 4

# How much of a section the model is shown. Enough to judge what it is about;
# the whole thing would triple the cost of the ask for no extra signal.
ASK_SECTION_CHARS = 700

# Things that look like identifiers when they appear in prose.
#
# Three separate readers, because a project writes a symbol name three ways and
# missing any one of them loses the passages that matter most. Found by hand,
# not by any test here: one project's ARCHITECTURE.md describes its whole
# pipeline as `detect() -> extract() -> build() -> cluster()` inside a fenced
# block, and NONE of those four names reached the candidate list. A single-line backtick rule cannot see inside a fence, and a
# bare lowercase word is not CamelCase or snake_case so the prose rule dropped
# it too. The passage most about the codebase was the one we could not read.
_FENCED = re.compile(r"```[^\n]*\n(.*?)```", re.S)
_BACKTICKED = re.compile(r"`([^`\n]{2,120})`")
_IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\b")
# `detect()` in running prose is a function reference and nothing else. The
# parentheses are what make a bare lowercase word safe to take.
_CALLED = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")

# How many tokens of reading the model is asked to do, per repository.
#
# The prose budget caps what is *read off disk*; this caps what is *put to the
# model*, and they are different numbers because a passage costs its own text
# plus a menu of candidates. Measured: sympy's uncapped ask is
# 363,000 tokens, which is more than most agents will spend on a whole session.
#
# Passages are spent strongest-first -- the ones whose subject is named in the
# heading or the running prose, before the ones that only appear in a code
# example -- and whatever the cap excluded is printed. A silent cap reads as
# full coverage.
DEFAULT_ASK_TOKENS = 120_000

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
    skipped_over_budget: int = 0
    tokens_over_budget: int = 0


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


def mentioned(text: str, title: str = "") -> dict[str, int]:
    """Every name a passage mentions, lowercased, and how strongly.

    The strength is what makes the menu worth reading. A symbol named in the
    passage's own heading is almost certainly its subject; one written as a call
    or in backticks in the running prose is probably relevant; one that appears
    only inside a code example is often just a step in a recipe. Ranking by that
    and keeping the top handful beats truncating an unordered list, which is how
    a model gets handed a menu of plausible wrong answers.

    **What is deliberately not counted: a bare English word in the prose.** An
    earlier version took every four-letter identifier it found outside a code
    block, which made "documentation", "provides" and "changes" into mentions --
    and every passage in both test repositories then had a strong mention of
    something. Only shapes an author cannot write by accident count: marked as
    code, written as a call, or spelled in CamelCase or snake_case.
    """
    names: dict[str, int] = {}

    def note(word: str, weight: int) -> None:
        if len(word) >= MIN_NAME_CHARS:
            key = word.lower()
            names[key] = max(names.get(key, 0), weight)

    def every_identifier(span: str, weight: int) -> None:
        """Inside code, every identifier is an identifier."""
        for piece in _IDENTIFIER.findall(span):
            for part in piece.split("."):
                note(part, weight)

    def code_shaped(span: str, weight: int) -> None:
        """In prose, only what an author could not have typed by accident."""
        for piece in _CALLED.findall(span):
            note(piece, weight)
        for piece in _IDENTIFIER.findall(span):
            for part in piece.split("."):
                if len(part) < MIN_NAME_CHARS:
                    continue
                # A capital letter or an underscore. `Parser` and `QuerySet`
                # are class names wherever they appear; a lowercase word in
                # running prose is just a word, and `build` or `report` would
                # otherwise match a symbol on every page that used them.
                capital = part[0].isupper() and not part.isupper()
                camel = any(c.isupper() for c in part[1:]) and not part.isupper()
                snake = "_" in part.strip("_")
                if capital or camel or snake:
                    note(part, weight)

    prose = _FENCED.sub(" ", text)

    # The heading is the strongest statement a passage makes about its subject,
    # and it is short and deliberate -- so every word in it counts, including a
    # bare lowercase one. A section titled `detect` is about detect.
    every_identifier(title, 4)

    # Running prose: backticks are the author saying "I mean the symbol".
    for span in _BACKTICKED.findall(prose):
        every_identifier(span, 3)
    code_shaped(prose, 3)

    # A fenced block is marked as code, but it is an example -- the names in it
    # are being *used*, which is weaker evidence than being talked about.
    for span in _FENCED.findall(text):
        every_identifier(span, 1)
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
               limit: int = MAX_CANDIDATES, title: str = "") -> list[str]:
    """The closed list for one section. Deterministic, and possibly empty.

    Empty is a real answer, and it arrives two ways. A section mentioning
    nothing in the graph is prose about installation or licensing. A section
    mentioning more real symbols than `INDEX_MENTIONS` is a reference table,
    which describes nothing in particular. Both should stay away from the model.

    In between, the list is ranked by how strongly the passage names each symbol
    and cut to `limit` -- heading first, then running prose, then code examples.
    """
    scored = mentioned(section_text, title)
    real = {name: weight for name, weight in scored.items() if name in index}
    if len(real) > INDEX_MENTIONS:
        return []
    picked: list[str] = []
    seen: set[str] = set()
    # Strongest mention first, then alphabetically -- a fixed order, so the same
    # passage always produces the same menu.
    for name in sorted(real, key=lambda n: (-real[n], n)):
        for nid in index.get(name, []):
            if nid not in seen:
                seen.add(nid)
                picked.append(nid)
    return picked[:limit]


def build_ask(reading: Reading, graph: dict,
              budget_tokens: int = DEFAULT_ASK_TOKENS) -> Ask:
    """The file the assistant is asked to read.

    Only sections that mention something already in the graph are included.
    That is not a saving so much as the closed-list rule applied one stage
    earlier: if nothing in a section is in the map, there is no correct pick,
    and asking anyway invites an incorrect one.
    """
    index = label_index(graph)
    by_id = {n["id"]: n for n in graph["nodes"]}

    # Everything worth asking, carrying the strength of its best mention -- so
    # the budget below buys the passages most likely to be describing something
    # rather than whichever ones happen to come first in the tree.
    ready: list[tuple[int, int, str, dict]] = []
    skipped_as_index = 0
    pairs = [(d, s) for d in reading.documents for s in d.sections]
    for order, (doc, section) in enumerate(pairs):
        scored = mentioned(section.text, section.title)
        real = {n: w for n, w in scored.items() if n in index}
        options = candidates(section.text, index, title=section.title)
        if not options:
            if len(real) > INDEX_MENTIONS:
                skipped_as_index += 1
            continue
        ready.append((-max(real.values()), order,
                      claim_id(doc.path, section.line),
                      {"candidates": options, "path": doc.path,
                       "line": section.line, "title": section.title,
                       "text": section.text}))
    ready.sort(key=lambda row: (row[0], row[1]))

    manifest: dict[str, dict] = {}
    kept: list[tuple[int, str]] = []
    spent = skipped_over_budget = tokens_over_budget = 0
    for _strength, order, cid, entry in ready:
        body = " ".join(entry["text"].split())[:ASK_SECTION_CHARS]
        head = f"### {cid}"
        if entry["title"]:
            head += f"  —  {entry['title']}"
        lines = [head, f"`{entry['path']}:L{entry['line']}`", "", f"> {body}", "",
                 "Candidates:"]
        for nid in entry["candidates"]:
            node = by_id[nid]
            lines.append(f"- `{nid}`  —  {node['label']}  "
                         f"({node['kind']}, {node['file']}:L{node['line']})")
        block = "\n".join(lines)
        cost = max(1, len(block) // 4)
        if spent + cost > budget_tokens:
            skipped_over_budget += 1
            tokens_over_budget += cost
            continue
        spent += cost
        manifest[cid] = {k: v for k, v in entry.items() if k != "text"}
        kept.append((order, block))

    # Back into document order, so the file reads like the repository does.
    kept.sort()
    blocks = [block for _order, block in kept]

    text = _ask_header(len(blocks)) + "\n\n" + "\n\n---\n\n".join(blocks) + "\n"
    digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    text = text.replace("__DIGEST__", digest)
    return Ask(text=text, manifest=manifest, digest=digest,
               sections=len(blocks), tokens=max(1, len(text) // 4),
               skipped_as_index=skipped_as_index,
               skipped_over_budget=skipped_over_budget,
               tokens_over_budget=tokens_over_budget)


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
    never created. That is the whole point of the stage.
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

    Content-addressing rather than invalidation: a stale answer is
    unrepresentable instead of being something a later stage has to notice.
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
