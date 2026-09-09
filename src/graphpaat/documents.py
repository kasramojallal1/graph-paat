"""Read what a repository writes about itself.

The code lane parses definitions; this one reads the prose beside them. A
README, a design note under `doc/`, a changelog entry -- the places where a
project explains itself in English rather than in identifiers.

Three things here, and the order matters.

**Finding the documents.** Nearest the root first. A README is the single most
informative file in most repositories and a translated tutorial fourteen levels
down is the least, so the walk is ordered rather than alphabetical, and the
budget below spends itself from the top.

**A budget, announced.** Prose is large: 363 markdown files in one real
repository come to roughly a million tokens, which is more than most agents can
hold and far more than anyone wants to pay for. So a cap, and -- the part that
matters -- **an explicit list of what the cap excluded.** A silent cap reads as
full coverage, and the whole stance of this tool is that invisible loss is the
enemy. It is a product decision as much as a budget one: someone with a
five-thousand-file documentation tree has exactly this problem.

**Sections, not files.** A 4,000-line manual is not one statement about the
codebase, it is two hundred. Splitting on headings gives each claim its own
line number, which is what lets an answer say `README.md:L12` instead of
`README.md`.

Nothing here calls a model. Nothing here decides what a document is *about* --
that is `attach.py`, and it is a separate stage on purpose.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .parse import SKIP_DIRS

# What counts as a document. `.markdown` is here because it is common enough to
# be worth two words; images are deliberately absent (D20).
TEXT_EXTENSIONS = {".md", ".markdown", ".rst", ".txt"}
PDF_EXTENSIONS = {".pdf"}
DOC_EXTENSIONS = TEXT_EXTENSIONS | PDF_EXTENSIONS

# The id namespace for everything this file produces. `#` cannot appear in an
# identifier, so no code symbol can ever mint one of these -- the same trick
# `parse.DOC_SUFFIX` uses, and for the same reason: the two lanes join on exact
# id match, so a namespace that can collide is a duplicate node waiting to
# happen.
DOC_MARK = "#document"

# How much prose the model is asked to read, per repository. Measured
# 2026-09-09: graphify's 363 markdown files are ~980k tokens, so an uncapped
# lane costs more than the rest of the session put together.
DEFAULT_BUDGET_TOKENS = 200_000

# A section longer than this is split further at paragraph boundaries. Long
# enough to hold a real explanation, short enough that a claim points at
# something specific.
MAX_SECTION_CHARS = 1_600
MIN_SECTION_CHARS = 40

CHARS_PER_TOKEN = 4

# Directories that are documentation *about the tooling* rather than about the
# code. Kept short for the same reason `SKIP_DIRS` is: a long list here quietly
# shrinks the corpus.
SKIP_DOC_DIRS = {"node_modules", "_build", "site-packages", ".git"}

_ATX = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_RST_UNDERLINE = re.compile(r"^([=\-~^\"'`*+#_:.]{2,})\s*$")

# What a document says in a form that survives translation: the identifiers in
# backticks, the fenced commands, the URLs. Prose changes language; `pip install
# sympy` does not.
_FINGERPRINT = re.compile(r"`{1,3}([^`\n]{2,80})`{1,3}|(https?://\S{4,})")

# How much of that fingerprint two documents must share to count as the same
# document twice. High on purpose: two pages about the same module legitimately
# share commands, and dropping a real page costs more than keeping a copy.
DUPLICATE_SHARE = 0.9
DUPLICATE_FLOOR = 12        # below this, a fingerprint is too small to trust


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass
class Section:
    """One claim-sized piece of a document, with the line it starts on."""
    title: str
    text: str
    line: int


@dataclass
class Document:
    path: str                 # relative to the corpus root
    sections: list[Section] = field(default_factory=list)
    tokens: int = 0


@dataclass
class Reading:
    """Everything the document pass produced, including what it refused to do."""
    documents: list[Document] = field(default_factory=list)
    skipped: list[tuple[str, int, str]] = field(default_factory=list)  # path, tokens, why
    unreadable: list[tuple[str, str]] = field(default_factory=list)  # (path, why)
    tokens_read: int = 0

    @property
    def tokens_skipped(self) -> int:
        return sum(t for _, t, _why in self.skipped)


def fingerprint(text: str) -> set[str]:
    """The parts of a document that survive being rewritten in another language.

    Measured 2026-09-09 on graphify: 31 translated copies of one README were
    taking 77,000 of the 200,000-token budget -- 39% of everything the model
    would be asked to read, spent on the same page in Arabic, Chinese, Korean
    and twenty-eight others. Ten near-identical `skill-*.md` files, one per
    host, took most of the rest.

    Prose translates; `graphify update` does not. So a document is fingerprinted
    by its backticked identifiers, its fenced commands and its URLs, and a later
    document sharing nearly all of one already read is a copy.
    """
    return {(a or b).strip().lower()
            for a, b in _FINGERPRINT.findall(text) if (a or b).strip()}


def _order_key(rel: Path) -> tuple:
    """Nearest the root first: README, then `doc/`, then breadth-first outward.

    The budget is spent from the top of this order, so it decides which prose a
    user actually gets when a repository has more than the cap allows.

    **History files sort last, and this is worth 99,484 tokens.** Measured
    2026-09-09: graphify's `CHANGELOG.md` is half the entire prose budget on its
    own, and a changelog describes what *changed*, not what anything *is* -- the
    weakest prose per token in a repository, and usually the largest file. It is
    deprioritised rather than excluded, so a repository with room to spare still
    reads it.
    """
    parts = rel.parts
    depth = len(parts) - 1
    at_root = depth == 0
    stem = rel.stem.lower()
    is_readme = stem == "readme"
    is_history = stem in ("changelog", "changes", "news", "history", "releases",
                          "release-notes", "release_notes", "whatsnew")
    top = parts[0].lower() if depth else ""
    in_docs = top in ("doc", "docs", "documentation")
    return (1 if is_history else 0,
            0 if (at_root and is_readme) else 1,
            0 if in_docs else 1,
            depth,
            str(rel).lower())


def collect(root: Path) -> list[Path]:
    """Every document under root, most informative first."""
    found = []
    for path in root.rglob("*"):
        if path.suffix.lower() not in DOC_EXTENSIONS or not path.is_file():
            continue
        rel = path.relative_to(root)
        parts = rel.parts[:-1]
        if any(d in SKIP_DIRS or d in SKIP_DOC_DIRS or d.startswith(".")
               for d in parts):
            continue
        found.append(rel)
    return [root / rel for rel in sorted(found, key=_order_key)]


def read_pdf(path: Path) -> tuple[str, str]:
    """Text out of a PDF, or a reason we could not get it.

    An optional extra, exactly like the tree-sitter grammars: a repository with
    no PDFs should not have to install a PDF library, and a missing library is
    reported rather than silently dropping the file.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        return "", ("no PDF support: pip install 'graph-paat[pdf]'")
    try:
        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "") for page in reader.pages]
    except Exception as exc:            # pragma: no cover - depends on the file
        # A malformed or encrypted PDF must not take the build down. It is a
        # gap, and a gap gets reported.
        return "", f"unreadable PDF ({type(exc).__name__})"
    text = "\n\n".join(p.strip() for p in pages if p.strip())
    if not text.strip():
        # A scanned PDF is images of text. We do not do images (D20), and
        # saying so is better than emitting an empty document.
        return "", "no extractable text (scanned or image-only)"
    return text, ""


def read_text(path: Path) -> tuple[str, str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace"), ""
    except OSError as exc:
        return "", f"unreadable ({exc.strerror or type(exc).__name__})"


def _split_long(title: str, body: str, line: int) -> list[Section]:
    """Break an over-long section at paragraph boundaries.

    A section is meant to be one claim about the codebase. A 900-line reference
    chapter is not, and handing it over whole makes the model attach it to
    everything it happens to mention.
    """
    out: list[Section] = []
    buffer: list[str] = []
    size = 0
    start = line
    cursor = line
    for para in re.split(r"\n\s*\n", body):
        para_lines = para.count("\n") + 1
        if size and size + len(para) > MAX_SECTION_CHARS:
            out.append(Section(title, "\n\n".join(buffer).strip(), start))
            buffer, size, start = [], 0, cursor
        buffer.append(para)
        size += len(para)
        cursor += para_lines + 1
    if buffer:
        out.append(Section(title, "\n\n".join(buffer).strip(), start))
    return [s for s in out if len(s.text) >= MIN_SECTION_CHARS]


def split(text: str, suffix: str) -> list[Section]:
    """A document, as the claims it makes.

    Markdown splits on `#` headings; reStructuredText on the underlined kind
    (`Title` over `=====`). Anything else -- plain text, a PDF -- falls back to
    paragraphs, which is what those formats give you.
    """
    lines = text.splitlines()
    marks: list[tuple[int, str]] = []      # (line index, heading text)

    if suffix in (".md", ".markdown"):
        fenced = False
        for i, raw in enumerate(lines):
            if raw.lstrip().startswith("```"):
                # A `# comment` inside a fenced code block is code, not a
                # heading. Without this every shell example in a README starts
                # a new section.
                fenced = not fenced
                continue
            if fenced:
                continue
            m = _ATX.match(raw)
            if m:
                marks.append((i, m.group(2)))
    elif suffix == ".rst":
        for i, raw in enumerate(lines):
            if i == 0 or not _RST_UNDERLINE.match(raw):
                continue
            title = lines[i - 1].strip()
            # An underline must be at least as long as what it underlines,
            # otherwise a row of dashes in a table reads as a heading.
            if title and len(raw.strip()) >= len(title) and not _RST_UNDERLINE.match(title):
                marks.append((i - 1, title))

    sections: list[Section] = []
    if marks:
        bounds = [i for i, _ in marks] + [len(lines)]
        for (start, title), end in zip(marks, bounds[1:]):
            body = "\n".join(lines[start + 1:end]).strip()
            if len(body) < MIN_SECTION_CHARS:
                continue
            sections.extend(_split_long(title, body, start + 1))
        # Text before the first heading is still a statement about the project.
        head = "\n".join(lines[:marks[0][0]]).strip()
        if len(head) >= MIN_SECTION_CHARS:
            sections = _split_long("", head, 1) + sections
    else:
        sections = _split_long("", text.strip(), 1)
    return sections


def read(root: Path, budget_tokens: int = DEFAULT_BUDGET_TOKENS) -> Reading:
    """Every document under root, in order, until the budget runs out.

    What the budget excluded is returned, not dropped. That list is printed by
    the CLI and written into the graph's coverage block, so a partial read
    cannot be mistaken for a complete one a week later.
    """
    root = root.resolve()
    reading = Reading()
    seen: list[set[str]] = []
    for path in collect(root):
        rel = str(path.relative_to(root))
        suffix = path.suffix.lower()
        if suffix in PDF_EXTENSIONS:
            text, why = read_pdf(path)
        else:
            text, why = read_text(path)
        if why:
            reading.unreadable.append((rel, why))
            continue
        sections = split(text, suffix)
        if not sections:
            continue
        cost = estimate_tokens("\n".join(s.text for s in sections))

        marks = fingerprint(text)
        if len(marks) >= DUPLICATE_FLOOR and any(
                len(marks & earlier) / len(marks) >= DUPLICATE_SHARE
                for earlier in seen):
            reading.skipped.append((rel, cost, "near-duplicate of a document already read"))
            continue

        if reading.tokens_read + cost > budget_tokens:
            reading.skipped.append((rel, cost, "over the prose budget"))
            continue
        seen.append(marks)
        reading.tokens_read += cost
        reading.documents.append(Document(path=rel, sections=sections, tokens=cost))
    return reading
