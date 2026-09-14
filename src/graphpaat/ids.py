"""Node identity.

Everything in graph-paat depends on this file. The two extraction lanes join
on *exact id string match* -- so an id that is slightly wrong does not cause a
slightly wrong graph, it causes a duplicate node that nothing will ever
reconcile.

Each rule below was fitted against real corpora: mint an id, look at what
collided or split, find the rule that explains it, repeat.
"""
from __future__ import annotations

from pathlib import Path


def normalise(part: str) -> str:
    """Lowercase. Underscores are kept, because they carry meaning.

    This used to strip underscores from both ends, and the reason it no longer
    does is measured.

    Stripping destroys a standard Python idiom. Django writes a public entry
    point that delegates to a private implementation:

        def changeform_view(self, ...):        # options.py:1817
            return self._changeform_view(...)
        def _changeform_view(self, ...):       # options.py:1821

    Two different methods. Stripping underscores minted one id for both and one
    was lost. That cost **197 symbols on Django**, where the idiom is everywhere.

    The argument for stripping is that a language model can then reproduce an
    id from memory, forgiving of a near-miss on a private name. We do not need
    that: the `vocab` command hands the agent the exact spelling, so stripping
    was paying a cost for a benefit nothing here uses.

    Path components keep their own rule; see `file_prefix`.
    """
    return part.lower()


def normalise_path_part(part: str) -> str:
    """Lowercase, and drop underscores at either end.

    Applied to FILE names only, where the underscore carries no such meaning:
    `__init__.py` exists in every package directory and `_minhash.py` is one
    module, not the private twin of a `minhash.py`. Two files whose names differ
    only by underscores would collide, and no corpus measured here has any.
    """
    return part.strip("_").lower()


def file_prefix(path: Path, root: Path, keep_extension: bool = False) -> str:
    """The id prefix for a file: its path relative to root.

        pkg/resolve.py                -> resolve
        pkg/languages/python.py       -> languages_python
        pkg/__main__.py               -> main
        pkg/_private.py               -> private

    Using the whole relative path rather than just the filename is what keeps
    `a/utils.py` and `b/utils.py` from minting the same ids. An early version
    used the filename alone, and that single mistake was most of its errors.

    `keep_extension` keeps the extension as a final segment:

        lib/url.c -> lib_url_c
        lib/url.h -> lib_url_h

    Dropping the extension assumes one file per name, and that assumption holds
    for most languages. Where a language pairs a definition file with a
    declaration file of the same name it is simply false, and the cost is not
    small: measured on curl, `lib/` loses **159 of 385 files** to a
    name two files claim, and redis `src/` loses 67 of 218. A third of the
    corpus would arrive as duplicate nodes sharing one id.

    A language module states this about itself (`KEEP_EXTENSION = True`) rather
    than being named here, so nothing outside `languages/` learns which
    languages those are.
    """
    parts = list(path.relative_to(root).parts)
    name = Path(parts[-1])
    parts[-1] = f"{name.stem}_{name.suffix.lstrip('.')}" if keep_extension else name.stem
    return "_".join(normalise_path_part(p) for p in parts)


def mint(prefix: str, label: str, scope: list[str] | None = None) -> str:
    """`{prefix}[_{scope}...]_{label}`.

    A symbol's name is only unique within the thing that encloses it, so the
    enclosing chain goes into the id. Two cases, both real:

      class A: def __init__      -> ..._a_init
      class B: def __init__      -> ..._b_init
      def outer: def add_edge    -> ..._outer_add_edge
      def other: def add_edge    -> ..._other_add_edge

    Qualifying methods by their class is the usual choice; qualifying nested
    functions is the less usual one, because many tools do not emit nested
    functions at all. We emit them (a nested helper can do real work and being
    invisible is worse than being verbose), which means we have to qualify them
    or they collide -- one real extractor module has three nested
    `add_edge`/`_add_edge` helpers that mint one id without this.
    """
    parts = [prefix]
    for s in (scope or []):
        parts.append(normalise(s))
    parts.append(normalise(label))
    return "_".join(parts)


class Collisions:
    """Records ids claimed by more than one symbol.

    The case that is easiest to miss is same file, same label. That is the
    case where a genuinely distinct function is destroyed -- a module that
    defines `_nfc` twice, nine hundred lines apart, keeps only the first unless
    someone counts.

    The id scheme is kept and what it loses is reported. This class is that
    report. It does not fix the merge; it makes the merge visible.
    """

    def __init__(self) -> None:
        self._claims: dict[str, list[str]] = {}

    def claim(self, node_id: str, where: str) -> None:
        self._claims.setdefault(node_id, []).append(where)

    def collided(self) -> dict[str, list[str]]:
        return {i: w for i, w in self._claims.items() if len(w) > 1}

    def report(self) -> list[str]:
        lines = []
        for node_id, wheres in sorted(self.collided().items()):
            kept, dropped = wheres[0], ", ".join(wheres[1:])
            lines.append(
                f"[graph-paat] COLLISION: '{node_id}' claimed by {len(wheres)} symbols "
                f"- keeping {kept}, losing {dropped}"
            )
        return lines
