"""Which language reads which file.

Every language module exposes `EXTENSIONS`, `available()`, `WHY` and a
`parse(source, parsed)` that fills the language-neutral shapes in
`graphpaat.parse`. Nothing downstream -- resolution, storage, query -- knows a
language exists.

A language whose parser is not installed is skipped rather than crashing the
run, and `missing()` says which, so the coverage report can be honest about a
file it did not read instead of quietly leaving it out.

**Order matters where two languages claim one extension.** `.h` is the only
real case: it is C's header extension and also C++'s, and a header decides
nothing about which language wrote it. Measured 2026-09-07 over 108 real
headers, the C++ grammar ties the C grammar on C headers (5 errors against 5 on
curl, 10 against 10 on redis) and is fourteen times better on C++ ones (727
against 10,009 on fmt). So C++ is registered after C and takes `.h` when its
grammar is installed; with only the `c` extra installed, C still reads them.
"""
from __future__ import annotations

from pathlib import Path

from . import python

_MODULES = [python]
_UNAVAILABLE: dict[str, str] = {}

# tree-sitter languages are optional: reading Python needs nothing installed,
# and a user who only has Python code should not have to carry grammars.
#
# C++ follows C deliberately -- see the note above about `.h`.
for _name in ("go", "typescript", "javascript", "java", "csharp", "rust",
              "ruby", "php", "swift", "kotlin", "bash", "lua", "c", "cpp"):
    try:
        _mod = __import__(f"graphpaat.languages.{_name}", fromlist=[_name])
        if _mod.available():
            _MODULES.append(_mod)
        else:
            _UNAVAILABLE[_name] = _mod.WHY
    except Exception as _exc:         # pragma: no cover - depends on install
        # Deliberately wider than ImportError. A grammar package that installs
        # but fails to load raises something else entirely, and one unreadable
        # language must never take the other thirteen down with it.
        _UNAVAILABLE[_name] = str(_exc)

EXTENSIONS: dict[str, object] = {}
for _module in _MODULES:
    for _ext in _module.EXTENSIONS:
        EXTENSIONS[_ext] = _module


def for_path(path: Path):
    """The language module owning this file, or None."""
    return EXTENSIONS.get(path.suffix.lower())


def missing() -> dict[str, str]:
    """Languages we could have read but cannot, and why."""
    return dict(_UNAVAILABLE)


def names() -> list[str]:
    return sorted(m.__name__.rsplit(".", 1)[-1] for m in _MODULES)
