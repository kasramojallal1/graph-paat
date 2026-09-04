"""Which language reads which file.

Every language module exposes `EXTENSIONS` and a `parse(source, parsed)` that
fills the language-neutral shapes in `graphpaat.parse`. Nothing downstream --
resolution, storage, query -- knows a language exists.

A language whose parser is not installed is skipped rather than crashing the
run, and `missing()` says which, so the coverage report can be honest about a
file it did not read instead of quietly leaving it out.
"""
from __future__ import annotations

from pathlib import Path

from . import python

_MODULES = [python]
_UNAVAILABLE: dict[str, str] = {}

# tree-sitter languages are optional: reading Python needs nothing installed,
# and a user who only has Python code should not have to carry grammars.
for _name in ("go",):
    try:
        _mod = __import__(f"graphpaat.languages.{_name}", fromlist=[_name])
        if _mod.available():
            _MODULES.append(_mod)
        else:
            _UNAVAILABLE[_name] = _mod.WHY
    except ImportError as _exc:       # pragma: no cover - depends on install
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
