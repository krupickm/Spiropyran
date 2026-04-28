from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol

IsReady = Callable[[dict[str, Any], Path], bool]
Submit = Callable[[dict[str, Any], Path, dict[str, Any]], dict[str, Any]]
Collect = Callable[[dict[str, Any], Path, dict[str, Any]], dict[str, Any]]


class Stage(Protocol):
    """Uniform stage contract (see project.md section 4).

    Stages are modules, not classes. A module satisfies this Protocol when it
    exposes is_ready, submit, and collect as top-level callables. Declaring
    them as callable attributes (rather than methods with self) lets a module
    namespace match structurally.
    """

    is_ready: IsReady
    submit: Submit
    collect: Collect
