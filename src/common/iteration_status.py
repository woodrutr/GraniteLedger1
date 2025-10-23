"""Shared iteration status structures for solver loops."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Mapping


_MODULE = sys.modules.get(__name__)
if _MODULE is not None:
    sys.modules[__name__] = _MODULE
    sys.modules.setdefault("src.common.iteration_status", _MODULE)


@dataclass(frozen=True)
class IterationStatus:
    """Describes the outcome of an iterative procedure.

    Attributes
    ----------
    iterations:
        Number of iterations that were executed.
    converged:
        ``True`` when the stopping criteria were met before hitting the
        iteration limit.
    limit:
        Maximum number of iterations that were allowed.
    message:
        Optional human-readable detail about the termination condition.
    metadata:
        Optional mapping of additional contextual information.
    """

    iterations: int
    converged: bool
    limit: int | None = None
    message: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


__all__ = ["IterationStatus"]
