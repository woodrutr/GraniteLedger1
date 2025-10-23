"""Protocol definitions for dispatch linear programming backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence, Tuple


@dataclass(frozen=True)
class LpSolveStatus:
    """Structured status returned by LP backends."""

    success: bool
    status: int
    message: str
    dual_eq: Sequence[float] = ()
    dual_ineq: Sequence[float] = ()
    dual_lower: Sequence[float] = ()
    dual_upper: Sequence[float] = ()
    residual_eq: Sequence[float] = ()
    residual_ineq: Sequence[float] = ()


class LpBackend(Protocol):
    """Protocol describing a linear programming backend."""

    def solve(
        self,
        c: Sequence[float],
        A_ub: Sequence[Sequence[float]] | None,
        b_ub: Sequence[float] | None,
        A_eq: Sequence[Sequence[float]] | None,
        b_eq: Sequence[float] | None,
        bounds: Sequence[Tuple[float | None, float | None]],
        names: Iterable[str] | None = None,
    ) -> tuple[Sequence[float], float, LpSolveStatus]:
        """Solve the LP with the provided matrices."""


__all__ = ["LpBackend", "LpSolveStatus"]
