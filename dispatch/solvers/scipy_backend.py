"""SciPy-based linear programming backend with optional PuLP fallback."""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

from .base import LpBackend, LpSolveStatus

try:  # pragma: no cover - optional dependency guard
    from scipy.optimize import linprog  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    linprog = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency guard
    from scipy import sparse  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    sparse = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency guard
    import pulp
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pulp = None  # type: ignore[assignment]


class SciPyBackend(LpBackend):
    """Linear programming backend powered by :mod:`scipy.optimize.linprog`."""

    def __init__(self) -> None:
        # Availability is checked lazily in :meth:`solve`.  Many applications
        # probe capabilities by instantiating the backend; raising here would
        # cause the UI to disable the dispatch controls before the user even
        # attempts a solve.
        pass

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
        if not dependencies_available():
            raise ModuleNotFoundError(
                "No LP backend available. Install SciPy (`pip install scipy`) or PuLP (`pip install pulp`)."
            )

        self._validate_inputs(c, A_ub, b_ub, A_eq, b_eq, bounds)

        if linprog is not None:
            return self._solve_scipy(c, A_ub, b_ub, A_eq, b_eq, bounds)
        if pulp is not None:  # pragma: no cover - fallback path
            return self._solve_pulp(c, A_ub, b_ub, A_eq, b_eq, bounds, names)
        raise ModuleNotFoundError(  # pragma: no cover - defensive guard
            "SciPy backend requested but SciPy and PuLP are unavailable."
        )

    @staticmethod
    def _validate_inputs(
        c: Sequence[float],
        A_ub: Sequence[Sequence[float]] | None,
        b_ub: Sequence[float] | None,
        A_eq: Sequence[Sequence[float]] | None,
        b_eq: Sequence[float] | None,
        bounds: Sequence[Tuple[float | None, float | None]],
    ) -> None:
        num_vars = len(c)
        if num_vars == 0:
            raise ValueError("LP requires at least one decision variable")
        if len(bounds) != num_vars:
            raise ValueError("Bounds length must match number of variables")

        def _check_matrix(
            matrix: Sequence[Sequence[float]] | None,
            rhs: Sequence[float] | None,
            label: str,
        ) -> None:
            if matrix is None or rhs is None:
                if matrix is None and rhs is None:
                    return
                raise ValueError(f"{label} constraint matrix and rhs must both be provided")
            if len(matrix) != len(rhs):
                raise ValueError(f"{label} constraint matrix row count must match rhs length")
            for row in matrix:
                if len(row) != num_vars:
                    raise ValueError(
                        f"{label} constraint row has {len(row)} columns; expected {num_vars}"
                    )

        _check_matrix(A_eq, b_eq, "Equality")
        _check_matrix(A_ub, b_ub, "Inequality")

    @staticmethod
    def _solve_scipy(
        c: Sequence[float],
        A_ub: Sequence[Sequence[float]] | None,
        b_ub: Sequence[float] | None,
        A_eq: Sequence[Sequence[float]] | None,
        b_eq: Sequence[float] | None,
        bounds: Sequence[Tuple[float | None, float | None]],
    ) -> tuple[Sequence[float], float, LpSolveStatus]:
        if sparse is not None:
            A_eq_matrix = sparse.csr_matrix(A_eq) if A_eq is not None else None
            A_ub_matrix = sparse.csr_matrix(A_ub) if A_ub is not None else None
        else:
            A_eq_matrix = A_eq
            A_ub_matrix = A_ub

        result = linprog(  # type: ignore[misc]
            c,
            A_ub=A_ub_matrix,
            b_ub=b_ub,
            A_eq=A_eq_matrix,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

        def _extract(attribute: str, field: str) -> Tuple[float, ...]:
            block = getattr(result, attribute, None)
            values = getattr(block, field, None)
            if values is None:
                return ()
            return tuple(float(value) for value in values)

        status = LpSolveStatus(
            success=bool(result.success),
            status=int(result.status),
            message=str(result.message),
            dual_eq=_extract("eqlin", "marginals"),
            dual_ineq=_extract("ineqlin", "marginals"),
            dual_lower=_extract("lower", "marginals"),
            dual_upper=_extract("upper", "marginals"),
            residual_eq=_extract("eqlin", "residual"),
            residual_ineq=_extract("ineqlin", "residual"),
        )

        values_raw = getattr(result, "x", None)
        if values_raw is None:
            values = ()
        else:
            values = tuple(float(value) for value in values_raw)
        objective = float(result.fun) if result.fun is not None else math.nan
        return values, objective, status

    @staticmethod
    def _solve_pulp(
        c: Sequence[float],
        A_ub: Sequence[Sequence[float]] | None,
        b_ub: Sequence[float] | None,
        A_eq: Sequence[Sequence[float]] | None,
        b_eq: Sequence[float] | None,
        bounds: Sequence[Tuple[float | None, float | None]],
        names: Iterable[str] | None,
    ) -> tuple[Sequence[float], float, LpSolveStatus]:  # pragma: no cover - fallback path
        problem = pulp.LpProblem("dispatch", pulp.LpMinimize)
        variable_names = list(names) if names is not None else [f"x{i}" for i in range(len(c))]
        variables = []
        for idx, (name, (low, high)) in enumerate(zip(variable_names, bounds)):
            var = pulp.LpVariable(name, lowBound=low, upBound=high, cat="Continuous")
            variables.append(var)
        problem += pulp.lpSum(cost * var for cost, var in zip(c, variables))

        eq_constraints: list[pulp.LpConstraint] = []
        ub_constraints: list[pulp.LpConstraint] = []

        if A_eq is not None and b_eq is not None:
            for idx, (row, rhs) in enumerate(zip(A_eq, b_eq)):
                constraint = pulp.lpSum(coeff * var for coeff, var in zip(row, variables)) == rhs
                problem += constraint, f"eq_{idx}"
                eq_constraints.append(constraint)
        if A_ub is not None and b_ub is not None:
            for idx, (row, rhs) in enumerate(zip(A_ub, b_ub)):
                constraint = pulp.lpSum(coeff * var for coeff, var in zip(row, variables)) <= rhs
                problem += constraint, f"ub_{idx}"
                ub_constraints.append(constraint)

        status_code = problem.solve(pulp.PULP_CBC_CMD(msg=False))
        success = status_code == pulp.LpStatusOptimal
        values = tuple(var.value() if var.value() is not None else math.nan for var in variables)
        objective = float(problem.objective.value()) if success else math.nan

        dual_eq: tuple[float, ...] = ()
        dual_ineq: tuple[float, ...] = ()
        if success:
            dual_eq = tuple(getattr(constraint, "pi", 0.0) for constraint in eq_constraints)
            dual_ineq = tuple(getattr(constraint, "pi", 0.0) for constraint in ub_constraints)

        status = LpSolveStatus(
            success=success,
            status=int(status_code),
            message=pulp.LpStatus[status_code],
            dual_eq=dual_eq,
            dual_ineq=dual_ineq,
        )
        return values, objective, status


def dependencies_available() -> bool:
    """Return whether at least one optional solver dependency is importable."""

    return bool(linprog is not None or pulp is not None)


__all__ = ["SciPyBackend", "dependencies_available"]
