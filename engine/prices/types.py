from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class CarbonPriceVector:
    """Canonical container for carbon price components.

    all: price applied to all covered emissions (allowance component)
    effective: price after program design features (e.g., CCR/ECR floors or triggers)
    exempt: price for uncovered/exempt emissions buckets (often 0)
    last: prior-year or prior-iteration scalar reference (optional)
    year: the model year this vector refers to (optional)
    """
    all: float
    effective: float
    exempt: float
    last: Optional[float] = None
    year: Optional[int] = None
