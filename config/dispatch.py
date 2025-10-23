"""Dispatch configuration helpers."""

from __future__ import annotations

import os

DISPATCH_SOLVER = os.environ.get("DISPATCH_SOLVER", "scipy").strip().lower() or "scipy"


__all__ = ["DISPATCH_SOLVER"]
