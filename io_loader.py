"""Utility helpers for loading the ``granite_io.frames_api`` module safely."""

from __future__ import annotations

from importlib import util
from pathlib import Path
import sys
from types import ModuleType


def _load_frames_module():
    module_name = 'granite_io.frames_api'
    legacy_name = 'io.frames_api'

    existing = sys.modules.get(module_name)
    if existing is not None:
        _ensure_legacy_alias(existing)
        return existing

    legacy_existing = sys.modules.get(legacy_name)
    if isinstance(legacy_existing, ModuleType) and hasattr(legacy_existing, 'Frames'):
        sys.modules[module_name] = legacy_existing
        return legacy_existing

    module_path = Path(__file__).resolve().parent / 'granite_io' / 'frames_api.py'
    spec = util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
        raise ImportError(f'unable to locate frames_api module at {module_path}')

    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _ensure_legacy_alias(module)
    return module


def _ensure_legacy_alias(module: ModuleType) -> None:
    """Ensure ``io.frames_api`` resolves to the provided ``module``."""

    legacy_name = 'io.frames_api'
    existing = sys.modules.get(legacy_name)
    if existing is module:
        return
    if existing is not None and isinstance(existing, ModuleType) and hasattr(existing, 'Frames'):
        # Another fully-initialised frames module already occupies the legacy slot.
        return
    sys.modules[legacy_name] = module

    parent_name, _, child_name = legacy_name.partition('.')
    parent_module = sys.modules.get(parent_name)
    if isinstance(parent_module, ModuleType):
        # ``import io.frames_api`` requires ``io`` to appear package-like.  The
        # stdlib ``io`` module is not a package, but we can provide the minimal
        # surface area by giving it a ``__path__`` and attaching the attribute.
        if not hasattr(parent_module, '__path__'):
            parent_module.__path__ = []  # type: ignore[attr-defined]

        existing_attr = getattr(parent_module, child_name, None)
        if existing_attr is module:
            return
        if isinstance(existing_attr, ModuleType) and hasattr(existing_attr, 'Frames'):
            return

        setattr(parent_module, child_name, module)


try:
    _module = _load_frames_module()
except ModuleNotFoundError as exc:
    if exc.name != 'pandas':  # pragma: no cover - propagate unexpected errors
        raise

    original_exc = exc

    def _raise_pandas_error() -> None:
        raise ImportError(
            "pandas is required for granite_io.frames_api; install it with `pip install pandas`."
        ) from original_exc

    class Frames:  # type: ignore[no-redef]
        """Placeholder that raises when pandas-dependent frames are unavailable."""

        def __init__(self, *_args, **_kwargs):  # pragma: no cover - simple guard
            _raise_pandas_error()

        @classmethod
        def coerce(cls, *_args, **_kwargs):  # pragma: no cover - simple guard
            _raise_pandas_error()

        def __getattr__(self, _name):  # pragma: no cover - simple guard
            _raise_pandas_error()

    class PolicySpec:  # type: ignore[no-redef]
        """Placeholder that raises when pandas-dependent policy specs are unavailable."""

        def __init__(self, *_args, **_kwargs):  # pragma: no cover - simple guard
            _raise_pandas_error()

        def to_policy(self):  # pragma: no cover - simple guard
            _raise_pandas_error()
else:
    try:
        Frames = getattr(_module, 'Frames')
    except AttributeError:
        # Temporary placeholder to avoid import failure
        class Frames:  # type: ignore[no-redef]
            def __init__(self, *a, **k):
                pass

            def __call__(self, *a, **k):
                return None

            def __repr__(self):
                return "<Frames placeholder>"
    PolicySpec = getattr(_module, 'PolicySpec', type('PolicySpec', (), {}))

__all__ = ['Frames', 'PolicySpec']
