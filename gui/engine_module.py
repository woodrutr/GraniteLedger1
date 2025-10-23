import sys
import importlib
import pathlib
import os


def ensure_engine_package():
    """
    Ensure that the engine package can be imported when running the Streamlit GUI.
    This adjusts sys.path so that 'engine' resolves to the project-local engine/.
    """
    root = pathlib.Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        importlib.import_module("engine")
    except ImportError as e:
        raise RuntimeError("Could not import engine package") from e

    if os.getenv("GRANITELEDGER_INPUT_ROOT"):
        return

    csv_path = root / "input" / "electricity" / "load_forecasts" / "load_forecasts.csv"
    if csv_path.exists():
        try:
            from engine.settings import configure_load_forecast_path
        except Exception:
            # Fail silently if the helper is unavailable (backwards compatibility).
            return

        configure_load_forecast_path(csv_path)


__all__ = ["ensure_engine_package"]
