import logging
from pathlib import Path

import pytest

from engine.data_loaders import load_forecasts
from engine.io import load_forecasts_strict as strict


def test_load_table_logs_strict_validation_error(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    root = tmp_path / "load_forecasts"
    csv_path = root / "nyiso" / "Baseline" / "ZoneA.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("Year,Load_GWh\n2020,\n", encoding="utf-8")

    with caplog.at_level(logging.ERROR, logger=load_forecasts.LOGGER.name):
        with pytest.raises(strict.ValidationError):
            load_forecasts.load_table(base_path=root)

    error_logs = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(error_logs) == 1
    assert error_logs[0].message == (
        f"Strict load forecast validation failed for {csv_path}: missing value"
    )
