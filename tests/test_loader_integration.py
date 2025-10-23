import os

from engine.data_loaders.load_forecasts import available_iso_scenarios, load_iso_scenario


def test_discover_and_load(tmp_path):
    root = os.path.join("input")
    manifests = available_iso_scenarios(base_path=root)
    assert isinstance(manifests, list)
    if manifests:
        first = manifests[0]
        df = load_iso_scenario(root, first.iso, first.scenario)
        assert set(df.columns) == {"iso", "scenario", "zone", "year", "load_gwh"}
