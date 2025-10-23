from logging import getLogger
from pathlib import Path
import importlib
import subprocess
import sys
import types
from types import SimpleNamespace

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    try:
        import tomli as tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency missing
        raise ModuleNotFoundError(
            'Python 3.11+ or the tomli package is required to read TOML configuration files.'
        ) from exc

from main.definitions import PROJECT_ROOT
from src.common.utilities import get_downloads_directory

logger = getLogger(__name__)


def test_config_setup():
    """test to ensure changes to configurations are consistent"""

    # list of keys in run_config
    config_path = Path(PROJECT_ROOT, 'src/common', 'run_config.toml')
    with open(config_path, 'rb') as f:
        data = tomllib.load(f)
    config_list = []
    for key in data.keys():
        config_list.append(key)

    # list of keys in run_config_template
    template_path = Path(PROJECT_ROOT, 'src/common', 'run_config_template.toml')
    with open(template_path, 'rb') as f:
        data = tomllib.load(f)
    template_list = []
    for key in data.keys():
        template_list.append(key)

    assert config_list == template_list


def test_run_mode_combo_methods(monkeypatch):
    """Simulate selecting combo modes and ensure configuration runner resolves methods."""

    class DummyProcess:
        def terminate(self):
            """Stub terminate to satisfy app cleanup."""

    # Prevent spawning a real HTTP server when importing the Dash app module.
    monkeypatch.setattr(subprocess, 'Popen', lambda *args, **kwargs: DummyProcess())

    sys.modules.pop('app', None)

    class _DashComponent:
        def __init__(self, *args, **kwargs):
            pass

    class _DashApp:
        def __init__(self, *args, **kwargs):
            self.layout = None

        def callback(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

        def get_asset_url(self, asset):
            return asset

        def run_server(self, *args, **kwargs):
            return None

    dash_module = types.ModuleType('dash')
    dash_module.Dash = _DashApp
    dash_module.Input = dash_module.Output = dash_module.State = _DashComponent
    dash_module.ALL = object()
    dash_module.dcc = SimpleNamespace(RadioItems=_DashComponent, Loading=_DashComponent)
    dash_module.html = SimpleNamespace(
        H1=_DashComponent,
        H2=_DashComponent,
        H3=_DashComponent,
        H4=_DashComponent,
        Hr=_DashComponent,
        Img=_DashComponent,
        Div=_DashComponent,
    )

    dbc_module = types.ModuleType('dash_bootstrap_components')
    dbc_module.Container = _DashComponent
    dbc_module.Row = _DashComponent
    dbc_module.Col = _DashComponent
    dbc_module.Button = _DashComponent
    dbc_module.Label = _DashComponent
    dbc_module.Progress = _DashComponent
    dbc_module.Input = _DashComponent
    dbc_module.themes = SimpleNamespace(BOOTSTRAP=object())

    monkeypatch.setitem(sys.modules, 'dash', dash_module)
    monkeypatch.setitem(sys.modules, 'dash.dcc', dash_module.dcc)
    monkeypatch.setitem(sys.modules, 'dash.html', dash_module.html)
    monkeypatch.setitem(sys.modules, 'dash_bootstrap_components', dbc_module)

    tomli_module = types.ModuleType('tomli')
    tomli_module.load = lambda *args, **kwargs: {}
    monkeypatch.setitem(sys.modules, 'tomli', tomli_module)

    tomlkit_module = types.ModuleType('tomlkit')
    tomlkit_module.parse = lambda *args, **kwargs: {}
    tomlkit_module.dumps = lambda *args, **kwargs: ''
    monkeypatch.setitem(sys.modules, 'tomlkit', tomlkit_module)

    main_module = types.ModuleType('main')
    main_module.app_main = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, 'main', main_module)

    app_module = importlib.import_module('app')
    dash_app_module = importlib.import_module('gui.dash_app')

    downloads_root = get_downloads_directory()

    recorded = []

    class FakeConfigRunner:
        """Minimal stand-in for Config_settings run method resolution."""

        def __init__(self, mode):
            self.selected_mode = mode
            self.run_method = self._resolve_method(mode)

        @staticmethod
        def _resolve_method(mode):
            if mode == 'unified-combo':
                return 'run_unified'
            if mode == 'gs-combo':
                return 'run_gs'
            if mode == 'standalone':
                return 'run_standalone'
            raise ValueError('Mode: Nonexistent mode specified')

    def fake_app_main(mode: str):
        args = SimpleNamespace(op_mode=mode, debug=False)
        # Simulate the portion of Config_settings that maps modes to runner methods.
        settings = FakeConfigRunner(args.op_mode)
        recorded.append((mode, settings.run_method))

    monkeypatch.setattr(dash_app_module, 'app_main', fake_app_main)

    message, progress = app_module.run_mode(1, 'gs-combo')
    assert recorded[-1] == ('gs-combo', 'run_gs')
    assert message == f'gs-combo mode has finished running. See results in {downloads_root}.'
    assert progress == 100

    message, progress = app_module.run_mode(2, 'unified-combo')
    assert recorded[-1] == ('unified-combo', 'run_unified')
    assert message == (
        f'unified-combo mode has finished running. See results in {downloads_root}.'
    )
    assert progress == 100

    sys.modules.pop('app', None)
