"""
BlueSky Graphical User Interface.

Built using Dash - https://dash.plotly.com/.

Created on Wed Sept 19 2024 by Adam Heisey
"""

# Import packages
import csv
import dash
from dash import dash_table, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import ast
from collections.abc import Mapping
import logging
import subprocess
from pathlib import Path
import os
import tomli
import tomlkit
import sys

LOGGER = logging.getLogger(__name__)

# Import python modules
from main.definitions import PROJECT_ROOT
from main import app_main
from pathlib import Path
import logging

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency shim
    from src.common.utilities import get_downloads_directory as _get_downloads_directory
except ImportError:  # pragma: no cover - compatibility fallback
    _get_downloads_directory = None

_download_directory_fallback_used = False


def _fallback_downloads_directory(app_subdir: str = 'GraniteLedger') -> Path:
    """Return a reasonable downloads location when utilities helper is unavailable."""
    base_path = Path.home() / 'Downloads'
    if app_subdir:
        base_path = base_path / app_subdir
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path


def get_downloads_directory(app_subdir: str = 'GraniteLedger') -> Path:
    """Resolve the downloads directory, falling back to the user's home folder."""
    global _download_directory_fallback_used

    if _get_downloads_directory is not None:
        try:
            return _get_downloads_directory(app_subdir=app_subdir)
        except Exception:  # pragma: no cover - defensive: ensure GUI still loads
            LOGGER.warning('Falling back to home Downloads directory; helper raised an error.')
    if not _download_directory_fallback_used:
        LOGGER.warning(
            'get_downloads_directory is unavailable; using ~/Downloads for model outputs.'
        )
        _download_directory_fallback_used = True
    return _fallback_downloads_directory(app_subdir)

from src.models.electricity.scripts.technology_metadata import (
    get_technology_label,
    resolve_technology_key,
)
from src.models.electricity.scripts.incentives import TechnologyIncentives


ELECTRICITY_OVERRIDES_KEY = 'electricity_expansion_overrides'
ELECTRICITY_INCENTIVES_KEY = 'electricity_incentives'
SW_BUILDS_PATH = Path(PROJECT_ROOT, 'input', 'electricity', 'sw_builds.csv')

# Debugging must never be enabled in production deployments because Dash's
# debugger exposes a remote code execution surface. Developers can opt in
# locally by exporting the environment variable documented in the README.
DASH_DEBUG_ENV_VAR = 'BLUESKY_DASH_DEBUG'


def dash_debug_enabled() -> bool:
    """Return whether the Dash app should run in debug mode.

    Debug mode is only intended for local development and should never be
    enabled in production deployments because it exposes the werkzeug
    debugger and automatic reloader.
    """

    raw_value = os.getenv(DASH_DEBUG_ENV_VAR)

    if raw_value is None:
        return False

    return raw_value.strip().lower() in {'1', 'true', 't', 'yes', 'y', 'on'}


# Initialize the Dash app
app = dash.Dash(
    __name__,
    prevent_initial_callbacks=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    assets_folder='docs/images/',
)
app.title = 'BlueSky Model Runner'

docs_dir = os.path.abspath('docs/build/html')


def _launch_docs_server():
    """Start the HTML documentation server bound to localhost."""

    # Use the current Python interpreter to serve the built docs in the
    # background. Binding to 127.0.0.1 limits exposure to the local machine.
    with open(os.devnull, 'w') as devnull:
        process = subprocess.Popen(
            [
                sys.executable,
                '-m',
                'http.server',
                '8000',
                '--bind',
                '127.0.0.1',
                '--directory',
                docs_dir,
            ],
            stdout=devnull,
            stderr=devnull,
        )

    return process

# blusesky image in assets folder
image_src = app.get_asset_url('ProjectBlueSkywebheaderimageblack.jpg')

# Define layout
app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                dbc.Button(
                    'Code Documentation',
                    href='http://127.0.0.1:8000/index.html',
                    color='info',
                    className='mt-3',
                    target='_blank',
                ),
                width='auto',
                className='text-left',
            ),
            justify='start',
        ),
        html.H1('BlueSky Model Runner', className='text-center'),
        html.Img(src=image_src),
        html.H2(id='status', className='text-center', style={'color': 'red'}),
        html.H3(id='output-state'),
        dbc.Label('Select Mode to Run:'),
        dcc.RadioItems(
            id='mode-selector',
            options=[
                {'label': mode, 'value': mode}
                for mode in ['unified-combo', 'gs-combo', 'standalone']
            ],
            value='standalone',
        ),
        dbc.Button('Run', id='run-button', color='primary', className='mt-2'),
        dcc.Loading(dbc.Progress(id='progress', value=0, max=100, style={'height': '30px'})),
        # Section for uploading and editing TOML config file
        html.Hr(),
        html.H4('Edit Configuration Settings'),
        # dcc.Upload(id='upload-toml', children=html.Button('Upload TOML'), multiple=False),
        html.Div(id='config-editor'),
        dbc.Button('Save Changes', id='save-toml-button', className='mt-2', disabled=False),
    ],
    fluid=True,
)


# Auto load the template
@app.callback(
    Output('config-editor', 'children'),
    Output('save-toml-button', 'disabled'),
    Input('config-editor', 'id'),
    prevent_initial_call=False,
)
def auto_load_toml(selected_mode):
    """reads in the configuration settings into the app that are saved in the config template.

    Parameters
    ----------
    selected_mode :
        user selected run mode option, current options are 'unified-combo', 'gs-combo', 'standalone'

    Returns
    -------
        config default settings
    """
    config_template_path = Path('src/common', 'run_config_template.toml')
    config_file_path = Path('src/common', 'run_config.toml')

    config_source_path = config_file_path if config_file_path.exists() else config_template_path

    if not config_source_path.exists():
        return [], True

    with open(config_source_path, 'rb') as f:
        config_content = tomli.load(f)

    general_inputs = []

    for key, value in config_content.items():
        if key in {ELECTRICITY_OVERRIDES_KEY, ELECTRICITY_INCENTIVES_KEY}:
            continue
        general_inputs.append(
            html.Div(
                [
                    dbc.Label(f'{key}:'),
                    dbc.Input(
                        id={'type': 'config-input', 'index': key},
                        value=str(value),
                        debounce=True,
                    ),
                ],
                style={'margin-bottom': '10px'},
            )
        )

    electricity_tab = build_electricity_override_tab(
        config_content.get(ELECTRICITY_OVERRIDES_KEY, {}),
        config_content.get(ELECTRICITY_INCENTIVES_KEY, {}),
    )

    tabs = dcc.Tabs(
        [
            dcc.Tab(label='General', value='general', children=general_inputs),
            dcc.Tab(label='Electricity', value='electricity', children=electricity_tab),
        ],
        value='general',
    )

    return tabs, False


# Save the modified TOML file with comments
@app.callback(
    Output('output-state', 'children'),
    Input('save-toml-button', 'n_clicks'),
    State({'type': 'config-input', 'index': dash.ALL}, 'value'),
    State({'type': 'config-input', 'index': dash.ALL}, 'id'),
    State({'type': 'expansion-toggle', 'index': dash.ALL}, 'value'),
    State({'type': 'expansion-toggle', 'index': dash.ALL}, 'id'),
    State('incentives-table', 'data'),
    prevent_initial_call=True,
)
def save_toml(
    n_clicks,
    input_values,
    input_ids,
    toggle_values,
    toggle_ids,
    incentive_rows,
):
    """saves the configuration settings in the app to the config file.

    Parameters
    ----------
    n_clicks :
        click to save toml button
    input_values :
        config values associated with components specified in the web app
    input_ids :
        config components associated with values specified in the web app

    Returns
    -------
        empty string
    """
    config_template = os.path.join('src/common', 'run_config_template.toml')
    config_file_path = os.path.join('src/common', 'run_config.toml')

    if n_clicks:
        # Load the original file to preserve its structure and comments
        with open(config_template, 'r') as f:
            config_doc = tomlkit.parse(f.read())

        # Update the config_doc with new values
        for item, value in zip(input_ids, input_values):
            config_doc[item['index']] = convert_value(value)

        overrides_table = tomlkit.table()
        overrides = {}
        for item, value in zip(toggle_ids or [], toggle_values or []):
            tech_index = item.get('index')
            tech_id = resolve_technology_key(tech_index)
            if tech_id is None:
                continue
            overrides[int(tech_id)] = bool(value)
        if overrides:
            for tech_id in sorted(overrides):
                label = get_technology_label(tech_id)
                overrides_table[label] = overrides[tech_id]
            config_doc[ELECTRICITY_OVERRIDES_KEY] = overrides_table
        elif ELECTRICITY_OVERRIDES_KEY in config_doc:
            del config_doc[ELECTRICITY_OVERRIDES_KEY]

        incentives_obj = TechnologyIncentives.from_table_rows(incentive_rows)
        incentives_config = incentives_obj.to_config()
        if incentives_config:
            incentives_table = incentives_config_to_toml(incentives_config)
            if len(incentives_table) > 0:
                config_doc[ELECTRICITY_INCENTIVES_KEY] = incentives_table
        elif ELECTRICITY_INCENTIVES_KEY in config_doc:
            del config_doc[ELECTRICITY_INCENTIVES_KEY]

        # Write the updated content back, preserving comments
        with open(config_file_path, 'w') as f:
            f.write(tomlkit.dumps(config_doc))

        return "Configuration settings saved successfully as 'run_config.toml'."
    return ''


def build_electricity_override_tab(overrides_config, incentives_config):
    """Create the electricity tab content with technology toggles and incentives."""

    tech_overrides = normalize_override_config(overrides_config)
    available_techs = load_available_technologies()

    if not available_techs:
        return [html.Div('No electricity technologies found.')]

    switches = []
    for tech_id in available_techs:
        label = get_technology_label(tech_id)
        value = tech_overrides.get(tech_id, True)
        switches.append(
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Switch(
                            id={'type': 'expansion-toggle', 'index': str(tech_id)},
                            label=label,
                            value=value,
                        ),
                        width='auto',
                    )
                ],
                className='mb-2',
            )
        )

    incentives_obj = TechnologyIncentives.from_config(incentives_config)
    incentive_rows = format_incentive_rows(
        incentives_obj.to_table_rows(), available_techs
    )

    incentives_section = build_incentives_editor(available_techs, incentive_rows)

    description = html.Div(
        [
            html.P(
                'Toggle capacity expansion eligibility for each electricity technology.',
                className='mb-1',
            ),
            html.P(
                'Disabling a technology prevents the optimizer from building new capacity.',
                className='text-muted',
            ),
        ]
    )

    return [description] + switches + [html.Hr(), incentives_section]


def build_incentives_editor(available_techs, incentive_rows):
    """Return a Dash component tree for editing technology incentives."""

    tech_options = [
        {'label': get_technology_label(tech_id), 'value': get_technology_label(tech_id)}
        for tech_id in available_techs
    ]
    if not tech_options:
        tech_options = [{'label': 'Select technology', 'value': ''}]

    table = dash_table.DataTable(
        id='incentives-table',
        columns=[
            {'name': 'Type', 'id': 'type', 'presentation': 'dropdown'},
            {'name': 'Technology', 'id': 'technology', 'presentation': 'dropdown'},
            {'name': 'Year', 'id': 'year', 'type': 'numeric'},
            {'name': 'Credit ($/unit)', 'id': 'credit_value', 'type': 'numeric'},
            {'name': 'Limit', 'id': 'limit_value', 'type': 'numeric'},
            {'name': 'Limit Units', 'id': 'limit_units', 'presentation': 'dropdown', 'editable': False},
        ],
        data=incentive_rows,
        editable=True,
        row_deletable=True,
        style_table={'overflowX': 'auto'},
        dropdown={
            'type': {
                'options': [
                    {'label': 'Production (credit $/MWh)', 'value': 'Production'},
                    {'label': 'Investment (credit $/MW)', 'value': 'Investment'},
                ]
            },
            'technology': {'options': tech_options},
            'limit_units': {
                'options': [
                    {'label': 'MWh', 'value': 'MWh'},
                    {'label': 'MW', 'value': 'MW'},
                ]
            },
        },
        css=[{'selector': '.dash-spreadsheet-menu', 'rule': 'display: none'}],
    )

    return html.Div(
        [
            html.H5('Technology Incentives'),
            html.P(
                'Specify optional production ($/MWh) or investment ($/MW) incentives by '
                'technology and year. Limits cap the eligible quantity of generation '
                'or capacity that can receive the credit.',
                className='text-muted',
            ),
            table,
            dbc.Button(
                'Add Incentive',
                id='add-incentive',
                color='secondary',
                className='mt-2',
            ),
        ],
        className='mt-3',
    )


def format_incentive_rows(raw_rows, available_techs):
    """Normalise incentive rows for display in the DataTable."""

    valid_labels = {get_technology_label(tech_id) for tech_id in available_techs}
    formatted = []
    for row in raw_rows or []:
        if not isinstance(row, Mapping):
            continue
        entry = {
            'type': row.get('type', 'Production'),
            'technology': row.get('technology', ''),
            'year': row.get('year'),
            'credit_value': row.get('credit_value'),
            'limit_value': row.get('limit_value'),
            'limit_units': row.get('limit_units', 'MWh'),
        }
        if entry['technology'] not in valid_labels:
            entry['technology'] = ''
        if entry['limit_value'] is None:
            entry['limit_value'] = ''
        formatted.append(entry)

    if not formatted:
        formatted = [default_incentive_row(available_techs)]

    return formatted


def default_incentive_row(available_techs):
    """Return a default incentive row for initial table population."""

    technology_label = get_technology_label(available_techs[0]) if available_techs else ''
    return {
        'type': 'Production',
        'technology': technology_label,
        'year': None,
        'credit_value': None,
        'limit_value': None,
        'limit_units': 'MWh',
    }


def incentives_config_to_toml(incentives_config):
    """Convert an incentives configuration mapping into a TOML table."""

    table = tomlkit.table()
    for key in ('production', 'investment'):
        entries = incentives_config.get(key, []) if isinstance(incentives_config, Mapping) else []
        if not entries:
            continue
        aot = tomlkit.aot()
        for record in entries:
            if not isinstance(record, Mapping):
                continue
            entry = tomlkit.table()
            for field, value in record.items():
                if value is None:
                    continue
                entry.add(field, value)
            aot.append(entry)
        if len(aot) > 0:
            table.add(key, aot)
    return table


@app.callback(
    Output('incentives-table', 'data'),
    Input('add-incentive', 'n_clicks'),
    Input('incentives-table', 'data_timestamp'),
    State('incentives-table', 'data'),
    prevent_initial_call=True,
)
def update_incentives_table(add_clicks, _timestamp, rows):
    """Handle add-row events and keep limit units aligned with credit type."""

    ctx = getattr(dash, 'callback_context', None)
    if ctx is None:
        ctx = getattr(dash, 'ctx', None)
    if ctx is None or not getattr(ctx, 'triggered', None):
        raise PreventUpdate

    trigger = ctx.triggered[0]['prop_id']
    rows = list(rows or [])

    if trigger.startswith('add-incentive'):
        available = load_available_technologies()
        rows.append(default_incentive_row(available))
        return rows

    if trigger.startswith('incentives-table'):
        if not rows:
            raise PreventUpdate
        changed = False
        updated_rows = []
        for row in rows:
            expected_unit = 'MWh'
            if str(row.get('type', '')).strip().lower().startswith('inv'):
                expected_unit = 'MW'
            if row.get('limit_units') != expected_unit:
                new_row = dict(row)
                new_row['limit_units'] = expected_unit
                updated_rows.append(new_row)
                changed = True
            else:
                updated_rows.append(row)
        if changed:
            return updated_rows

    raise PreventUpdate


def normalize_override_config(overrides_config):
    """Normalize override configuration values into a mapping."""

    normalized = {}
    if isinstance(overrides_config, dict):
        for key, value in overrides_config.items():
            tech_id = resolve_technology_key(key)
            if tech_id is None:
                continue
            normalized[tech_id] = bool(value)
    return normalized


def load_available_technologies():
    """Load the technology identifiers present in the build switches file."""

    if not SW_BUILDS_PATH.exists():
        return []

    tech_ids = set()
    with SW_BUILDS_PATH.open(newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            tech_raw = row.get('tech')
            try:
                tech_id = int(tech_raw)
            except (TypeError, ValueError):
                continue
            tech_ids.add(tech_id)

    return sorted(tech_ids)


# Function to convert values back to original types
def convert_value(value):
    """Function to maintain the original type for config values"""
    # handle boolean speficially
    if isinstance(value, str):
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
    # now other types
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


# Callback to handle run button click and show progress
@app.callback(
    Output('status', 'children'),
    Output('progress', 'value'),
    Input('run-button', 'n_clicks'),
    State('mode-selector', 'value'),
    prevent_initial_call=True,
)
def run_mode(n_clicks, selected_mode):
    """passes the selected mode to main.py and runs the script.

    Parameters
    ----------
    n_clicks :
        click to the run button
    selected_mode :
        user selected run mode option, current options are 'unified-combo', 'gs-combo', 'standalone'

    Returns
    -------
        message stating either: model has finished or there was an error and it wasn't able to run
    """
    # define modes allowed - sanitize user input
    modes_available = {'unified-combo', 'gs-combo', 'standalone'}

    if selected_mode not in modes_available:
        return f"Error: '{selected_mode}' is not a valid mode.", 0

    try:
        # run selected mode
        app_main(selected_mode)

        downloads_root = get_downloads_directory()
        return (
            f"{selected_mode} mode has finished running. See results in {downloads_root}.",
            100,
        )
    except Exception:
        error_msg = f'Error, not able to run {selected_mode}. Please check the log script/terminal, exit out of browser, and restart.'
        return error_msg, 0

# ... your callback definitions above ...

http_server_process = None
debug_mode = dash_debug_enabled()

if debug_mode:
    print(
        'Dash debug mode enabled via '
        f"{DASH_DEBUG_ENV_VAR}. Do not enable this in production environments."
    )

if __name__ == "__main__":
    try:
        # Only expose the documentation server when running this module directly.
        # Bind explicitly to loopback so it is not accessible externally.
        with open(os.devnull, "w") as devnull:
            http_server_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "http.server",
                    "--bind",
                    "127.0.0.1",
                    "8000",   # docs server port
                    "--directory",
                    docs_dir,
                ],
                stdout=devnull,
                stderr=devnull,
            )

        app.run_server(debug=debug_mode, host="127.0.0.1", port=8080)
    except Exception:
        if http_server_process:
            http_server_process.terminate()
        raise
    finally:
        if http_server_process:
            http_server_process.terminate()
