from gui import app


def test_sidebar_sections_collapsed():
    labels = [label for label, _ in app.SIDEBAR_SECTIONS]
    assert labels == [
        'General config',
        'Demand curves',
        'Carbon policy',
        'Electricity dispatch',
        'Incentives / credits',
        'Outputs',
    ]
    assert all(expanded is False for _, expanded in app.SIDEBAR_SECTIONS)
