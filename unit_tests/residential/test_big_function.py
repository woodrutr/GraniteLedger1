from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from main.definitions import PROJECT_ROOT

data_source = Path(PROJECT_ROOT, 'unit_tests', 'residential', 'res_test_data', 'rates.csv')


# Some big complicated formula that is in another file...
# This would be imported for purpose of test, just putting it here
# for example...
def loan_payment(amt, rate, term):
    """calculate loan payment"""
    rate = rate / 12
    return amt * (rate * (1 + rate) ** (term * 12)) / ((1 + rate) ** (term * 12) - 1)


# make a reusable fixture (pytest dox have good documentation on this)
# this one is very silly, but shows the idea.  This fixture will read the
# data, present dataframe and can be REUSED anywhere (or imported in other tests)
@pytest.fixture()
def rate_table() -> pd.DataFrame:
    df = pd.read_csv(data_source).set_index('state')
    return df


# this is the basic design pattern of a parameterized test.  The test will be
# run for each instance of params (4x).  The 'mark.parameterize' below makes
# the connections.  Many variants in dox.

params = [
    # state, loan amt, term (yrs), expected value of payment
    ('CA', 100, 10, 1.06),
    ('CA', 100, 20, 0.66),
    ('NV', 100, 10, 465.99),  # WRONG...  will fail the 3rd of 4 tests
    ('NV', 100, 1, 8.79),
]


@pytest.mark.skip('example test')
@pytest.mark.parametrize('state, amount, term, expected', params)
def test_loan_cost(rate_table, state, amount, term, expected):
    df = rate_table

    # get the rate from the df
    rate = df['interest_rate'][state]

    # make sure we pulled something reasonable
    assert rate > 0

    # the main element of the test
    assert loan_payment(amt=amount, rate=rate, term=term) == pytest.approx(
        expected, rel=0.01
    ), 'loan payment is incorrect'
