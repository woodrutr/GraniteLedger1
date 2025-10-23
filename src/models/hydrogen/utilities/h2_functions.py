"""This file is a collection of functions that are used in support of the hydrogen model."""

import typing

# guard against circular import
if typing.TYPE_CHECKING:
    from model.h2_model import H2Model


def get_electricty_consumption(hm: 'H2Model', region, year):
    """get electricity consumption for region, year

    Parameters
    ----------
    hm : H2Model
        model
    region : str
        region
    year : int
        year

    Returns
    -------
    float
        the elecctricity consumption for a region and year in the model
    """
    return sum(
        hm.electricity_consumption_rate[tech] * hm.h2_volume[hub, tech, year]
        for tech in hm.technology
        for hub in hm.grid.registry.regions[region].hubs.keys()
    )


def get_electricity_consumption_rate(hm: 'H2Model', tech):
    """the electricity consumption rate for technology type tech

    Parameters
    ----------
    hm : H2Model
        model
    tech : str
        technology type

    Returns
    -------
    float
        GWh per kg H2
    """
    rates = {'PEM': 54.3 / 1000000, 'SMR': 5.1 / 1000000}
    return rates[tech]


def get_production_cost(hm: 'H2Model', hub, tech, year):
    """return production cost for tech at hub in year

    Parameters
    ----------
    hm : H2Model
        model
    hub : str
        hub
    tech : str
        technology type
    year : int
        year

    Returns
    -------
    float
        production cost of H2 for tech at hub in year
    """
    if hm.mode == 'standard':
        if tech == 'PEM':
            return (
                hm.electricity_price[hm.grid.registry.hubs[hub].region.name, year]
                * hm.electricity_consumption_rate[tech]
            )
        elif tech == 'SMR':
            return (
                hm.gas_price[hm.grid.registry.hubs[hub].region.name, year]
                + hm.electricity_price[hm.grid.registry.hubs[hub].region.name, year]
                * hm.electricity_consumption_rate[tech]
            )
        else:
            return 0

    elif hm.mode == 'integrated':
        if tech == 'PEM':
            return (
                hm.electricity_price[hm.grid.registry.hubs[hub].region.name, year]
                * hm.electricity_consumption_rate[tech]
            )
        elif tech == 'SMR':
            return (
                hm.gas_price[hm.grid.registry.hubs[hub].region.name, year]
                + hm.electricity_price[hm.grid.registry.hubs[hub].region.name, year]
                * hm.electricity_consumption_rate[tech]
            )
        else:
            return 0


def get_elec_price(hm: 'H2Model', region, year):
    """get electricity price in region, year

    Parameters
    ----------
    hm : H2Model
        _model
    region : str
        region
    year : int
        year

    Returns
    -------
    float
        electricity price in region and year
    """
    # TODO add year
    if hm.mode == 'standard':
        if hm.grid.registry.regions[region].data is None:
            return 0
        else:
            return hm.grid.registry.regions[region].get_data('electricity_cost')

    elif hm.mode == 'integrated':
        return hm.grid.registry.regions[region].get_data('electricity_cost')


def get_gas_price(hm: 'H2Model', region, year):
    """get gas price for region, year

    Parameters
    ----------
    hm : H2Model
        model
    region : str
        region
    year : int
        year

    Returns
    -------
    float
        gas price in region and year
    """
    # TODO add year
    if hm.grid.registry.regions[region].data is None:
        return 0

    else:
        return hm.grid.registry.regions[region].get_data('gas_cost')


def get_demand(hm: 'H2Model', region, time):
    """get demand for region at time. If mode not standard, just increase demand by 5% per year

    Parameters
    ----------
    hm : H2Model
        model
    region : str
        region
    time : int
        year

    Returns
    -------
    float
        demand
    """
    if hm.mode == 'standard':
        if hm.grid.registry.regions[region].data is None:
            return 0
        else:
            return hm.grid.registry.regions[region].get_data('demand') * 1.05 ** (
                time - hm.year.first()
            )

    elif hm.mode == 'integrated':
        return hm.demand[region, time]

    return 0
