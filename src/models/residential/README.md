# Residential Model

### Table of Contents
- [Introduction](#introduction)
- [Prepare Data](#prepare-data)
- [Sensitivity](#sensitivity)
- [Model Overview](#model-overview)
- [Code Documentation](#code-documentation)

## Introduction
The residential model calculates updated values for energy consumption based on new input electricity prices. It takes in the new prices from a Pyomo Parameter or a dataframe and calculates the new consumption values. Base values are automatically loaded in from the input files for the calculations, but base "Load" values can be used instead if given to the module.  

It then returns a Pyomo Block instance with a "Load" variable and constraints that bind the variable to the new consumption values. A key feature of this design is that the Block instance is self-contained and modular; it can be inserted into other Pyomo model objects directly, meaning that it can be combined with other problem formulations without having to recreate each of its constituent features in the new model object (e.g., parameters, constraints). 

If running in standalone, the module will return graphs of the updated load values for the specified regions and years. This also gives an option for generating sensitivity results that will return the calculated load values and estimates for the load value if one of the input values changes. These graphs will be stored in the output directory at the top level. More information on the sensitivity method is given in the Sensitivity section below. 

## Prepare Data

The module is set to take in a Pyomo Parameter of prices that is indexed by [region, year, hour]. It will also work for dataframes indexed in the same fashion. The regions can be any subset of integers 1 to 25. The years can be any subset of integers 2023 to 2050. The hours are currently only programmed to work with seasonal data (1 to 4) or a custom subset that is valued 1 to 96. See the residential preprocessor directory for more information on how input data is prepared.  

There are several settings that are pulled from the 'src/integrator' directory. The settings are specified in the [run_config.toml](/src/common/run_config.toml) file. There is a section noted for "Residential Inputs." The following settings can be found there: 

| Setting | Values | Information |
|:------- | :----- | :---------- |
|view_years| Any subset of years from sw_year.csv | The years that you want to be graphed at the end of calculations.|
|view_reg | Any subset of region identifiers from ``regions.REGIONS`` | The regions that you want to be graphed at the end of calculations.|
|sensitivity| **true** = On <br> **false** = Off | Determines if the run will calculate Load estimates if a specified input variable is changed by a specified percent.|
|change_var | One of: <br> 'price', 'p_elas', 'income', 'i_elas', 'trendGR' | The input variable that you want to change for sensitivity estimates. The model can estimate the change in Load based on a percent change in price, price elasticity, income, income elasticity, or long-term trend.|
|percent_change| $>0$| The percent amount that the desired input variable changes. The calculation gives values for percent increase and decrease, so there is no need for negative values|
|complex| **true** = On <br> **false** = Off | This determines which derivative method to use. If it is set to true, it will use the complex-step method. If it is false, it will use the analytical derivative method. |

## Sensitivity

We include two sample sensitivity methods that use a first-order Taylor approximation to estimate the effect a change in one of the input variables will have on the output Load. Note that both of these methods are generalizable to any inputs and any outputs, but we have presented them here for Load as an example. 

**The first is an analytical method that calculates the derivative directly by using the sympy package for python.** Sympy is a computer algebra system, so functions can be written and treated symbolically before substituting in values. **The second method uses a complex-step derivative approximation.** Both find a value for the derivative and calculate an upper and lower range based on the percent change (Δ) provided as in: 



$$
\begin{aligned}
\begin{aligned}
f\left(a\pm a\cdot Δ\right)\ \approx \ f\left(a\right)\pm f^{'}\left(a\right)\left(a\cdot Δ\right)
\end{aligned}
\end{aligned}
$$



The user may choose which input variable to change and by what amount. These settings are found as described in the “Prepare Data” section. The output for both of these methods will be a graph with the Load value calculated normally and the upper and lower values associated with an increase or decrease in the desired input value. For example, the following graph shows the calculated Load for Region 7 in Year 2023, and it has the estimates for the Load if the input price increased or decreased by 10%: 

![Sensitivity](input/NewLoadSensitivity_7_2023.png "Sensitivity Graph")

## Model Overview

### Sets
|Set    | Code    | Data Type  | Short Description |
|:----- | :------ | :--------- | :---------------- |
|$\Theta_{Price}$  | price_set       | Sparse Set | $(r,y,h)$ or $(r,season,tech,step,y)$ values for given updating prices
|$\Theta_{Load}$ | load_set | Sparse Set | $(r,y,h)$ values for base loads

### Parameters
| Parameter | Code     | Data Type     | Short Description      | Index |
|:-----     | :------  | :---------    | :----------------      | :-----|
|$BaseLoad$ | loads[BaseLoads] | DataFrame | The base values for Load | $(r,y,h)$ |
|$BasePrice$ | prices[BaseSupplyPrices] or prices[BaseDualPrices}] | DataFrame | The base values for price | $(r,season,tech,step,y)\in\Theta_{Price}$ or $(r,y,h)\in\Theta_{Price}$ |
|$PriceElasticity$ | p_elas | DataFrame | Price elasticities | (r, y) |
|$Income$ | income | DataFrame | Income values | $(r,y,h)\in\Theta_{Load}$ |
$IncomeElasticity$ | i_elas | DataFrame | Income elasticities | (r,y) |
|$LastModelYear$ | LastMYr | int | Last Year in model data (2050) | |
|$BaseYear$ | baseyear | int | The year used for base values | |
|$TrendGrowthRate$ | TrendGR | DataFrame | Long Term Trend values | (r,y) |

### Variables
| Variable  | Code     | Data Type     | Short Description      | Index |
|:-----     | :------  | :---------    | :----------------      | :-----|
|$Price$ | avgPrices or newPrices| DataFrame | Updating prices | $(r,season,tech,step,y)\in\Theta_{Price}$ or $(r,y,h)\in\Theta_{Price}$ |
|$NewLoad$ | newLoad or Load| DataFrame | Values for newly calculated Load | $(r,y,h)\in\Theta_{Load}$

### Updating Function 

The updating function takes in base values from the input files. The original equations used the term "Consumption," but we will use the term "Load" to match the other modules that call upon this one.

The updating function is as follows for each region ($r$), year ($y$), and hour ($h$):



$$
\begin{aligned}
\begin{aligned}
NewLoad\\_{r,y,h} = BaseLoad\\_{r,y,h} * PriceIndex\\_{r,y,h} * IncomeIndex\\_{r,y,h} * LongTermTrend\\_{r,y}
\end{aligned}
\end{aligned}
$$



The indexes are defined as:



$$
\begin{aligned}
\begin{aligned}
PriceIndex\\_{r,y,h} = \left(\frac{Price\\_{r,y,h}}{BasePrice\\_{r,y,h}}\right)^{PriceElasticity\\_{r,y}}
\end{aligned}
\end{aligned}
$$





$$
\begin{aligned}
\begin{aligned}
IncomeIndex\\_{r,y,h} = \left(\frac{Income\\_{r,y,h}}{Income\\_{r,baseYear,h}}\right)^{IncomeElasticity\\_{r,y}}
\end{aligned}
\end{aligned}
$$





$$
\begin{aligned}
\begin{aligned}
LongTermTrend\\_{r,y} = 1 + \left(\frac{y-LastModelYear}{LastModelYear - BaseYear}\right) * \left({TrendGrowthRate\\_{r,y}+1}\right)^{LastModelYear - BaseYear}
\end{aligned}
\end{aligned}
$$



### Model Use

The purpose of this module is to calculate a value for $NewLoad$ based on the new values for $Price$. This will be used in different ways depending on what mode is being run from main.py. If the integrated modes are selected, the make_block function will be called, and pyomo blocks will be sent to whichever other module sent in prices. If the 'residential' mode is selected, then the residential model runs on its own based on the settings provided. After calculating the Load values, it will return graphs of the value. If sensitivity is selected, then it will also provide values on the graph for an increase or decrease of an input variable. These graphs can be found in the output directory at the top level. 

## Code Documentation

[Code Documentation](/docs/README.md)
