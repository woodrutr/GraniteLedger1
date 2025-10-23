# Electricity Model

### Table of Contents
- [Introduction](#introduction)
- [Prepare Data](#prepare-data)
- [Model Overview](#model-overview)
- [Code Documentation](#code-documentation)

## Introduction

**The electricity model is formulated as a least-cost optimization problem to meet electricity demand with generation from multiple technology options**. It includes both electricity dispatch and the option for capacity expansion. Users can also specify various power sector operations they would like to represent, for example, capabilities for representing capacity reserves, operating reserves, and ramping constraints, as well as others that will be described in more detail below. 

**The model has both temporal and spatial flexibility**, which can be specified by the user depending on the purpose of a study. The temporal flexibility includes the ability to specify both the years and the time segments within a year, with the finest granularity available being 8760 hours annually. The 8760 hours within a year can be aggregated up by hours within a day as well as days within a season. If a user is running the model for more than one year, the user can choose how non-modeled intervals between years are aggregated. In terms of spatial flexibility, users can specify which regions they want to run and if regions have the capability to trade with one another. 

**The model uses linear optimization by default, but if running the model with capacity expansion, a user has the option of representing cost reductions using a nonlinear technology learning function**. This function can be modeled endogenously, by either turning the problem into a nonlinear program, or by running successive iterations of linear programs over a fixed nonlinear learning function. Users can also specify which technology options are allowed to expand or retire. 

After completing a run, the module will return datasets in the output directory for the variables, parameters, sets, and constraints. These datasets will be stored in the output directory at the top level. The viewer within the output directory includes options for reviewing electricity model variable results. If running the model in standalone mode, the model will also produce a graphic showing the distribution of electricity prices within the output directory. 

## Prepare Data

The data needed for the electricity model is stored within the input directory in the capacity expansion model e.g., cem_input subdirectory. The inputs include regionally indexed and technology indexed input assumptions data for items like transmission and technology costs and operations. In addition, there is supply curve price and quantity data that provides the fuel cost assumptions for each power technology.  

The data is prepared within the preprocessor.py file within the scripts directory. The preprocessor script creates an initial sets class for the model, based on the setting data provided by the integrator code. Sets are organized into regional sets, temporal sets, and technology-based sets. Next the preprocessor reads in all of the input data within the cem_inputs directory and processes it into the format needed for the PowerModel based on the spatial and temporal settings specified. When the preprocessor is finished, it passes a dictionary of input data as well as the sets class to the PowerModel for further processing.  

There are several files where features can be switched (sw) on/off and different crosswalks (cw) can be selected. All of these options exist in [run_config.toml](/src/common/run_config.toml) file the integrator directory. 

### Feature Settings

The [run_config.toml](/src/common/run_config.toml) file contains the main switches through which features for the electricity module can be toggled. The setup column names various constraint settings: 

|Switch    | Description   | Values | Notes |
|:----- | :------ | :--------- | :---: |
|sw_trade | Interregional trade | **0** = Off <br> **1** = On | |
|sw_expansion | Capacity expansion/retirement | **0** = Off <br> **1** = On | Note the file cem_inputs/AllowBuilds.csv also contains settings of which technologies are available to expand. cem_inputs/AllowRet.csv contains which technologies have the option to economically retire. |
|sw_agg_year | Aggregate years | **0** = Only runs sw_year <br> **1** = Aggregates all unselected years into subsequent selected year | Switch to aggregate years based on the selected years in sw_year.csv|
|sw_rm | Reserve margin requirement | **0** = Off <br> **1** = On | |
|sw_ramp | Maximum ramping constaint | **0** = Off <br> **1** = On | |
|sw_reserves | Operating reserve requirement | **0** = Off <br> **1** = On ||
|sw_learning | Technology cost learning | **0** = Exogenous learning <br> **1** = Iterative linear learning <br> **2** = Nonlinear learning | The method of which technology costs decrease as more capacity is built. Note this switch does nothing unless sw_expansion=1 |

### Technology Settings

The model contains 14 technologies (tech) in its initial layout. Users could change the technology assignments and add more technology types or remove technology types, but any changes to the code would require updates to the cooresponding input data. The technologies represented include:
<br> 1.)	Coal Steam 
<br> 2.)	Oil Steam
<br> 3.)	Natural Gas Single-Cycle Combustion Turbine
<br> 4.)	Natural Gas Combined-Cycle
<br> 5.)	Hydrogen Turbine
<br> 6.)	Nuclear
<br> 7.)	Biomass
<br> 8.)	Geothermal
<br> 9.)	Municipal-Solid-Waste
<br> 10.)	Hydroelectric Generation
<br> 11.)	Pumped Hydroelectric Storage
<br> 12.)	Battery Energy Storage
<br> 13.)	Wind, Offshore
<br> 14.)	Wind, Onshore
<br> 15.)	Solar  (step 1 = utility-scale; step 2 = end-use)

The technologies (tech) are also combined into group based on the applicability of different constraints. These groups are defined in tech_subsets.csv within the electricity/input directory and includes: 
* T_conv: conventional 
* T_re: renewable energy
* T_hydro: hydroelectric 
* T_stor: storage 
* T_vre: variable renewable energy
* T_wind: wind 
* T_solar: solar 
* T_h2: hydrogen 
* T_disp: dispatchable 
* T_gen: generating

When the capacity expansion switch is turned on, a user can select which technologies they want to have expansion and retirement capabilities. Turning these switches on allows for builds and/or retirements of a given technology and supply curve step. These files are located in the electricity/input directory.

|Switch    | Description   | Values | Notes |
|:----- | :------: | :--------- | :---: |
|sw_builds | Contains switches for technologies and supply curve steps where capacity is allowed to build | **0** = Not Allowed to Build <br> **1** = Allowed to Build | Switches contained in Sw_ptbuilds.csv |
|sw_retires | Contains switches for technologies and supply curve steps where capacity is allowed to retire | **0** = Not Allowed to Retire <br> **1** = Allowed to Retire | Switches contained in Sw_ptbuilds.csv |


## Model Overview

### Sets
|Set    | Code    | Data Type  | Short Description |
|:----- | :------ | :--------- | :---------------- |
|$$H$$ | hour | Set | All representative hours|
|$$Y$$ | year | Sparse set | All selected model years|
|$$SEA$$ | season | Set | All seasons|
|$$D$$ | day | Set | All representative days|
|$$R$$ | region | Set | All selected model domestic regions|
|$$R^{int}$$ | region_int | Set | All selected model international regions|
|$$\Theta_{load}$$ |demand_balance_index | Sparse set | All load sparse set|
|$$\Theta_{gen}$$ |generation_total_index | Sparse set | All non-storage generation sparse set|
|$$\Theta_{H2gen}$$ |H2GenSet | Sparse set | All hydrogen generation sparse set|
|$$\Theta_{stor}$$ | StorageSet | Sparse set | All storage set|
|$$\Theta_{um}$$ |unmet_load_index | Sparse Set | All unmet load set|
|$$\Theta_{SC}$$ | capacity_total_index| Sparse Set | Existing capacity set|
|$$\Theta_{dt^{max}}$$ | generation_dispatchable_ub_index| Sparse Set | Dispatchable technology generation upper bound set|
|$$\Theta_{it^{max}}$$ | generation_vre_ub_index| Sparse Set | Intermittent technology generation upper bound set|
|$$\Theta_{ht^{max}}$$ | generation_hydro_ub_index | Sparse Set | Hydroelectric generation upper bound set|
|$$\Theta_{hs}$$ | capacity_hydro_ub_index | Sparse Set | Hydroelectric generation seasonal upper bound set|
|$$\Theta_{ret}$$ | capacity_retirements_index| Sparse Set | Retirable capacity set|
|$$\Theta_{new}$$ | BuildSet| Sparse Set | Buildable capacity set|
|$$\Theta_{cc}$$ | capacity_builds_index| Sparse Set | Set of  capacity costs |
|$$\Theta_{cc0}$$ | CapCostInitial_index| Sparse Set | Set of initial year's capacity costs |
|$$\Theta_{SBFH}$$ |storage_first_hour_balance_index | Sparse set | First hour storage balance set|
|$$\Theta_{SBH}$$ | storage_most_hours_balance_index | Sparse set | (non-first hour) storage balance set|
|$$\Theta_{proc}$$ | reserves_procurement_index | Sparse set | Set for procurement of operating reserves |
|$$\Theta_{ramp}$$ | generation_ramp_index | Sparse set | Set for ramping |
|$$\Theta_{ramp1}$$ | ramp_first_hour_balance_index | Sparse set | Set for ramping in first hour of each representative day |
|$$\Theta_{ramp23}$$ | ramp_most_hours_balance_index | Sparse set | Set for ramping in non-first hour of each representative day |
|$$\Theta_{tra}$$ | trade_interregional_index | Sparse set | Domestic interregional trade set |
|$$\Theta_{tracan}$$ | trade_interational_index | Sparse set | International interregional trade set |
|$$\Theta_{traL^{int}}$$ | TranLimitInt_index | Sparse set | International interregional trade limit set |
|$$\Theta_{traLL}$$ | TranLimit_index | Sparse set | Domestic interregional trade limit set |
|$$\Theta_{traLL^{int}}$$ | TranLineLimitInt_index | Sparse set | International interregional trade line limit set |

### Re-Indexed Sets
These sets are re-indexed for specific constraints. They are all sub-sets accessed by certain indicies to return the remaining indicies.
|Set    | Code    | Data Type  | Short Description |
|:----- | :------ | :--------- | :---------------- |
|$$\theta^{H2H}_h$$ | H2GenSetByHour | Sparse subset | Set for H2 generation indexed by hour |
|$$\theta^{GSH}_h$$ | GenSetByHour | Sparse subset | Set for generation indexed by hour |
|$$\theta^{SSH}_h$$ | StorageSetByHour | Sparse subset | Set for storage indexed by hour |
|$$\theta^{GDB}_{y,r,h}$$ | GenSetDemandBalance | Sparse subset | Set for generation indexed by y,r,h |
|$$\theta^{SDB}_{y,r,h}$$ | StorageSetDemandBalance | Sparse subset | Set for storage indexed by y,r,h |
|$$\theta^{TDB}_{y,r,h}$$ | TradeSetDemandBalance | Sparse subset | Set for trade indexed by y,r,h |
|$$\theta^{TCDB}_{y,r,h}$$ | TradeCanSetDemandBalance| Sparse subset | Set for international trade indexed by y,r,h |
|$$\theta^{windor}_{y,r,h}$$ |WindSetReserves| Sparse subset | Set for wind generaton for operational reserves indexed by y,r,h |
|$$\theta^{solor}_{y,r,h}$$ | SolarSetReserves| Sparse subset | Set for solar capacity for operational reserves indexed by y,r,h |
|$$\theta^{opres}_{y,r,h}$$ | ProcurementSetReserves| Sparse subset | Set for procurement of operating reserves for operational reserves indexed by y,r,h |
|$$\theta^{scrm}_{y,r,seas}$$ | SupplyCurveRM| Sparse subset | Set for supply curve for reserve margin indexed by y,r,seas |
|$$\theta^{HSH}_{seas}$$ | HourSHydro| Sparse subset | Set for hours indexed by season |

### Parameters
Note: the existing code shows cost units in MW/MWh instead of GW/GWh; we are aware and just haven't updated the code yet.
| Parameter | Code     | Domain     | Short Description      | Units |
|:-----     | :------  | :---------    | :----------------      | :-----|
|$$YR0$$ | y0 | $$\mathbb{I}$$ | First year of model | unitless |
|$$N$$ | num_hr_day | $$\mathbb{I}$$ | Number of representative hours in a representative day | unitless |
|$$LOAD_{r,y,h}$$ | Load | $$\mathbb{R}^+_0$$ | Electricity demand | instantaneous GW |
|$$CAP^{exist}_{r,seas,t,s,y}$$ | SupplyCurve | $$\mathbb{R}^+_0$$ | Existing capacity (prescribed or initial) | GW |
|$$SPR_{r,seas,t,s,y}$$ | SupplyPrice | $$\mathbb{R}^+_0$$ | Fuel + variable O&M price | $/GWh |
|$$ICF_{t,y,r,s,h}$$ | CapFactorVRE | $$\mathbb{R}^+_0$$ | Intermittent technology maximum capacity factor | fraction |
|$$HCF_{t,y,r,s,h}$$ | HydroCapFactor | $$\mathbb{R}^+_0$$ | Hydroelectric technology maximum capacity factor | fraction |
|$$STORLC$$ | StorageLevelCost |$$\mathbb{R}^+_0$$  | Cost to hold storage (mimics losses) | $/GWh |
|$$EFF_t$$ | StorageEfficiency | $$\mathbb{R}^+_0$$ | Roundtrip efficiency of storage | fraction |
|$$STOR^{dur}_t$$ | HourstoBuy | $$\mathbb{R}^+_0$$ | Storage duration | hours |
|$$UMLPEN$$ | UnmetLoadPenalty | $$\mathbb{R}^+_0$$ | Unmet load penalty | $/GWh |
|$$WY_y$$ | WeightYear | $$\mathbb{I}$$ | number of years represented by a representative year (weight) | years/representative years |
|$$HW_h$$ | WeightHour | $$\mathbb{I}$$ | number of hours represented by a representative hours(weight) | hours/representative hours |
|$$WeightDay_d$$ | WeightDay | $$\mathbb{I}$$ | number of days representated by a representative day (weight) | days/representative day |
|$$MHD_h$$ | MapHourDay | $$\mathbb{I}$$ | map representative hour to representative day | unitless |
|$$WHS_{seas}$$ | WeightSeason | $$\mathbb{I}$$ | number of hours (per year) in a season (weight) | unitless |
|$$MHS_h$$ | MapHourSeason |$$\mathbb{I}$$  | map representative hour to season | unitless |
|$$FOMC_{r,t,s}$$ | FOMCost | $$\mathbb{R}^+_0$$ | Fixed O&M cost | $/GW-year |
|$$CC_{t,y,r,s,h}$$ | CapacityCredit | $$\mathbb{R}^+_0$$ | Capacity credit | fraction |
|$$RM_r$$ | ReserveMargin | $$\mathbb{R}^+_0$$ | Reserve margin requirement | fraction |
|$$RUC_{t}$$ | RampUpCost | $$\mathbb{R}^+_0$$ | Ramp up cost | $/GW |
|$$RDC_{t}$$ | RampDownCost | $$\mathbb{R}^+_0$$ | Ramp down cost | $/GW |
|$$RR_t$$ | RampRate | $$\mathbb{R}^+_0$$ | Max ramp rate | GW |
|$$TRALINLIM_{r,r1,seas,y}$$ | TranLimit | $$\mathbb{R}^+_0$$ | Domestic interregional trade line limit | GW |
|$$TRALIM^{int}_{r^{int},c,y,h}$$ | TranLimitGenInt | $$\mathbb{R}^+_0$$ |  International interregional trade limit | GW |
|$$TRALINLIM^{int}_{r,r^{int},y,h}$$ | TranLimitCapInt | $$\mathbb{R}^+_0$$ | International interregional trade line limit | GW |
|$$TRAC_{r,r1,y}$$ | TranCost | $$\mathbb{R}^+_0$$ | Transmission hurdle rate (cost) | $/GWh |
|$$TRACC_{r,r^{int},c,y}$$ | TranCostInt | $$\mathbb{R}^+_0$$ | International transmission hurdle rate (cost) | $/GWh |
|$$LL$$ | TransLoss | $$\mathbb{R}^+_0$$ | Transmission line losses from 1 region to another  | fraction |
|$$OPRP_t$$ | RegReservesCost | $$\mathbb{R}^+_0$$ | Cost of operating reserve procurement (TODO: update this in code so it contains all optypes) | $/GWh |
|$$RTUB_{o,t}$$ | ResTechUpperBound | $$\mathbb{R}^+_0$$ | Maximum amount of capacity which can be used to procure operating reserves | fraction |
|$$H2HR$$ | H2Heatrate | $$\mathbb{R}^+_0$$ | Hydrogen heatrate | kg/GWh |
|$$H2PR_{r,seas,t,s,y} $$ | H2Price | $$\mathbb{R}^+_0$$ | Hydrogen fuel price. Mutable parameter. | $/kg |
|$$CAPCL_{r,t,y,s} $$ | CapCostLearning | $$\mathbb{R}^+_0$$ | Cost of capacity based on technology learning. Mutable parameter. | $/GW |
|$$CAPC0_{r,t,s} $$ |CapCostInitial | $$\mathbb{R}^+_0$$ | Initial year's capacity cost to build | $/GW |
|$$LR_t $$ | LearningRate | $$\mathbb{R}^+_0$$ | Learning rate factor | unitless |
|$$SCL_t $$ | SupplyCurveLearning | $$\mathbb{R}^+_0$$ | Learning rate factor | unitless |

### Variables
| Variable  | Code     | Domain     | Short Description      | Units | Switch notes |
|:-----     | :------  | :---------    | :----------------      | :-----| :---------|
|$$STOR^{in}_{t,y,r,s,h}$$ | storage_inflow | $$\mathbb{R}^+_0$$ | Storage inflow | GW | |
|$$STOR^{out}_{t,y,r,s,h}$$ | storage_outflow |$$\mathbb{R}^+_0$$  | Storage outflow | GW | |
|$$STOR^{level}_{t,y,r,s,h}$$ | storage_level |$$\mathbb{R}^+_0$$  | Storage level (state-of-charge) | GWh | |
|$$GEN_{t,y,r,s,h}$$ | generation_total | $$\mathbb{R}^+_0$$ | Instantaneous generation | GW | |
|$$UNLOAD_{r,y,h}$$ | unmet_load | $$\mathbb{R}^+_0$$ | Unmet load | GW | |
|$$CAP^{tot}_{r,seas,t,s,y}$$ | capacity_total | $$\mathbb{R}^+_0$$  | Total capacity | GW | |
|$$CAP^{new}_{r,t,y,s}$$ | capacity_builds | $$\mathbb{R}^+_0$$ | New capacity built | GW | Only created if sw_expansion=1 |
|$$CAP^{ret}_{t,y,r,s}$$ | capacity_retirements | $$\mathbb{R}^+_0$$ | Retirement capacity | GW | Only created if sw_expansion=1|
|$$TRA_{r,r1,y,h}$$ | trade_interregional | $$\mathbb{R}^+_0$$ | Interregional trade from region $r1$ to region $r$ | GW | Only created if sw_trade=1 |
|$$TRA^{int}_{r,r^{int},y,c,h}$$ | trade_interational | $$\mathbb{R}^+_0$$ | International interregional trade from region $r^{int}$ to region $r$ | GW | Only created if sw_trade=1 |
|$$RAMP^{up}_{t,y,r,s,h}$$ | generation_ramp_up | $$\mathbb{R}^+_0$$  | Ramp up (increase in generation for dispatchable cap) | GW | Only created if sw_ramp=1 |
|$$RAMP^{down}_{t,y,r,s,h}$$ | generation_ramp_down  | $$\mathbb{R}^+_0$$ | Ramp down (decrease in generation for dispatchable cap) | GW | Only created if sw_ramp=1 |
|$$ORP_{o,t,y,r,s,h}$$ | reserves_procurement | $$\mathbb{R}^+_0$$ | Operating reserves procurement amount | GW | Only created if sw_reserves=1 |
|$$STOR^{avail}_{t,y,r,s,h}R$$ | storage_avail_cap | $$\mathbb{R}^+_0$$ | Available storage capacity to meet the reserve margin | GW | Only created if sw_rm=1 |


### Objective Function

Objective is to minimize costs to the electric power system. Costs include dispatch cost (e.g., variable O&M cost), fixed operation and maintenance (FOM) cost, capacity expansion cost component (nonlinear and linear options available), interregional trade cost, ramping cost, operating reserve cost and unmet load cost (note: unmet load cost should equal zero). 

Minimize total cost ($)



$$
\begin{aligned}
\begin{aligned}
        \min \mathbf{C\\_{tot}} =  C\\_{disp}+ C\\_{unload} \\
        (+ C\\_{exp} + C\\_{fom} \quad if \quad sw\\\_expansion = 1 )\\ 
        (+ C\\_{tra} \quad if \quad sw\\\_trade = 1 )\\
        (+ C\\_{ramp} \quad if \quad sw\\\_ramp = 1 )\\
        (+ C\\_{or}\quad if \quad sw\\\_reserves = 1 )
\end{aligned}
\end{aligned}
$$



where:

Dispatch cost: 


$$
\begin{aligned}
\begin{aligned}
C\\_{disp} = 
        \sum\\_{h \in H \\| s=MHS\\_h}{}
        (WD\\_h \times 
        \sum\\_{{t,y,r,s} \in \theta^{GSH}\\_h}{WY\\_y \times SPR\\_{r,seas,t,s,y} \times \mathbf{GEN}\\_{t,y,r,s,h}}\\
        +\sum\\_{{t,y,r,s} \in \theta^{SSH}\\_h}{(WY\\_y \times (0.5 \times SPR\\_{r,seas,t,s,y} \times (\mathbf{STOR^{in}}\\_{t,y,r,s,h} + \mathbf{STOR^{out}}\\_{t,y,r,s,h})}\\
        + (HW\\_h \times STORLC) \times \mathbf{STOR^{level}}\\_{t,y,r,s,h}))\\
        +\sum\\_{{t,y,r,s} \in \theta^{H2SH}\\_h}{WY\\_y \times H2PR\\_{r,seas,t,s,y} \times H2HR \times \mathbf{GEN}\\_{t,y,r,1,h}}) 
\end{aligned}
\end{aligned}
$$



Unmet load cost:


$$
\begin{aligned}
\begin{aligned}
        C\\_{unload} = 
        \sum\\_{{r,y,h} \in \Theta\\_{um}}{
        WD\\_h \times 
        WY\\_y \times UMLPEN \times \mathbf{UNLOAD}\\_{r,y,h}}
\end{aligned}
\end{aligned}
$$



Capacity expansion cost: 


$$
\begin{aligned}
\begin{aligned}
C\\_{exp} = 
        \sum\\_{{r,t,y,s} \in \Theta\\_{cc}}
       ( CAPC0\\_{r,t,y,s} 
       \\
       \times \left( \frac{
            SCL\\_t + 0.001 \times (y-YR0) 
            + \sum\\_{{r,t1,s} \in \Theta\\_{cc0} \\| t1 = t}{ \sum\\_{y1 \in Y \\| y1 \ \<  y}{\mathbf{CAP^{new}}\\_{r,t1,y1,s}}}
            }{SCL\\_t} \right) ^{-LR\\_t} 
            \\
            \times \mathbf{CAP^{new}}\\_{r,t,y,s} )
         \\
        \quad if \quad sw\\\_learning = 2 
\end{aligned}
\end{aligned}
$$


<br />



$$
\begin{aligned}
\begin{aligned}
        C\\_{exp} = 
        \sum\\_{{r,t,y,s} \in \Theta\\_{cc}}{
       CAPCL\\_{r,t,y,s} \times \mathbf{CAP^{new}}\\_{r,t,y,s}} \\
       \quad if \quad sw\\\_learning  \ \<   2 
\end{aligned}
\end{aligned}
$$



Fixed O\&M cost:


$$
\begin{aligned}
\begin{aligned}
        C\\_{fom} =
        \sum\\_{{r,seas,t,s,y} \in \Theta\\_{sc} \\| seas=2}{
        WY\\_y \times FOMC\\_{r,t,s} \times \mathbf{CAP^{tot}}\\_{r,seas,t,s,y}}
\end{aligned}
\end{aligned}
$$



Interregional trade cost:


$$
\begin{aligned}
\begin{aligned}
        C\\_{tra} =
        \sum\\_{{r,r1,y,h} \in \Theta\\_{tra}}{
        WD\\_h \times WY\\_y \times TRAC\\_{r,r1,y} \times \mathbf{TRA}\\_{r,r1,y,h}}\\
        +
        \sum\\_{{r,r^{int},y,c,h} \in \Theta\\_{tracan}}{WD\\_h \times WY\\_y \times TRACC\\_{r,r^{int},c,y} \times 
        \mathbf{TRA^{int}\\_{r,r^{int},y,c,h}}}
\end{aligned}
\end{aligned}
$$



Ramping cost:


$$
\begin{aligned}
\begin{aligned}
        C\\_{ramp} =
        \sum\\_{{t,y,r,s,h} \in \Theta\\_{ramp}}{
        WD\\_h \times WY\\_y \times 
        (RUC\\_t \times \mathbf{RAMP^{up}}\\_{t,y,r,s,h}
        + RDC\\_t \times \mathbf{RAMP^{up}}\\_{t,y,r,s,h})}
\end{aligned}
\end{aligned}
$$



Operating reserve cost:


$$
\begin{aligned}
\begin{aligned}
    C\\_{op} =
        \sum\\_{{o,t,y,r,s,h} \in \Theta\\_{orp}}{
        WD\\_h \times WY\\_y \times 
        ORC\\_t \times
        \mathbf{ORP}\\_{o,t,y,r,s,h}
        }
\end{aligned}
\end{aligned}
$$



### Constraints

#### Balance Constraints
Balance constraints exist for generation as well as energy storage. For demand, this means that generation must equal to or exceed demand for electricity.  

For energy storage technologies, the balance constraints ensure that the storage level in the current time segment is equal to the storage level in the previous time-segment plus any storage charge and/or discharge (while also accounting for round-trip efficiency losses). 

Demand balance constraint:


$$
\begin{aligned}
\begin{aligned}
    LOAD\\_{r,y,h} \leq \sum\\_{{t,s} \in \theta^{GDB}\\_{y,r,h}}{\mathbf{GEN}\\_{t,y,r,s,h}}\\
    + \sum\\_{{t,s} \in \theta^{SDB}\\_{y,r,h}}{(\mathbf{STOR^{out}}\\_{t,y,r,s,h} 
    - \mathbf{STOR^{in}}\\_{t,y,r,s,h})}\\
        + \mathbf{UNLOAD}\\_{r,y,h}\\
        (+ \sum\\_{r1 \in \theta^{TDB}\\_{y,r,h}}{\left(\mathbf{TRA}\\_{r,r1,y,h} \times (1 - LL) - \mathbf{TRA}\\_{r1,r,y,h}\right)} 
        \quad if \quad sw\\\_trade = 1)\\
    (+ \sum\\_{r\\_{int},c \in \theta^{TCDB}\\_{y,r,h}}{(\mathbf{TRA}^{int}\\_{r,r\\_{int},y,c,h} 
    \times (1 - LL) - \mathbf{TRA}^{int}\\_{r\\_{int},r,y,c,h})} 
    \quad if \quad sw\\\_trade = 1)\\
        \forall  {r,y,h} \in \Theta\\_{load}
\end{aligned}
\end{aligned}
$$



First hour storage balance constraint:


$$
\begin{aligned}
\begin{aligned}
        \mathbf{STOR^{level}}\\_{t,y,r,s,h} = 
        \mathbf{STOR^{level}}\\_{t,y,r,s,h+N - 1}\\
        + EFF\\_t \times \mathbf{STOR^{in}}\\_{t,y,r,s,h} - \mathbf{STOR^{out}}\\_{t,y,r,s,h}\\
        \forall {t,y,r,s,h} \in \Theta\\_{SBFH}
\end{aligned}
\end{aligned}
$$




Storage balance (not first hour) constraint:


$$
\begin{aligned}
\begin{aligned}
        \mathbf{STOR^{level}}\\_{t,y,r,s,h} = 
        \mathbf{STOR^{level}}\\_{t,y,r,s,h - 1}\\
        + EFF\\_t \times \mathbf{STOR^{in}}\\_{t,y,r,s,h} - \mathbf{STOR^{out}}\\_{t,y,r,s,h}\\
        \forall {t,y,r,s,h} \in \Theta\\_{SBH}
\end{aligned}
\end{aligned}
$$



#### Generation Upper Bounds

Generation upper bound constraints limit generation from generating technologies, accounting for reserve requirements, operating capacity, and capacity factors where: 

$$ Generation + Reserve Procurement <= Capacity \times Capacity Factor $$

This is the same constraint for dispatchable, hydroelectric, and intermittent technologies. For intermittent technologies, the capacity factors are exogenously specified in the input data. In addition, hydroelectric generation has an additional seasonal constraint, where hydroelectric capacity is seasonally limited based on assumed availability of water resources seasonally, as specified in the input data. Storage upper bound constraints need to account for the upper bounds on both the charge and discharge of the technology, as well as the operating level in any given time segment. 

Hydroelectric generation seasonal upper bound:


$$
\begin{aligned}
\begin{aligned}
        \sum\\_{h \in \theta^{HSH}\\_{seas}}{\mathbf{GEN}\\_{t,y,r,1,h} \times WeightDay\\_{MHD\\_{h}}} \leq \mathbf{CAP^{tot}}\\_{r,seas,t,1,y} \times HCF\\_{r,seas}
        \times WHS\\_{seas}\\
            \forall {t,y,r,seas} \in \Theta\\_{hs}
\end{aligned}
\end{aligned}
$$




Dispatchable technology generation upper bound:


$$
\begin{aligned}
\begin{aligned}
        \mathbf{GEN}\\_{t,y,r,s,h} \\
        (+ \sum\\_{rt \in RT}{\mathbf{OPRP}\\_{rt,t,y,r,s,h}} 
        \quad if \quad sw\\\_rm = 1)\\
        \leq \mathbf{CAP^{tot}}\\_{r,MHS\\_h,t,s,y} \times HW\\_h\\
        \forall {t,y,r,s,h} \in \Theta\\_{dt^{max}}
\end{aligned}
\end{aligned}
$$




Hydroelectric technology generation upper bound:


$$
\begin{aligned}
\begin{aligned}
        \mathbf{GEN}\\_{t,y,r,s,h} \\
        (+ \sum\\_{rt \in RT}{\mathbf{OPRP}\\_{rt,t,y,r,s,h}}
        \quad if \quad sw\\\_rm = 1)\\
        \leq \mathbf{CAP^{tot}}\\_{r,MHS\\_h,t,s,y} \times HCF\\_{r,MHS\\_h} \times HW\\_h\\
        \forall {t,y,r,s,h} \in \Theta\\_{ht^{max}}
\end{aligned}
\end{aligned}
$$




Intermittent technology upper bound:


$$
\begin{aligned}
\begin{aligned}
        \mathbf{GEN}\\_{t,y,r,s,h} \\
        (+ \sum\\_{rt \in RT}{\mathbf{OPRP}\\_{rt,t,y,r,s,h}}
        \quad if \quad sw\\\_rm = 1)\\
        \leq \mathbf{CAP^{tot}}\\_{r,MHS\\_h,t,s,y} \times ICF\\_{t,y,r,s,h} \times HW\\_h\\
        \forall {t,y,r,s,h} \in \Theta\\_{it^{max}}
\end{aligned}
\end{aligned}
$$




Storage technology inflow upper bound:


$$
\begin{aligned}
\begin{aligned}
        \mathbf{STOR^{in}}\\_{t,y,r,s,h} + 
        \leq \mathbf{CAP^{tot}}\\_{r,MHS\\_h,t,s,y} \times HW\\_h\\
        \forall {t,y,r,s,h} \in \Theta\\_{stor}
\end{aligned}
\end{aligned}
$$



Storage technology outflow upper bound:


$$
\begin{aligned}
\begin{aligned}
        \mathbf{STOR^{out}}\\_{t,y,r,s,h} \\
        (+\sum\\_{rt \in RT}{\mathbf{OPRP}\\_{rt,t,y,r,s,h}}
        \quad if \quad sw\\\_rm = 1)\\
        \leq \mathbf{CAP^{tot}}\\_{r,MHS\\_h,t,s,y} \times HW\\_h\\
        \forall {t,y,r,s,h} \in \Theta\\_{stor}
\end{aligned}
\end{aligned}
$$




Storage technology level upper bound:


$$
\begin{aligned}
\begin{aligned}
        \mathbf{STOR^{level}}\\_{t,y,r,s,h} 
        \leq \mathbf{CAP^{tot}}\\_{r,MHS\\_h,t,s,y} \times STOR^{dur}\\_t\\
        \forall {t,y,r,s,h} \in \Theta\\_{stor}
\end{aligned}
\end{aligned}
$$



#### Capacity Expansion

The model can build new generating technologies each year (when the expansion switch is turned on). The expansion constraint ensures that operating capacity is based on the capacity in the previous year, plus any additions and minus any retirements. The retirement constraint ensures that retirements never exceed the capacity available on the system.

Total capacity balance:


$$
\begin{aligned}
\begin{aligned}
        \mathbf{CAP^{tot}}\\_{r,seas,t,s,y}
        = CAP^{exist}\\_{r,seas,t,s,y} \\
        (+ \sum\\_{cy \in Y \leq y}{\mathbf{CAP^{new}}\\_{r,t,cy,s}} 
        \quad if \quad sw\\\_expansion = 1)\\
        (+ \sum\\_{cy \in Y \leq y}{\mathbf{CAP^{ret}}\\_{t,cy,r,s}} 
        \quad if \quad sw\\\_expansion = 1)\\
        \forall {r,seas,t,s,y} \in \Theta\\_{SC}
\end{aligned}
\end{aligned}
$$




Capacity retirement upper bound:


$$
\begin{aligned}
\begin{aligned}
        \mathbf{CAP^{ret}}\\_{t,y,r,s} \leq
        CAP^{exist}\\_{r,2,t,s,y} +
        \sum\\_{cy \in Y  \ \<   y}{\mathbf{CAP^{new}}\\_{r,t,cy,s}} -
        \sum\\_{cy \in Y  \ \<   y}{\mathbf{CAP^{ret}}\\_{t,cy,r,s}} \\
        \forall {t,y,r,s} \in \Theta\\_{ret} \\
        \quad if \quad sw\\\_expansion = 1 \\
\end{aligned}
\end{aligned}
$$




#### Trade
Electricity trade constraints ensure that trade within any given time segment cannot exceed the capabilities of the transmission lines between the regions trading. In addition, there are supply quantity/price constraints for international trade, where supply from international regions cannot exceed the availability from the region.  

International interregional trade line capacity upper bound:


$$
\begin{aligned}
\begin{aligned}
        \sum\\_{c}{\mathbf{TRA^{int}}\\_{r,r^{int},y,c,h}} \leq
        TRALINLIM^{int}\\_{r,r^{int},y,h} * HW\\_h \\
        \forall {r,r^{int},y,h} \in \Theta\\_{traLL^{int}} \\
        \quad if \quad sw\\\_trade = 1\\
\end{aligned}
\end{aligned}
$$



International interregional trade resource capacity upper bound:


$$
\begin{aligned}
\begin{aligned}
        \sum\\_{r}{\mathbf{TRA^{int}}\\_{r,r^{int},y,c,h}} \leq
        TRALIM^{int}\\_{r^{int},c,y,h} * HW\\_h \\
        \forall {r,r^{int},y,h} \in \Theta\\_{traL^{int}} \\
        \quad if \quad sw\\\_trade = 1\\
\end{aligned}
\end{aligned}
$$




Domestic interregional trade line capacity upper bound:


$$
\begin{aligned}
\begin{aligned}
        \mathbf{TRA}\\_{r,r1,y,h} \leq
        TRALINLIM\\_{r,r1,MHS\\_h,y} * HW\\_h \\
        \forall {r,r1,y,h} \in \Theta\\_{traLL} \\
        \quad if \quad sw\\\_trade = 1\\
\end{aligned}
\end{aligned}
$$



#### Reserve Margin
Reserve margin constraints ensure that there is additional quantity of capacity available beyond load requirements in each time segment. Available capacity that can contribute to the reserve margin is also potentially decremented based on capacity credit assumptions. Storage technologies have additional reserve margin constraints accounts for both the power capacity and the energy capacity availability towards contributing to reserve margin requirements.  

Reserve margin requirement constraint:


$$
\begin{aligned}
\begin{aligned}
        LOAD\\_{r,y,h} \times
        (1 + RM\\_r ) \leq
        HW\\_h \times \\
        \sum\\_{{t,s} \in \theta^{scrm}\\_{y,r,MHS\\_h}}{CC\\_{t,y,r,s,h} \times (\mathbf{STOR^{avail}}\\_{t,y,r,s,h} + \mathbf{CAP^{tot}\\_{r,MHS\\_h,t,s,y}})}\\
        \forall {r,y,h} \in \Theta\\_{load}\\
        \quad if \quad sw\\\_rm = 1\\
\end{aligned}
\end{aligned}
$$




Constraint to ensure available storage capacity to meet RM <= power cap, upper bound:


$$
\begin{aligned}
\begin{aligned}
    \mathbf{STOR^{avail}}\\_{t,y,r,s,h} \leq    \mathbf{CAP^{tot}}\\_{r,MHS\\_h,t,s,y}\\
        \forall {t,y,r,s,h} \in \Theta\\_{stor}\\
        \quad if \quad sw\\\_rm = 1\\
\end{aligned}
\end{aligned}
$$




Constraint to ensure available storage capacity to meet RM <= existing storage level, upper bound:


$$
\begin{aligned}
\begin{aligned}
    \mathbf{STOR^{avail}}\\_{t,y,r,s,h} \leq    
    \mathbf{STOR^{level}}\\_{t,y,r,s,h}\\
        \forall {t,y,r,s,h} \in \Theta\\_{stor}\\
        \quad if \quad sw\\\_rm = 1\\
\end{aligned}
\end{aligned}
$$



#### Ramping
Ramping constraints ensure that generating technologies are limited in the rate in which they can increase or decrease their generation from one time segment to the next. Ramping capabilities are balanced within each day.

First hour ramping balance constraint:


$$
\begin{aligned}
\begin{aligned}
    \mathbf{GEN}\\_{t,y,r,s,h} =    \mathbf{GEN}\\_{t,y,r,s,h+N-1} + \mathbf{RAMP^{up}}\\_{t,y,r,s,h} - \mathbf{RAMP^{down}}\\_{t,y,r,s,h}\\
    \forall {t,y,r,s,h} \in \Theta\\_{ramp1} \\
        \quad if \quad sw\\\_ramp = 1\\
\end{aligned}
\end{aligned}
$$



Ramping balance (not first hour) constraint:


$$
\begin{aligned}
\begin{aligned}
    \mathbf{GEN}\\_{t,y,r,s,h} =    \mathbf{GEN}\\_{t,y,r,s,h-1} + \mathbf{RAMP^{up}}\\_{t,y,r,s,h} - \mathbf{RAMP^{down}}\\_{t,y,r,s,h}\\
    \forall {t,y,r,s,h} \in \Theta\\_{ramp23} \\
        \quad if \quad sw\\\_ramp = 1\\
\end{aligned}
\end{aligned}
$$




Ramp up upper bound:


$$
\begin{aligned}
\begin{aligned}
    \mathbf{RAMP^{up}}\\_{t,y,r,s,h} \leq   
    HW\\_h \times RR\\_t \times
    \mathbf{CAP^{tot}}\\_{r,MHS\\_h,t,s,y}\\
    \forall {t,y,r,s,h} \in \Theta\\_{ramp} \\
        \quad if \quad sw\\\_ramp = 1\\
\end{aligned}
\end{aligned}
$$




Ramp down upper bound:


$$
\begin{aligned}
\begin{aligned}
    \mathbf{RAMP^{down}}\\_{t,y,r,s,h} \leq   
    HW\\_h \times RR\\_t \times
    \mathbf{CAP^{tot}}\\_{r,MHS\\_h,t,s,y}\\
    \forall {t,y,r,s,h} \in \Theta\\_{ramp} \\
        \quad if \quad sw\\\_ramp = 1\\
\end{aligned}
\end{aligned}
$$



#### Operating Reserves
The model allows for three different types of operating reserves to be represented within the model, either spinning reserves, regulation reserves, or flexibility reserve requirements. These operating reserves reflect the need to additional capacity to be held in reserve to meet and short-term needs for generation based on un-expected changes in things like electricity demand or variable renewable generation output.  

Spinning reserve requirement constraint. 3\% of load required:


$$
\begin{aligned}
\begin{aligned}
    0.03 \times LOAD\\_{r,y,h} \leq 
    \sum\\_{{t,s} \in \theta^{opres}\\_{1,r,y,h}}{\mathbf{ORP}\\_{1,t,y,r,s,h}}\\
    \forall {r,y,h} \in \Theta\\_{load} \\
        \quad if \quad sw\\\_reserves = 1\\
\end{aligned}
\end{aligned}
$$



Regulation reserve requirement constraint. 1\% of load + 0.5\% of wind generation + 0.3\% of solar capacity required:


$$
\begin{aligned}
\begin{aligned}
    0.01 \times LOAD\\_{r,y,h}\\
    + 0.005 \times \sum\\_{{t^w,s} \in \theta^{windor}\\_{y,r,h}}{\mathbf{GEN}\\_{t^w,y,r,s,h}} \\
    + 0.003 \times HW\\_h \times \sum\\_{{t^s,s} \in \theta^{solor}\\_{y,r,h}}{\mathbf{CAP^{tot}}\\_{r,MHS\\_h,t^s,s,y}}\\
    \leq
    \sum\\_{{t,s} \in \theta^{opres}\\_{2,r,y,h}}{\mathbf{ORP}\\_{2,t,y,r,s,h}}\\
    \forall {r,y,h} \in \Theta\\_{load} \\
        \quad if \quad sw\\\_reserves = 1\\
\end{aligned}
\end{aligned}
$$




Flexibility reserve requirement constraint. 10\% of wind generation + 4\% of solar capacity required:


$$
\begin{aligned}
\begin{aligned}
    0.1 \times \sum\\_{{t^w,s} \in \theta^{windor}\\_{y,r,h}}{\mathbf{GEN}\\_{t^w,y,r,s,h}} \\
    + 0.04 \times HW\\_h \times \sum\\_{{t^s,s} \in \theta^{solor}\\_{y,r,h}}{\mathbf{CAP^{tot}}\\_{r,MHS\\_h,t^s,s,y}}\\
    \leq
    \sum\\_{{t,s} \in \theta^{opres}\\_{3,r,y,h}}{\mathbf{ORP}\\_{3,t,y,r,s,h}}\\
    \forall {r,y,h} \in \Theta\\_{load} \\
        \quad if \quad sw\\\_reserves = 1\\
\end{aligned}
\end{aligned}
$$



Operating reserve procurement upper bound:


$$
\begin{aligned}
\begin{aligned}
    \mathbf{ORP}\\_{o,t,y,r,s,h}
    \leq
    RTUB\\_{o,t} \times HW\\_h \times
    \mathbf{CAP^{tot}}\\_{r,MHS\\_h,t^s,s,y}\\
    \forall {o,t,y,r,s,h} \in \Theta\\_{proc} \\
    \quad if \quad sw\\\_reserves = 1\\
\end{aligned}
\end{aligned}
$$




## Code Documentation

[Code Documentation](/docs/README.md)