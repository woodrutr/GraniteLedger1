# Load Forecast Methodology

The following methodology was used to generate the annual load forecasts for the Southwest Power Pool (SPP) sub-regions from 2025 through 2050.

## Data Source

The primary source for this forecast is the report "Future Load Scenarios for Southwest Power Pool" published by Evolved Energy Research in September 2024. Specifically, the total annual electricity demand projections were extracted from Table 5: *SPP Electricity Load Metrics across Scenarios*.

The "Baseline" scenario was used, and the total system load was calculated by summing the annual demand (TWh) from the "Buildings & Industry," "Data Centers," and "Transportation" subsectors.

## Apportionment Methodology

The source report provides load data for the entire SPP territory and does not offer a sub-regional breakdown for the required modeling zones (SPP_MO, SPP_AR, SPP_OK, SPP_LA).

To address this, the total SPP load was apportioned across the four zones using state population as a proxy. The latest available 2023 US Census Bureau estimates for the corresponding states were used to calculate percentage weights:

- Missouri (MO): ~6.2M people (34.6%)
- Arkansas (AR): ~3.1M people (17.3%)
- Oklahoma (OK): ~4.0M people (22.4%)
- Louisiana (LA): ~4.6M people (25.7%)

> **Disclaimer:** This method uses the population of the entire state as a proxy and is therefore an approximation, as the SPP zones do not perfectly align with state borders.

## Interpolation

The source data provided forecast values for specific years (2023, 2034, 2050). To create a complete annual forecast, linear interpolation was applied between these data points. This process calculates intermediate yearly values by assuming a constant rate of change between the known points, effectively drawing a straight line of load growth between them.
