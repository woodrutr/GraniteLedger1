# BlueSky Model Documentation

This folder contains code documentation for the BlueSky Prototype, which is documented using [Sphinx](https://www.sphinx-doc.org/). Sphinx allows us to automatically generate documentation from the codebase and provides outputs in various formats, including HTML and Markdown.

- [BlueSky Model Documentation](#bluesky-model-documentation)
    - [Overview](#overview)
    - [How to Access the Documentation](#how-to-access-the-documentation)
        - [HTML Output](#html-output)
        - [Alternatively:](#alternatively)
        - [Markdown Output](#markdown-output)
            - [On GitLab](#on-gitlab)
            - [Locally in VS-Code](#locally-in-vs-code)
        - [PDF documentation](#pdf-documentation)
    - [Work in Progress and Feedback](#work-in-progress-and-feedback)

## Overview

The energy model consists of several packages and subpackages that are organized into functional sections. The primary goal of the documentation is to help users understand how the different components fit together and how they can be used.

The documentation is generated using Sphinx, and the output is available in both HTML and Markdown formats. The documentation automatically captures docstrings from the code, organizes modules and packages, and provides references to each class, function, and method.


## How to Access the Documentation

In general, the **HTML** code documentation provided in the **`docs`** folder is the **BEST** method for exploring BlueSky code documentation. Below are instructions for viewing the html and markdown code documentation outputs.  


### HTML Output

1. Clone the BlueSky repository. Instructions can be found [here](https://git.eia.gov/oea/nextgen-bluesky/bluesky_prototype/-/wikis/Clone-a-repo-using-VS-code)

2. From the cloned **bluesky_prototype** directory, open the **`docs`** folder and *click* on the following file: **`html_documentation.bat`**

3. This should open the code documentation locally using your web browser. 

### Alternatively:

1. From the cloned **bluesky_prototype** directory, Navigate to the following: `docs\build\html`

2. *Click* on file called `index`. This should open a webpage and allow you to navigate and click on links to the code documentation. 


### Markdown Output

#### On GitLab

Markdown [Table of Contents](build/markdown/index.md)

#### Locally in VS-Code

1. From the cloned **bluesky_prototype** directory in VS-Code, Navigate to the following: `docs\build\markdown`

2. *Click* on file called `index`. Using the markdown preview functionality of VS-Code, you should be able to follow the markdown Table of contents list.  

### PDF Documentation

Use the PDF link on the Wiki homepage to access the latest version of code documentation. 
[PDF documentation](build/blueskyprototypemodel.pdf)

## Work in Progress and Feedback

Please note that this documentation is a **work in progress**. As the model evolves, additional details and sections will be added. The documents generated (html, markdown, pdf) are not meant for external publication at this time.

We welcome any feedback, suggestions, or contributions that could help improve the clarity and usefulness of the documentation. If you encounter any gaps or areas that need further explanation, feel free to open an issue or submit a pull request to enhance the documentation.

## Emissions reporting pipeline

The regional emissions outputs produced by the dispatch engine are normalised into a DataFrame stored on the engine result object as `outputs["emissions_by_region"]`. This frame now always contains the columns `year`, `region`, `region_label`, and `emissions_tons`, and it is propagated through the run payload (`current_run_payload`) as well as the reporting layer so the GUI can render the tables and charts consistently.

Downstream utilities should treat this table as the canonical source of regional emissions. When adding new reporting features ensure that the DataFrame is passed forward unchanged (aside from optional filtering) so that both the CSV exports and the on-screen summaries remain in sync.


## Analyst orientation: Transmission and nodal balance

GraniteLedger's transmission stack relies on the `dispatch.interface.InterfaceSpec` contract to keep nodal balance records consistent across the dispatch engine and annual aggregator. Each interface definition normalises its `from_region`/`to_region` identifiers so that the engine emits canonical region keys even when the raw transmission CSVs use legacy spellings or mixed case labels.

* When a dispatch run omits certain regions, the engine backfills zero self-flows and zeroed price/emission rows for every `InterfaceSpec` region so analysts can quickly spot absent trades without reconciling list lengths.
* Interface metadata enforces directionality and capacity across the canonical IDs, which lets the run loop sum inflows/outflows and nodal balances confidently regardless of original label formatting.
* Downstream audits and CSV exports consume the canonical region keys surfaced by `InterfaceSpec`, ensuring that multi-region diagnostics, peak-demand checks, and transmission summaries align with the normalized registry used throughout the model.


## Engine input vector expectations

The dispatch solver expects each forecast parameter to arrive as a fully expanded time-series vector (for example, an hourly profile spanning the modeled horizon). When upstream data pipelines only provide annual totals, normalize those annual values into the engine's required granularity before setting the forecast on the engine. A typical workflow looks like:

```python
annual_load = df.loc[
    (df["year"] == selected_year)
    & (df["iso_zone"] == zone),
    "load_mwh",
].values[0]
hourly_profile = base_profile * (annual_load / base_profile.sum())
engine.set_param(zone, "forecast", hourly_profile)
```

In this pattern, `base_profile` is a normalized (0–1) reference shape whose sum equals one. Scaling the profile by the annual load preserves the original shape while ensuring the expanded vector aggregates back to the supplied annual total. Always confirm the resulting series length matches the dispatch step resolution (hourly, sub-hourly, etc.) required by the solver before passing it into `engine.set_param`.


