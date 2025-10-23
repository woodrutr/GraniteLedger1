# Output

## Introduction

This directory contains the main outputs for each model run. Outputs are organized by solve mode (standalone, iterative, or unified solves) and a deterministic folder name that combines the selected mode with a hash of the configuration used for that run. When a folder with the same name already exists, a numeric suffix (for example, `_01`) is appended automatically to avoid overwriting previous results.

Each output directory should contain a run log. The run log provides basic information on the status of the model solves and any additional information that may be useful for debugging model results. In addition, each model run should produce at least one graphic that provides minimum details on the success of the model solve. Note that the prototype only contains basic output visualization capabilities. This is because the prototype is not meant to include any capabilities of producing results that need to be analyzed. That said, in order to aid with interpreting results, a results viewer was developed. The results viewer script currently only allows for comparison of electricity module results and is for demonstration purposes only. The intent is to build out this framework with the next version of this prototype. 

## Results Viewer

The results viewer is a python code that utilizes dash from plotly to create an interactive dashboard that can visualize results from multiple runs. When running the viewer from the output folder, a url "http://127.0.0.1:8050/" can be opened in a browser to display the dashboard. 

The viewer is setup to only view electricity related results right now, such as generation, capacity, and trade. After each electricity run, a separate electricity folder will output results within the deterministically named run directory (for example, `gs-combo_a1b2c3d4`). The viewer python code can go through each of the run folders and append the variables for the charts. The [tech_colors.csv](/output/tech_colors.csv) controls the mapping of technology types to colors.

The [open_viewer.bat](/output/open_viewer.bat) is a batch file that can use used to open the viewer which opens the url, activates the python environment from conda, and run the python code. The anaconda prompt should be accessible on the laptop and conda env should be named "bsky" for the batch file to work. After the url opens in the default browser, it will take a moment to refresh for dash to setup the dashboard. If the batch file does not execute, users may need to edit the file to point to their local version of the anaconda activate file (e.g., C:\ProgramData\anaconda3\Scripts\activate.bat).

