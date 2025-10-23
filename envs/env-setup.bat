@echo off

REM Locate and Open Anaconda prompt: EXAMPLE: "C:\Users\arh\anaconda3\condabin\activate.bat"
REM call "C:\PATH-TO-ANACONDA\user\anaconda3\condabin\activate.bat"

REM Navigate to project directory and create new environment using existing yml file: EXAMPLE: "C:\Users\arh\OneDrive - Energy Information Administration\gh_repos_n\bluesky_prototype"
cd %CD% && conda env create -f envs/conda_env.yml && conda activate bsky 


