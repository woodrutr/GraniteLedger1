@echo off
REM Open Anaconda prompt
call "C:\Program Files\Anaconda3\condabin\activate.bat"

REM navigate to project directory 
cd %CD% && conda activate sphinx_bsky && sphinx-apidoc -f -o ../docs/source/ ../src/ && sphinx-build -b html source/ build/html  > run_log/sphinx_build_html_log.txt 2>&1 && sphinx-build -M markdown source build > run_log/sphinx_build_mkdn_log.txt 2>&1  && sphinx-build -b latex source build/latex_pdf > run_log/sphinx_build_pdf_log.txt 2>&1 

