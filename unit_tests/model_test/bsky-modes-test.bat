@echo off

REM Open Anaconda prompt
call "C:\Program Files\Anaconda3\condabin\activate.bat"


REM test all BlueSky modes
cd "D:\Users\ARH\gh-repos-n\bluesky_prototype" && conda activate bsky && python main.py > unit_tests/model_test/mode_test_log/main.txt 2>&1 && python main.py --mode elec > unit_tests/model_test/mode_test_log/elec.txt 2>&1 && python main.py --mode h2 > unit_tests/model_test/mode_test_log/h2.txt 2>&1 && python main.py --mode residential > unit_tests/model_test/mode_test_log/residential.txt 2>&1 && python main.py --mode unified-combo > unit_tests/model_test/mode_test_log/unified-combo.txt 2>&1 && python main.py --mode standalone > unit_tests/model_test/mode_test_log/standalone.txt 2>&1 && conda deactivate







