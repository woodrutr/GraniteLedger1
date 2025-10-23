@echo off
REM Launch the BlueSky Streamlit interface
REM Optional: uncomment the following line if you need to initialize conda manually
REM call "C:\Program Files\Anaconda3\condabin\activate.bat"

cd %cd% && conda activate bsky && streamlit run app.py
