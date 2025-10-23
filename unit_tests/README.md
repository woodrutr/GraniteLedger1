# Unit Tests

This directory contains unit tests for the model. These unit tests are currently not comprehensive and are for demonstration purposes only. The intent is to build out this framework with the next version of this prototype.  

Tests are split up by module in subfolders but often test the interfaces between models too. 

Instructions to run the unit tests: 

In VSCode, open the project. Click the graduated flask icon ("Testing") on the left-bottom side of the menu. Configure testing using `pytest` and select the `unit_tests` directory. Run tests by clicking the "run tests" button on the top-left. If all tests pass, each test will show a green check mark. 

The purpose of unit tests is to test simple, standalone pieces of the code to ensure the code is working as intended; we expect all unit tests to pass. 

Pytest tests are governed by the [pyproject.toml](/pyproject.toml) configuration file settings, and will create a unified log file for each run.

